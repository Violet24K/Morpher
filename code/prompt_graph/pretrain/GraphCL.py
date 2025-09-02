import torch
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from random import shuffle
import random
from prompt_graph.utils import mkdir, graph_views
from prompt_graph.data import load4node, load4graph, NodePretrain
from torch.optim import Adam
import time
from.base import PreTrain
from torch_geometric.transforms import SVDFeatureReduction
from sklearn.decomposition import PCA, TruncatedSVD
import pdb

class GraphCL(PreTrain):
    def __init__(self, *args, hid_dim = 16, **kwargs):    # hid_dim=16
        super().__init__(*args, **kwargs)
        self.load_graph_data()
        self.initialize_gnn(self.input_dim, hid_dim)  
        self.hid_dim = hid_dim 
        self.projection_head = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Linear(hid_dim, hid_dim)).to(self.device)
    def load_graph_data(self):
        if self.dataset_name in ['PubMed', 'CiteSeer', 'Cora','Computers', 'Photo', 'Reddit', 'WikiCS', 'Flickr']:
            self.graph_list, self.input_dim = NodePretrain(dataname = self.dataset_name, num_parts=200, transfer=self.transfer, feature_dim=self.feature_dim)
        else:
            self.input_dim, _, _, _, _, self.graph_list= load4graph(self.dataset_name)

        if self.transfer:
            feature_reduction = SVDFeatureReduction(out_channels=self.feature_dim)
            # pca = PCA(n_components=1433, svd_solver='randomized')
            svd = TruncatedSVD(n_components=self.feature_dim)
            # X_reduced = pca.fit_transform(X)
            if self.input_dim > self.feature_dim:
                # print("===Feature reduction from {} to {}".format(self.input_dim, self.feature_dim))
                # self.graph_list = self.graph_list[:10]
                # for g in self.graph_list:
                #     # g = feature_reduction(g.to(self.device))
                #     # svd.fit_transform(g.x)
                #     try:
                #         g.x = torch.tensor(svd.fit_transform(g.x))
                #     except:
                #         g.x = torch.zeros((g.num_nodes, self.feature_dim))
                # self.input_dim = self.feature_dim
                # for g in self.graph_list:
                #     if g.x.shape[1] < self.feature_dim:
                #         g.x = torch.cat([g.x, torch.zeros(g.num_nodes, self.feature_dim - g.x.shape[1])], dim=1)
                pass
                
            elif self.input_dim < self.feature_dim:
                # pad zero to all the graphs in self.graph_list
                print("===Feature padding from {} to {}".format(self.input_dim, self.feature_dim))
                for g in self.graph_list:
                    g.x = torch.cat([g.x, torch.zeros(g.num_nodes, self.feature_dim - self.input_dim)], dim=1)
                self.input_dim = self.feature_dim

    def get_loader(self, graph_list, batch_size,aug1=None, aug2=None, aug_ratio=None):

        if len(graph_list) % batch_size == 1:
            raise KeyError(
                "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in GraphCL!")
        
        shuffle(graph_list)
        if aug1 is None:
            aug1 = random.sample(['dropN', 'permE', 'maskN'], k=1)
        if aug2 is None:
            aug2 = random.sample(['dropN', 'permE', 'maskN'], k=1)
        if aug_ratio is None:
            aug_ratio = random.randint(1, 3) * 1.0 / 10  # 0.1,0.2,0.3

        print("===graph views: {} and {} with aug_ratio: {}".format(aug1, aug2, aug_ratio))

        view_list_1 = []
        view_list_2 = []
        for g in graph_list:
            view_g = graph_views(data=g, aug=aug1, aug_ratio=aug_ratio)
            view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
            view_list_1.append(view_g)
            view_g = graph_views(data=g, aug=aug2, aug_ratio=aug_ratio)
            view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
            view_list_2.append(view_g)

        loader1 = DataLoader(view_list_1, batch_size=batch_size, shuffle=False,
                                num_workers=1)  # you must set shuffle=False !
        loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False,
                                num_workers=1)  # you must set shuffle=False !

        return loader1, loader2
    
    def forward_cl(self, x, edge_index, batch):
        x = self.gnn(x, edge_index, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = - torch.log(pos_sim / (sim_matrix.sum(dim=1) + 1e-4)).mean()
        # loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
        # loss = - torch.log(loss).mean() 
        return loss

    def train_graphcl(self, loader1, loader2, optimizer):
        self.train()
        train_loss_accum = 0
        total_step = 0
        for step, batch in enumerate(zip(loader1, loader2)):
            batch1, batch2 = batch
            optimizer.zero_grad()
            x1 = self.forward_cl(batch1.x.to(self.device), batch1.edge_index.to(self.device), batch1.batch.to(self.device))
            x2 = self.forward_cl(batch2.x.to(self.device), batch2.edge_index.to(self.device), batch2.batch.to(self.device))
            loss = self.loss_cl(x1, x2)

            loss.backward()
            optimizer.step()

            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / total_step

    def pretrain(self, batch_size=10, aug1='dropN', aug2="permE", aug_ratio=None, lr=0.01, decay=0.0001, epochs=100):
        
        self.to(self.device)
        loader1, loader2 = self.get_loader(self.graph_list, batch_size, aug1=aug1, aug2=aug2)
        print('start training {} | {} | {}...'.format(self.dataset_name, 'GraphCL', self.gnn_type))
        optimizer = Adam(self.parameters(), lr=lr, weight_decay=decay)

        train_loss_min = 1000000
        for epoch in range(1, epochs + 1):  # 1..100
            train_loss = self.train_graphcl(loader1, loader2, optimizer)

            print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, epochs, train_loss))

            if train_loss_min > train_loss:
                train_loss_min = train_loss
                if not self.transfer:
                    torch.save(self.gnn.state_dict(),
                            "./pre_trained_gnn/{}.{}.{}.{}.pth".format(self.dataset_name, 'GraphCL', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))
                    print("+++model saved ! {}.{}.{}.{}.pth".format(self.dataset_name, 'GraphCL', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))
                else:
                    torch.save(self.gnn.state_dict(),
                            "./pre_trained_gnn/transfer_{}.{}.{}.{}_featuredim{}.pth".format(self.dataset_name, 'GraphCL', self.gnn_type, str(self.hid_dim) + 'hidden_dim', self.feature_dim))
                    print("+++model saved ! transfer_{}.{}.{}.{}_featuredim{}.pth".format(self.dataset_name, 'GraphCL', self.gnn_type, str(self.hid_dim) + 'hidden_dim', self.feature_dim))
