import torch
from prompt_graph.data import load4graph, load4node, split_induced_graphs, load4edge
from torch_geometric.loader import DataLoader
from .task import BaseTask
from prompt_graph.utils import center_embedding, Gprompt_tuning_loss
from prompt_graph.evaluation import ImprovedAllInOneEva, ImprovedAllInOneEvaOnlyAnswering
import time
import os
import os.path as osp
import pickle
from torch_geometric.transforms import SVDFeatureReduction

class GraphTask(BaseTask):
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, **kwargs)
        self.load_data()
        self.initialize_gnn()
        self.initialize_prompt()
        self.answering =  torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
                                            torch.nn.Softmax(dim=1)).to(self.device)
        self.initialize_optimizer()

    def load_data(self):
        if self.dataset_name in ['MUTAG', 'ENZYMES','PROTEINS', 'MSRC_21', 'MSRC_21C']:
            self.input_dim, self.output_dim, self.train_dataset, self.test_dataset, self.val_dataset, _= load4graph(self.dataset_name, self.shot_num)
        elif self.dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
            if self.task == 'node' or self.transfer:
                self.data, self.dataset = load4node(self.dataset_name, shot_num = self.shot_num, transfer=self.transfer, feature_dim=400)
                self.data.to('cpu')
                self.input_dim = self.dataset.num_features
                self.output_dim = self.dataset.num_classes
                if self.transfer:
                    file_path = './data/induced_graph/' + 'transfer_' + self.dataset_name + '_induced_graph.pkl'
                    folder_path = './data/induced_graph/'
                else:
                    file_path = './data/induced_graph/' + self.dataset_name + '_induced_graph.pkl'
                    folder_path = './data/induced_graph/'
                # create the folder if not exists
                if not osp.exists(folder_path):
                    os.makedirs(folder_path)
                if osp.exists(file_path):
                    with open(file_path, 'rb') as f:
                            graphs_dict = pickle.load(f)
                    self.train_dataset = graphs_dict['train_graphs']
                    self.test_dataset = graphs_dict['test_graphs']
                    self.val_dataset = graphs_dict['val_graphs']
                    if self.dataset_name == 'PubMed':
                        self.test_dataset = graphs_dict['test_graphs'][:700]
                else:
                    print('Begin split_induced_graphs.')
                    split_induced_graphs(self.dataset_name, self.data, smallest_size=10, largest_size=30, transfer=True)
                    with open(file_path, 'rb') as f:
                            graphs_dict = pickle.load(f)
                    self.train_dataset = graphs_dict['train_graphs']
                    self.test_dataset = graphs_dict['test_graphs']
                    self.val_dataset = graphs_dict['val_graphs']
                    if self.dataset_name == 'PubMed':
                        self.test_dataset = graphs_dict['test_graphs'][:700]
            elif self.task == 'edge':
                self.input_dim, self.output_dim, self.train_dataset, self.test_dataset, self.val_dataset = load4edge(self.dataset_name, self.shot_num)
                


        if self.transfer:
            # turn all the features to 3 dimensions using PCA
            # feature_reduction = SVDFeatureReduction(3)
            # print("===Feature reduction from {} to 3".format(self.input_dim))
            # self.train_dataset = [feature_reduction(dataset.to(self.device)) for dataset in self.train_dataset]
            # self.test_dataset = [feature_reduction(dataset.to(self.device)) for dataset in self.test_dataset]
            # self.val_dataset = [feature_reduction(dataset.to(self.device)) for dataset in self.val_dataset]
            self.input_dim = 400
            pass

    def Train(self, train_loader):
        if self.transfer:
            self.gnn.eval()
        else:
            self.gnn.train()
        total_loss = 0.0 
        for batch in train_loader:  
            self.optimizer.zero_grad() 
            batch = batch.to(self.device)
            out = self.gnn(batch.x, batch.edge_index, batch.batch)
            out = self.answering(out)
            loss = self.criterion(out, batch.y)  
            loss.backward()  
            self.optimizer.step()  
            total_loss += loss.item()  
        return total_loss / len(train_loader)  
        
    def ImprovedAllInOneTrain(self, train_loader):
        #we update answering and prompt alternately.
        
        answer_epoch = self.config.answer_tune_epochs  # 50
        prompt_epoch = self.config.prompt_tune_epochs  # 50
        
        # tune prompt
        self.answering.eval()
        self.prompt.train()
        for epoch in range(1, prompt_epoch + 1):
            pg_loss = self.prompt.Tune(train_loader,  self.gnn, self.answering, self.criterion, self.pg_opi, self.device)

        # tune task head
        self.answering.train()
        self.prompt.eval()
        for epoch in range(1, answer_epoch + 1):
            answer_loss = self.prompt.Tune(train_loader, self.gnn,  self.answering, self.criterion, self.answer_opi, self.device)

        return answer_loss


    
    def ImprovedAllInOneTrainOnlyAnswering(self, train_loader):
        try: 
            answer_epoch = self.config.answer_tune_epochs  # 50
        except:
            answer_epoch = 50  # 50
        self.answering.train()
        self.prompt.eval()
        for epoch in range(1, answer_epoch + 1):
                answer_loss = self.prompt.TuneOnlyAnswering(train_loader, self.gnn,  self.answering, self.criterion, self.answer_opi, self.device)

        return answer_loss


    def run(self):
        train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=False)
        val_loader = DataLoader(self.val_dataset, batch_size=16, shuffle=False)
        print("prepare data is finished!")

        test_acc = 0
        if self.prompt_type == 'ImprovedAIO':
            initial_test_acc, F1 = ImprovedAllInOneEva(test_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)

        if self.prompt_type == 'answering':
            initial_test_acc, F1 = ImprovedAllInOneEvaOnlyAnswering(test_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
        print('Initial Test: ', initial_test_acc)
        test_acc = initial_test_acc

        if self.prompt_type == 'ImprovedAIO':
            val_acc, F1 = ImprovedAllInOneEvaOnlyAnswering(val_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
            print('initial validation of answering before training: ', val_acc)
            loss = self.ImprovedAllInOneTrainOnlyAnswering(train_loader)
            val_acc, F1 = ImprovedAllInOneEvaOnlyAnswering(val_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
            print('Answering function trained successfully. initial validation of answering: ', val_acc)

        best_train_loss = 1e9
        best_f1 = 0
        train_losses = []
        train_accs = []
        val_accs = []
        test_accs = []
        test_f1s = []

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            if self.prompt_type == 'ImprovedAIO':
                loss = self.ImprovedAllInOneTrain(train_loader)
                train_acc, _ = ImprovedAllInOneEva(train_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
                val_acc, _ = ImprovedAllInOneEva(val_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
                test_acc, test_f1 = ImprovedAllInOneEva(test_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
                    
            train_losses.append(loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            test_accs.append(test_acc)
            test_f1s.append(test_f1)
            
            print("Epoch {:03d}/{:03d}  |  Time(s) {:.4f}| Loss {:.5f} | Train Accuracy {:.5f} | val Accuracy {:.5f} | test Accuracy {:.5f} | test Weighted F1 {:.5f}".format(epoch, self.epochs, time.time() - t0, loss, train_acc, val_acc, test_acc, test_f1))
            
        if self.prompt_type == 'None':
            try:
                self.pick_best_model(train_losses, train_accs, val_accs, test_accs, test_f1s, warmup_epochs=self.config.pf_warmup_epochs)
            except:
                self.pick_best_model(train_losses, train_accs, val_accs, test_accs, test_f1s)
        else:
            try:
                self.pick_best_model(train_losses, train_accs, val_accs, test_accs, test_f1s, warmup_epochs=self.config.aio_warmup_epochs)
            except:
                self.pick_best_model(train_losses, train_accs, val_accs, test_accs, test_f1s)
        
        print("Graph Task completed")

        

    def pick_best_model(self, train_loss: list, train_acc: list, val_acc: list, test_acc: list, test_f1s: list, warmup_epochs=0):
        train_val_acc_diff_tol = self.config.train_val_acc_diff_tol
        val_acc_threshold = self.config.val_acc_threshold

        # clear the val_acc for the warmup epochs, the epochs with large train_val_acc_diff (potentially overfitted epochs), and the epochs with val_acc higher than the overfitting threshold
        for i in range(warmup_epochs):
            val_acc[i] = -1
        for i in range(len(val_acc)):
            if abs(train_acc[i] - val_acc[i]) > train_val_acc_diff_tol:
                val_acc[i] = -1
        for i in range(len(val_acc)):
            if val_acc[i] > val_acc_threshold:
                val_acc[i] = -1
        
        # set the epochs with best val_acc to be the best epoch candidate, note that there might be multiple epochs with the same val_acc
        best_epoch_candidate = []
        best_val_acc = -1
        for i in range(len(val_acc)):
            if val_acc[i] > best_val_acc:
                best_val_acc = val_acc[i]
        for i in range(len(val_acc)):
            if val_acc[i] == best_val_acc:
                best_epoch_candidate.append(i)

        # pick the best epoch(s) with the highest train_acc
        best_epoch = best_epoch_candidate[0]
        best_train_acc = train_acc[best_epoch]
        for i in best_epoch_candidate:
            if train_acc[i] > best_train_acc:
                best_epoch = i
                best_train_acc = train_acc[i]
        
        best_epoch_candidate2 = []
        for i in best_epoch_candidate:
            if train_acc[i] == best_train_acc:
                best_epoch_candidate2.append(i)

        best_epoch_candidate = best_epoch_candidate2

        # If still cannot distinguish which epoch is the best, pick the best epoch with smallest train loss
        best_epoch = best_epoch_candidate[0]

        if len(best_epoch_candidate) > 1:
            best_train_loss = train_loss[best_epoch]
            for i in best_epoch_candidate:
                if train_loss[i] < best_train_loss:
                    best_epoch = i
                    best_train_loss = train_loss[i]

        # best_epoch = best_epoch_candidate[0]
        picked_model_test_acc = test_acc[best_epoch]
        
        
        print("Filtered the epochs that seems to be overfitted or underfitted. Picking the best epoch...")
        print(f"Best epoch: {best_epoch + 1}, Test Acc: {picked_model_test_acc}, Test F1: {test_f1s[best_epoch]}")
        
