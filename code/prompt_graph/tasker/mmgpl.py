import torch
from torch import optim
import torchmetrics
from torch_geometric.loader import DataLoader
from torch.nn import functional as F
from prompt_graph.model import GAT, GCN, GCov, GIN, GraphSAGE, GraphTransformer
from prompt_graph.data import load4graph, load4node, split_induced_graphs, load4edge
from prompt_graph.prompt import MorpherGraphPrompt, MorpherTextPrompt
from prompt_graph.utils import seed_torch
from transformers import PreTrainedModel, PreTrainedTokenizer
import time
import os.path as osp
import pickle
import os

from torch_geometric.transforms import SVDFeatureReduction

SAVE_PROJ = False

class MMGPL:
    def __init__(self, pre_train_model_path=None, pretrain_method=None, gnn_type='GCN', hid_dim = 128, num_layer = 2, dataset_name='MUTAG', prompt_type='Morpher', epochs=10, shot_num=10, 
                batch_size=16, prompt_graph_token_num = 10, tokenizer: PreTrainedTokenizer = None, llm: PreTrainedModel = None, device : int = 1,
                projector_lr=0.01, projector_weight_decay=0.1, projector_tune_lr=0.001, projector_tune_weight_decay=0.1,
                pg_lr=0.001, pg_weight_decay=0.001, text_prompt_lr=0.001, text_prompt_weight_decay=0.001,
                projector_dropout_ratio=0.2, temperature=2.0, text_prompt_start_vocab='a graph with property',
                projector_epochs=2001, projector_train_eval_diff_threshold=0.1, projector_train_modular=100, projector_tune_epochs=50, prompt_tune_epochs=50,
                train_val_acc_diff_tol=0.0, val_acc_threshold=1.0, warmup_epochs=0, random_seed=42,
                task = 'graph', test_batch_size=20, source=None, feature_dim=None):
        
        # pretrained GNN, LLM and experiment settings hyperparameters
        self.pre_train_model_path = pre_train_model_path
        self.pretrain_method = pretrain_method
        self.device = torch.device('cuda:'+ str(device) if torch.cuda.is_available() else 'cpu')
        self.hid_dim = hid_dim
        self.num_layer = num_layer
        self.dataset_name = dataset_name
        self.shot_num = shot_num
        self.batch_size = batch_size
        self.gnn_type = gnn_type
        self.prompt_type = prompt_type
        self.epochs = epochs
        self.prompt_graph_token_num = prompt_graph_token_num
        self.tokenizer = tokenizer
        self.llm = llm
        self.llm_dim = llm.config.hidden_size

        self.projector_epochs = projector_epochs
        self.projector_train_eval_diff_threshold = projector_train_eval_diff_threshold
        self.projector_train_modular = projector_train_modular
        self.projector_tune_epochs = projector_tune_epochs
        self.prompt_tune_epochs = prompt_tune_epochs

        # optimization hyperparameters
        self.projector_lr = projector_lr
        self.projector_weight_decay = projector_weight_decay
        self.projector_tune_lr = projector_tune_lr
        self.projector_tune_weight_decay = projector_tune_weight_decay
        self.pg_lr = pg_lr
        self.pg_weight_decay = pg_weight_decay
        self.text_prompt_lr = text_prompt_lr
        self.text_prompt_weight_decay = text_prompt_weight_decay
        self.projector_dropout_ratio = projector_dropout_ratio
        self.temperature = temperature
        self.text_prompt_start_vocab = text_prompt_start_vocab
        self.initialize_lossfn()

        # picking best model
        self.train_val_acc_diff_tol = train_val_acc_diff_tol
        self.val_acc_threshold = val_acc_threshold
        self.warmup_epochs = warmup_epochs
        self.random_seed = random_seed

        # multi-task
        self.task = task
        self.test_batch_size = test_batch_size

        # transfer
        self.source = source
        self.feature_dim = feature_dim

        self.load_data()
        self.initialize_gnn()
        self.initialize_prompt()
        self.projector = torch.nn.Sequential(torch.nn.Dropout(self.projector_dropout_ratio), torch.nn.Linear(self.hid_dim, self.llm_dim), torch.nn.Tanh()).to(self.device)
        self.answering =  torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
                                            torch.nn.Softmax(dim=1)).to(self.device)
        self.initialize_optimizer()


    def meannormalize_labeltotextemb(self):
        # calculate the mean and substract it from the embeddings
        mean_emb = torch.stack([self.label_to_text_emb[label].clone() for label in self.label_to_text_emb]).mean(dim=0)
        for label in self.label_to_text_emb:
            self.label_to_text_emb[label] = self.label_to_text_emb[label] - mean_emb


    def load_data(self):
        if self.dataset_name in ['MUTAG', 'ENZYMES', 'PROTEINS', 'MSRC_21', 'MSRC_21C']:
            self.input_dim, self.output_dim, self.train_dataset, self.test_dataset, self.val_dataset, _= load4graph(self.dataset_name, self.shot_num)
        elif self.dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
            if self.task == 'node':
                self.data, self.dataset = load4node(self.dataset_name, shot_num = self.shot_num)
                self.data.to('cpu')
                self.input_dim = self.dataset.num_features
                self.output_dim = self.dataset.num_classes
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
                else:
                    print('Begin split_induced_graphs.')
                    split_induced_graphs(self.dataset_name, self.data, smallest_size=10, largest_size=30)
                    with open(file_path, 'rb') as f:
                            graphs_dict = pickle.load(f)
                    self.train_dataset = graphs_dict['train_graphs']
                    self.test_dataset = graphs_dict['test_graphs']
                    self.val_dataset = graphs_dict['val_graphs']

            elif self.task == 'edge':
                self.input_dim, self.output_dim, self.train_dataset, self.test_dataset, self.val_dataset = load4edge(self.dataset_name, self.shot_num)
            elif self.task == 'transfer':
                self.data, self.dataset = load4node(self.dataset_name, shot_num = self.shot_num, transfer=True, feature_dim=self.feature_dim)
                self.data.to('cpu')
                self.input_dim = self.dataset.num_features
                self.output_dim = self.dataset.num_classes
                file_path = './data/induced_graph/' + 'transfer_' + self.dataset_name + '_induced_graph.pkl'
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

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        if self.task == 'transfer':
            # # turn all the features to 3 dimensions using PCA
            # feature_reduction = SVDFeatureReduction(3)
            # print("===Feature reduction from {} to 3".format(self.input_dim))
            # self.train_dataset = [feature_reduction(dataset) for dataset in self.train_dataset]
            # self.test_dataset = [feature_reduction(dataset) for dataset in self.test_dataset]
            # self.val_dataset = [feature_reduction(dataset) for dataset in self.val_dataset]
            self.input_dim = self.feature_dim
            pass
        
        if self.dataset_name == 'MUTAG':
            self.label_to_text = {0: 'non-mutagenic on Salmonella typhimurium', 1: 'mutagenic on Salmonella typhimurium'}

        if self.dataset_name == 'ENZYMES':
            # for enzymes dataset, the labels are Enzyme Commission top level enzyme classes (EC classes)
            self.label_to_text = {0: 'oxidoreductases', 1: 'transferases', 2: 'hydrolases', 3: 'lyases', 4: 'isomerases', 5: 'ligases'}

        if self.dataset_name == 'PROTEINS':
            self.label_to_text = {0: 'enzyme', 1: 'non-enzyme'}

        if self.dataset_name == 'MSRC_21':
            self.label_to_text = {1: 'building', 2: 'grass', 3: 'tree', 4: 'cow', 5: 'sheep', 6: 'sky', 7: 'airplane', 8: 'water', 9: 'face', 10: 'car', 
                                11: 'bicycle', 12: 'flower', 13: 'sign', 14: 'bird', 15: 'book', 16: 'chair', 17: 'road', 18: 'cat', 19: 'dog', 20: 'body', 21: 'boat'}
            
        if self.dataset_name == 'MSRC_21C':
            self.label_to_text = {1: 'building', 2: 'grass', 3: 'tree', 4: 'cow', 5: 'sheep', 6: 'sky', 7: 'airplane', 8: 'water', 9: 'face', 10: 'car', 
                                11: 'bicycle', 12: 'flower', 13: 'sign', 14: 'bird', 15: 'book', 16: 'chair', 17: 'road', 18: 'cat', 19: 'dog', 20: 'body', 21: 'boat'}
            
        if self.dataset_name == 'Cora':
            if self.task == 'node' or self.task == 'transfer':
                self.label_to_text = {0: 'case based', 1: 'genetic algorithms', 2: 'neural networks', 3: 'probabilistic methods', 4: 'reinforcement learning', 5: 'rule learning', 6: 'theory'}
            else: # edge task
                self.label_to_text = {0: 'not connected', 1: 'connected'}


        if self.dataset_name == 'CiteSeer':
            if self.task == 'node' or self.task == 'transfer':
                self.label_to_text = {0: 'Agents', 1: 'AI', 2: 'DB', 3: 'IR', 4: 'ML', 5: 'HCI'}
            else:
                self.label_to_text = {0: 'not connected', 1: 'connected'}


        if self.dataset_name == 'PubMed':
            if self.task == 'node' or self.task == 'transfer':
                # PubMed dataset has 3 classes
                self.label_to_text = {0: 'Diabetes Mellitus Experimental', 1: 'Diabetes Mellitus Type 1', 2: 'Diabetes Mellitus Type 2'}


        self.tokenized_label_to_text = {i: self.tokenizer.encode(self.label_to_text[i], return_tensors='pt').to(self.device) for i in self.label_to_text}
        self.label_to_text_emb = {i: self.llm(self.tokenized_label_to_text[i])[0].mean(dim=1).squeeze() for i in self.label_to_text}
        self.meannormalize_labeltotextemb()

    def initialize_lossfn(self):
        # projector criterion is the norm of the difference between the projected embeddings and the text prompt embeddings
        self.projector_criterion = torch.nn.MSELoss()
        # self.critierion is the similarity loss between the projected embeddings and the text prompt embeddings, similar to CLIP loss.
        # projected embeddings and text prompt embeddings are both vectors of size (batch_size, llm_dim). First compute the cosine similarity 
        # between the two embeddings, then compute softmax
        self.criterion = self.contrastive_loss_with_label
    

    def contrastive_loss_with_label(self, graph_embeddings, text_embeddings_of_y, y):
        # normalize embeddings
        # contrastive loss
        logits = (graph_embeddings @ text_embeddings_of_y.T) / self.temperature

        exp_logits = torch.exp(logits)
        sum_exp_logits = exp_logits.sum(dim=1)

        # logits is in shape (batch_size, num_classes). for each row, the value at the index of the true class is the logit for that class. 
        # retrieve the logit for the true class
        true_class_logits = logits[torch.arange(logits.shape[0]), y]

        loss = true_class_logits - torch.log(sum_exp_logits)

        return -loss.mean()


    def initialize_gnn(self):
        if self.gnn_type == 'GAT':
            self.gnn = GAT(input_dim=self.input_dim, out_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GCN':
            self.gnn = GCN(input_dim=self.input_dim, out_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GraphSAGE':
            self.gnn = GraphSAGE(input_dim=self.input_dim, out_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GIN':
            self.gnn = GIN(input_dim=self.input_dim, out_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GCov':
            self.gnn = GCov(input_dim=self.input_dim, out_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GraphTransformer':
            self.gnn = GraphTransformer(input_dim=self.input_dim, out_dim=self.hid_dim, num_layer=self.num_layer)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        self.gnn.to(self.device)

        if self.pre_train_model_path != 'None':
            if self.gnn_type not in self.pre_train_model_path :
                raise ValueError(f"the Downstream gnn '{self.gnn_type}' does not match the pre-train model")
            if self.dataset_name not in self.pre_train_model_path :
                # raise ValueError(f"the Downstream dataset '{self.dataset_name}' does not match the pre-train dataset")
                print(f"Warning: the Downstream dataset '{self.dataset_name}' does not match the pre-train dataset")

            self.gnn.load_state_dict(torch.load(self.pre_train_model_path, map_location=self.device))
            print("Successfully loaded pre-trained weights!")

    
    def initialize_prompt(self):
        self.prompt = MorpherGraphPrompt(token_dim=self.input_dim, token_num=self.prompt_graph_token_num).to(self.device)
        self.start_vocab = self.text_prompt_start_vocab
        self.start_vocab_tokens = self.tokenizer.encode(self.start_vocab, return_tensors='pt').to(self.device)
        start_vocab_emb = self.llm(self.start_vocab_tokens)[0].squeeze()
        
        self.n_tokens = len(self.start_vocab_tokens[0])
        self.text_prompt = MorpherTextPrompt(self.llm.get_input_embeddings(), n_tokens=self.n_tokens, start_vocab_emb=start_vocab_emb).to(self.device)


    def initialize_optimizer(self):
        self.projector_opi = optim.Adam(filter(lambda p: p.requires_grad, self.projector.parameters()), lr=self.projector_lr, weight_decay=self.projector_weight_decay)
        self.projector_tune_opi = optim.Adam(filter(lambda p: p.requires_grad, self.projector.parameters()), lr=self.projector_tune_lr, weight_decay= self.projector_tune_weight_decay)
        self.pg_opi = optim.Adam(filter(lambda p: p.requires_grad, self.prompt.parameters()), lr=self.pg_lr, weight_decay= self.pg_weight_decay)
        self.text_prompt_opi = optim.Adam(filter(lambda p: p.requires_grad, self.text_prompt.parameters()), lr=self.text_prompt_lr, weight_decay= self.text_prompt_weight_decay)


    def eval_projector(self, eval_loader: DataLoader):
        self.projector.eval()
        total_loss = 0.0
        for batch in eval_loader:
            batch = batch.to(self.device)

            out = self.gnn(batch.x, batch.edge_index, batch.batch)
            out = self.projector(out)

            # create the text embeddings from batch.y according to the encoded_label_to_text dictionary
            with torch.no_grad():
                text_emb = torch.stack([self.label_to_text_emb[label].clone() for label in self.label_to_text_emb])

            out = F.normalize(out, p=2, dim=-1)
            text_emb = F.normalize(text_emb, p=2, dim=-1)

            loss = self.criterion(out, text_emb, batch.y)
            total_loss += loss.item()
        return total_loss/len(eval_loader)
                

    def train_projector(self, train_loader: DataLoader, val_loader: DataLoader, projector_epochs):
        best_projector = None
        self.projector.train()
        best_eval_loss = 1000000
        for epoch in range(projector_epochs):
            total_loss = 0.0
            for batch in train_loader:

                batch = batch.to(self.device)

                out = self.gnn(batch.x, batch.edge_index, batch.batch)
                out = self.projector(out)

                # create the text embeddings from batch.y according to the encoded_label_to_text dictionary
                with torch.no_grad():
                    text_emb = torch.stack([self.label_to_text_emb[label].clone() for label in self.label_to_text_emb])

                # row normalization
                out = F.normalize(out, p=2, dim=-1)
                text_emb = F.normalize(text_emb, p=2, dim=-1)

                loss = self.criterion(out, text_emb, batch.y)
                self.projector_opi.zero_grad()
                loss.backward()
                self.projector_opi.step()
                total_loss += loss.item()

            if epoch % self.projector_train_modular == 0:
                print(f"Projector Epoch: {epoch}, Train Loss: {total_loss/len(train_loader)}")
                # pdb.set_trace()
                eval_loss = self.eval_projector(val_loader)
                print(f"Projector Epoch: {epoch}, Eval Loss: {eval_loss}")
                if eval_loss < best_eval_loss and abs(eval_loss - total_loss/len(train_loader)) < self.projector_train_eval_diff_threshold:
                    best_eval_loss = eval_loss
                    print("Checkpointing best projector model...")
                    best_projector = self.projector.state_dict()

        print("Projector training finished! Loading best projector model...")
        self.projector.load_state_dict(best_projector)
                    

    def MorpherTrain(self, train_loader):
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(self.device)
            prompted_graph = self.prompt(batch)
            graph_emb = self.gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            out = self.projector(graph_emb)

            self.llm.set_input_embeddings(self.text_prompt)

            # update the prompted text embeddings
            self.label_to_text_emb = {i: self.llm(self.tokenized_label_to_text[i])[0].mean(dim=1).squeeze() for i in self.label_to_text}
            self.meannormalize_labeltotextemb()

            # text_emb = torch.stack([self.label_to_text_emb[label.item()].clone() for label in batch.y])
            text_emb = torch.stack([self.label_to_text_emb[label].clone() for label in self.label_to_text_emb])
            # similarity-based loss between out and the text embeddings

            # row normalization
            out = F.normalize(out, p=2, dim=-1)
            text_emb = F.normalize(text_emb, p=2, dim=-1)

            loss = self.criterion(out, text_emb, batch.y)

            self.pg_opi.zero_grad()
            self.text_prompt_opi.zero_grad()
            self.projector_tune_opi.zero_grad()
            loss.backward()
            self.pg_opi.step()
            self.text_prompt_opi.step()
            self.projector_tune_opi.step()
            total_loss += loss.item()
        
        return total_loss/len(train_loader)
    

    def MorpherEval(self, eval_loader, num_class, device):
        self.prompt.eval()
        self.text_prompt.eval()
        accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
        macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="weighted").to(device)
        accuracy.reset()
        macro_f1.reset()
        for batch in eval_loader:
            batch = batch.to(self.device)
            prompted_graph = self.prompt(batch)
            graph_emb = self.gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            out = self.projector(graph_emb)

            self.llm.set_input_embeddings(self.text_prompt)
            # update the prompted text embeddings
            self.label_to_text_emb = {i: self.llm(self.tokenized_label_to_text[i])[0].mean(dim=1).squeeze() for i in self.label_to_text}
            self.meannormalize_labeltotextemb()

            text_emb = torch.stack([self.label_to_text_emb[label].clone() for label in self.label_to_text_emb])
            out = F.normalize(out, p=2, dim=-1)
            text_emb = F.normalize(text_emb, p=2, dim=-1)
            sims = out @ text_emb.T
            pred = sims.argmax(dim=1)

            acc = accuracy(pred, batch.y)
            f1 = macro_f1(pred, batch.y)
        acc = accuracy.compute()
        f1 = macro_f1.compute()

        return acc, f1



    def run(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)
        print("prepare data is finished!")
        print("Setting Language Model and GNN to eval mode.")
        self.llm.eval()
        self.gnn.eval()
        self.llm.set_input_embeddings(self.text_prompt)

        for i in self.tokenized_label_to_text:
            self.tokenized_label_to_text[i] = torch.cat([self.start_vocab_tokens, self.tokenized_label_to_text[i]], 1)

        self.label_to_text_emb = {i: self.llm(self.tokenized_label_to_text[i])[0].mean(dim=1).squeeze() for i in self.label_to_text}
        self.meannormalize_labeltotextemb()

        if self.task == 'graph':
            projector_path = osp.join('trained_projector', f'best_projector_{self.dataset_name}_{self.pretrain_method}_{self.gnn_type}.pt')
        elif self.task == 'node' or self.task == 'edge':
            projector_path = osp.join('trained_projector', f'best_projector_{self.dataset_name}_{self.pretrain_method}_{self.gnn_type}_{self.task}.pt')
        elif self.task == 'transfer':
            projector_path = osp.join('trained_projector', f'best_projector_transfer_from_{self.source}_to_{self.dataset_name}_{self.pretrain_method}_{self.gnn_type}_dim_{self.input_dim}.pt')
        if not osp.exists(projector_path) or SAVE_PROJ:
            self.train_projector(train_loader, val_loader, projector_epochs=self.projector_epochs)
            torch.save(self.projector.state_dict(), projector_path)
        else:
            self.projector.load_state_dict(torch.load(projector_path, map_location=self.device))

        self.projector.eval()

        if self.dataset_name != 'MUTAG' or self.pretrain_method != 'GraphCL' or self.gnn_type != 'GCN' or self.task == 'transfer':
            # reset the random seed to 42
            seed_torch(self.random_seed)

        train_losses = []
        train_accs = []
        val_accs = []
        test_accs = []
        test_f1s = []

        for epoch in range(1, self.epochs+1):
            # pdb.set_trace()
            start = time.time()
            for _ in range(self.prompt_tune_epochs):
                self.prompt.train()
                self.text_prompt.train()
                self.projector.eval()
                train_loss = self.MorpherTrain(train_loader)
            for _ in range(self.projector_tune_epochs):
                self.prompt.eval()
                self.text_prompt.eval()
                self.projector.train()
                train_loss = self.MorpherTrain(train_loader)

            print(f"Epoch: {epoch}, Train Loss: {train_loss}, Time: {time.time()-start}")
            train_acc, train_f1 = self.MorpherEval(train_loader, self.output_dim, self.device)
            val_acc, val_f1 = self.MorpherEval(val_loader, self.output_dim, self.device)
            test_acc, test_f1 = self.MorpherEval(test_loader, self.output_dim, self.device)
            print(f"Epoch: {epoch}, Train Acc: {train_acc:.5f}, Val Acc: {val_acc:.5f}, Test Acc: {test_acc:.5f}, Test F1: {test_f1:.5f}")

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            test_accs.append(test_acc)
            test_f1s.append(test_f1)

        self.pick_best_model(train_losses, train_accs, val_accs, test_accs, test_f1s, warmup_epochs=self.warmup_epochs)



    def pick_best_model(self, train_loss: list, train_acc: list, val_acc: list, test_acc: list, test_f1s: list, warmup_epochs=0):
        train_val_acc_diff_tol = self.train_val_acc_diff_tol
        val_acc_threshold = self.val_acc_threshold

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