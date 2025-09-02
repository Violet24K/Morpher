from prompt_graph.pretrain import GraphCL, SimGRACE
from prompt_graph.utils import seed_everything
from prompt_graph.utils import mkdir, get_args
import os


args = get_args()
seed_everything(args.seed)

if not os.path.exists('./pre_trained_gnn/'):
    mkdir('./pre_trained_gnn/')

if __name__ == '__main__':
    
    if args.task == 'SimGRACE':
        pt = SimGRACE(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device)
    if args.task == 'GraphCL':
        pt = GraphCL(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device, transfer=args.transfer, feature_dim=args.feature_dim)
    pt.pretrain(batch_size=args.batch_size, lr=args.lr, decay=args.decay, epochs=args.epochs)

