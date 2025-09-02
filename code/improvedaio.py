from prompt_graph.tasker import GraphTask
from prompt_graph.utils import seed_everything, load_yaml
from prompt_graph.utils import  get_args
import os.path as osp

if __name__ == '__main__':
    args = get_args()
    config = load_yaml(osp.join('configurations_improvedAIO', args.dataset_name + '_' + args.pretrain_method + '_' + args.gnn_type + '.yaml'))
    seed_everything(config.seed)

    if args.pre_train_model_path == 'None':
        args.pre_train_model_path = osp.join('pre_trained_gnn', args.dataset_name + '.' + args.pretrain_method + '.' + args.gnn_type + '.' + str(config.hid_dim) + 'hidden_dim.pth')

    tasker = GraphTask(pre_train_model_path = args.pre_train_model_path, 
                    dataset_name = args.dataset_name, num_layer = config.num_layer, gnn_type = args.gnn_type, 
                    prompt_type = args.prompt_type, epochs = config.epochs, shot_num = config.shot_num, device = args.device, 
                    config=config, task=args.task)
        
    tasker.run()