import torch
from prompt_graph.utils import get_morpher_args, load_yaml
from prompt_graph.utils import seed_torch
from prompt_graph.tasker import MMGPL
from transformers import RobertaModel, RobertaTokenizer
import os.path as osp

if __name__ == '__main__':
    args = get_morpher_args()
    if args.task == 'graph':
        config = load_yaml(osp.join('configurations_morpher', args.dataset_name + '_' + args.pretrain_method + '_' + args.gnn_type + '.yaml'))
    elif args.task == 'node' or args.task == 'edge':
        config = load_yaml(osp.join('configurations_morpher', args.dataset_name + '_' + args.pretrain_method + '_' + args.gnn_type + '_' + args.task + '.yaml'))
    elif args.task == 'transfer':
        config = load_yaml(osp.join('configurations_morpher', 'transfer_from_' + args.source + '_to_' + args.dataset_name + '_' + args.pretrain_method + '_' + args.gnn_type + '.yaml'))

    seed_torch(config.seed)
    torch.cuda.set_device('cuda:'+ str(args.device) if torch.cuda.is_available() else 'cpu')
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # Load pre-trained model (weights)
    model = RobertaModel.from_pretrained('roberta-base').to(args.device)
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    if args.pre_train_model_path == 'None':
        args.pre_train_model_path = osp.join('pre_trained_gnn', args.dataset_name + '.' + args.pretrain_method + '.' + args.gnn_type + '.' + str(config.hid_dim) + 'hidden_dim.pth')

    a = args.pre_train_model_path

    try:
        test_batch_size = config.test_batch_size
    except:
        test_batch_size = config.batch_size
        
    try:
        feature_dim = config.feature_dim
    except:
        feature_dim = 0

    tasker = MMGPL(pre_train_model_path=args.pre_train_model_path, pretrain_method=args.pretrain_method, gnn_type=args.gnn_type, hid_dim=config.hid_dim, num_layer=config.num_layer, dataset_name=args.dataset_name, prompt_type=args.prompt_type, 
                   epochs=config.epochs, shot_num=config.shot_num, batch_size=config.batch_size, prompt_graph_token_num=config.prompt_graph_token_num, tokenizer=tokenizer, llm=model, device=args.device,
                   projector_lr=config.projector_lr, projector_weight_decay=config.projector_weight_decay, projector_tune_lr=config.projector_tune_lr, projector_tune_weight_decay=config.projector_tune_weight_decay,
                   pg_lr=config.pg_lr, pg_weight_decay=config.pg_weight_decay, text_prompt_lr=config.text_prompt_lr, text_prompt_weight_decay=config.text_prompt_weight_decay,
                   projector_dropout_ratio=config.projector_dropout_ratio, temperature=config.temperature, text_prompt_start_vocab=config.text_prompt_start_vocab,
                   projector_epochs=config.projector_epochs, projector_train_eval_diff_threshold=config.projector_train_eval_diff_threshold, projector_train_modular=config.projector_train_modular,
                   projector_tune_epochs=config.projector_tune_epochs, prompt_tune_epochs=config.prompt_tune_epochs,
                   train_val_acc_diff_tol=config.train_val_acc_diff_tol, val_acc_threshold=config.val_acc_threshold, warmup_epochs=config.warmup_epochs, random_seed=config.seed,
                   task=args.task, test_batch_size=test_batch_size, feature_dim = feature_dim)
    
    tasker.run()