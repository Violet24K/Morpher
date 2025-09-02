# Morpher
[ACL 2025 Main] https://arxiv.org/abs/2412.08174

**[Paper (ACL Anthology)](https://aclanthology.org/2025.acl-long.545/)** | **[Paper (arXiv version)](./Morpher_ACL_2025_main.pdf)** | **[Poster](./ACL_2025_main_Poster.pdf)**

------------------------
![Overview](morpher_vision.png)
Figure: CLIP backbone (top) and this work (bottom). If a research paper cites many papers from biology and computer science, we realize this paper will likely be about computational biology, even if we do not know what exactly computational biology is. This work leverages Multi-modal Prompt Learning for Graph Neural Networks that can effectively teach GNNs language dependency given few training samples with weak text supervision.

------------------------
## How to Use
The "code" folder contains the source code for reproducing our experiments. The coding framework inherits from the [ProG](https://github.com/sheldonresearch/ProG) benchmark repository (we started from a fork of it in Spring 2024).

### Environment Installation
One can manually install the environment by
```sh
# tested on Linux with CUDA 12.4
conda create -n morpher python=3.9.18
pip3 install torch torchvision torchaudio
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install transformers scikit-learn yacs pytorch_lightning
```

### Quick Start
[morpher.py](code/morpher.py) is the script for running our Morpher (multimodal prompt learning for graph neural networks) method.
[improvedaio.py](code/improvedaio.py) is the script for running our ImprovedAIO method. Our Morpher uses our ImprovedAIO for prompting graphs.
```sh
python improvedaio.py --dataset_name MUTAG --gnn_type GCN --pretrain_method GraphCL
python morpher.py --dataset MUTAG --gnn GCN --pretrain_method GraphCL
```

### GNN supporting
Our framework supports various GNN architectures, including GAT, GCN, GCov, GIN, GraphSAGE and GraphTransformer. You can specify the GNN type using the `--gnn_type` argument when running the scripts.

### Other Script Arguments
Please refer to [prompt_graph/utils/get_args.py](code/prompt_graph/utils/get_args.py) for details.


### Pretrain on your own data or other tasks
The [pre_train.py](code/pre_train.py) script allows you to pretrain the model on your own dataset or for other tasks. An example lightweight usage is
```sh
python pre_train.py --task GraphCL --dataset_name MUTAG --gnn_type GCN --hid_dim 128 --num_layer 2 --batch_size 10 --epochs 20 --seed 42 --lr 0.01 --decay 0.0001
```



## Cite
If you find this repository useful in your research, please consider citing the following paper:
```
@inproceedings{DBLP:conf/acl/0006ZJFJBH025,
  author       = {Zihao Li and
                  Lecheng Zheng and
                  Bowen Jin and
                  Dongqi Fu and
                  Baoyu Jing and
                  Yikun Ban and
                  Jingrui He and
                  Jiawei Han},
  editor       = {Wanxiang Che and
                  Joyce Nabende and
                  Ekaterina Shutova and
                  Mohammad Taher Pilehvar},
  title        = {Can Graph Neural Networks Learn Language with Extremely Weak Text
                  Supervision?},
  booktitle    = {Proceedings of the 63rd Annual Meeting of the Association for Computational
                  Linguistics (Volume 1: Long Papers), {ACL} 2025, Vienna, Austria,
                  July 27 - August 1, 2025},
  pages        = {11138--11165},
  publisher    = {Association for Computational Linguistics},
  year         = {2025},
  url          = {https://aclanthology.org/2025.acl-long.545/},
  timestamp    = {Thu, 24 Jul 2025 21:25:39 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/0006ZJFJBH025.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```