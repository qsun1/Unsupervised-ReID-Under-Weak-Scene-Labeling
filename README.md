<p align="center">

  <h1 align="center">Unsupervised Person Re-Identification with Wireless Positioning under Weak Scene Labeling </h1>
  <h3 align="center"><a href="https://arxiv.org/abs/2110.15610">Paper Link </a> </h3>
  <div align="center"></div>
</p>


<p align="center">
Existing unsupervised person re-identification methods only rely on visual clues to match pedestrians under different cameras. Since visual data is essentially susceptible to occlusion, blur, clothing changes, etc., a promising solution is to introduce heterogeneous data to make up for the defect of visual data. Some works based on full-scene labeling introduce wireless positioning to assist cross-domain person re-identification, but their GPS labeling of entire monitoring scenes is laborious. To this end, we propose to explore unsupervised person re-identification with both visual data and wireless positioning trajectories under weak scene labeling, in which we only need to know the locations of the cameras. Specifically, we propose a novel unsupervised multimodal training framework (UMTF), which models the complementarity of visual data and wireless information. Our UMTF contains a multimodal data association strategy (MMDA) and a multimodal graph neural network (MMGN). MMDA explores potential data associations in unlabeled multimodal data, while MMGN propagates multimodal messages in the video graph based on the adjacency matrix learned from histogram statistics of wireless data. Thanks to the robustness of the wireless data to visual noise and the collaboration of various modules, UMTF is capable of learning a model free of the human label on data. Extensive experimental results conducted on two challenging datasets, i.e., WP-ReID and DukeMTMC-VideoReID demonstrate the effectiveness of the proposed method.

</p>
<br>

# Setup

## Installation
The hash encoder will be compiled on the fly when running the code.

## Dataset
```
bash scripts/download_dataset.sh
```
# Training

Run the following command to train monosdf:
```
cd ./code
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf CONFIG  --scan_id SCAN_ID
```
where CONFIG is the config file in `code/confs`, and SCAN_ID is the id of the scene to reconstruct.

We provide example commands for training DTU, ScanNet, and Replica dataset as follows:

# Evaluations

## WP-ReID
please check the script for more details.

## Campus4K
please check the script for more details.


# Acknowledgements
This project is built upon [VolSDF](https://github.com/lioryariv/volsdf). Evaluation scripts for DTU, Replica, and ScanNet are taken from [DTUeval-python](https://github.com/jzhangbs/DTUeval-python)respectively. We thank all the authors for their great work and repos. 


# Citation
If you find our code or paper useful, please cite
```bibtex
@ARTICLE{Liu2022UMTF,
  author={Liu, Yiheng and Zhou, Wengang and Xie, Qiaokang and Li, Houqiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Unsupervised Person Re-Identification with Wireless Positioning under Weak Scene Labeling}, 
  year={2022},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TPAMI.2022.3196364}}
}
```

```bibtex
@inproceedings{Liu2020RCPM,
author = {Liu, Yiheng and Zhou, Wengang and Xi, Mao and Shen, Sanjing and Li, Houqiang},
title = {Vision Meets Wireless Positioning: Effective Person Re-Identification with Recurrent Context Propagation},
year = {2020},
booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
}```
