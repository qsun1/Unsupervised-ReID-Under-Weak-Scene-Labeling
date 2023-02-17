<p align="center">

  <h1 align="center">Unsupervised Person Re-Identification with Wireless Positioning under Weak Scene Labeling </h1>
  <h3 align="center"><a href="https://arxiv.org/abs/2110.15610">Paper Link </a> </h3>
  <div align="center"></div>
</p>

## This repo is still updating...

<p align="center">
TL; DR: Conbined with wireless positioning data, we propose a novel method to boost unsupervised person re-identification performance under weak scene labeling.
</p>
<br>
Here is the overview of our model (in Chinese version).

![](https://github.com/qsun1/Unsupervised-ReID-Under-Weak-Scene-Labeling/blob/main/assets/framework_cn.jpg)

![](https://github.com/qsun1/Unsupervised-ReID-Under-Weak-Scene-Labeling/blob/main/assets/afm_cn.png)
And here is the performance on the WPReID and Campus-4K datasets. (in Chinese version)
# Setup

![](https://github.com/qsun1/Unsupervised-ReID-Under-Weak-Scene-Labeling/blob/main/assets/performance_wpreid_cn.jpg)

![](https://github.com/qsun1/Unsupervised-ReID-Under-Weak-Scene-Labeling/blob/main/assets/performance_4k_cn.jpg)

## Installation

## Dataset
We will release the link of our dataset soon.
# Training

Run the following command to train the network
```
sh tools/run.sh
```
# Evaluations

## WP-ReID

## Campus4K


# Acknowledgements
This project is built upon [SpCL](https://github.com/yxgeee/SpCL) and [Cluster-Contrast](https://github.com/alibaba/cluster-contrast-reid). We thank all the authors for their great work and repos. 


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
}
```
