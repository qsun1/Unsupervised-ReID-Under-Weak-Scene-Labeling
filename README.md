<p align="center">

  <h1 align="center">Unsupervised Person Re-Identification with Wireless Positioning under Weak Scene Labeling </h1>
  <h3 align="center"><a href="https://arxiv.org/abs/2110.15610">Paper Link </a> </h3>
  <div align="center"></div>
</p>

## Implementation of [UMTF(TPAMI'22)](https://arxiv.org/abs/2110.15610) and [another variant](https://github.com/qsun1/Unsupervised-ReID-Under-Weak-Scene-Labeling/blob/main/assets/qisun_bachelor.pdf)
) in the weak scene labeling scenario
+ still updating, please stay tuned...

<p align="center">
TL; DR: Conbined with wireless positioning data, we propose a novel method to boost unsupervised person re-identification performance under weak scene labeling.
</p>
<br>
# Method
Here is the weak scene labeling setting.
![](https://github.com/qsun1/Unsupervised-ReID-Under-Weak-Scene-Labeling/blob/main/assets/setting.png)

Here is the UMTF framework.
![](https://github.com/qsun1/Unsupervised-ReID-Under-Weak-Scene-Labeling/blob/main/assets/umtf.png)

Here is the overview of the variant(in Chinese).

![](https://github.com/qsun1/Unsupervised-ReID-Under-Weak-Scene-Labeling/blob/main/assets/framework_cn.png)

![](https://github.com/qsun1/Unsupervised-ReID-Under-Weak-Scene-Labeling/blob/main/assets/afm_cn.png)

# Setup
## Installation

## Dataset
We will release the link of our dataset soon. Please contact Yiheng Liu or [Qi Sun](sq008@mail.ustc.edu.cn).
# Training

Run the following command to train the network
```
sh tools/run.sh
```
# Evaluations

## WP-ReID
The evaluation results should be consistent with the table below.(in Chinese) 
![](https://github.com/qsun1/Unsupervised-ReID-Under-Weak-Scene-Labeling/blob/main/assets/performance_wpreid_cn.png)

## Campus4K
The evaluation results should be consistent with the table below.(in Chinese) 
![](https://github.com/qsun1/Unsupervised-ReID-Under-Weak-Scene-Labeling/blob/main/assets/performance_4k_cn.png)


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
