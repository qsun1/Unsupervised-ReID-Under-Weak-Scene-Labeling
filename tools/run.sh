CUDA_VISIBLE_DEVICES=2,3 python examples/cluster_contrast_train_usl.py -b 32 -a resnet50 -d wpreid --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16

