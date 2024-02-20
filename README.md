# Visual Reasoning in Object-Centric Deep Neural Networks

Code for the paper 'Visual Reasoning in Object-Centric Deep Neural Networks: A Comparative Cognition Approach' [arXiv](https://arxiv.org/)

## Datasets
To generate all the datasets used in the paper use the `generate` python scripts. For example, `generate_MTS_original.py` will generate the Original dataset for the match-to-sample task and `generate_MTS_ood.py` will generate the 13 out-of-distribution datasets.

To generate the Original datasets of the SOSD and RMTS tasks, the file `svrt_task1_64x64.zip` needs to be decompressed inside the `data` directory.

## Simulations

We provide training-testing scripts for all models in our main simulations. For example, to train and test 10 randomly inizialized instances of ResNet50 in the match-to-sample task run `MTS_ResNet50_train.py`. 

The scripts staring with 'RTE' train and test the models in `Simulation 5: Rich training regime`. For example, to train 10 instances of Resnet50 in 10 datasets of the match-to-sample task and test them on 4 datasets witheld out-of-distribution datasets of the same task run `RTE_MTS_ResNet50_train.py`.