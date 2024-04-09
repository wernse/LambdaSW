# λ Stability Wrapper (λSW)


##Requirements

```
Python 3.6.1+
PyTorch 1.8.0
CUDA 11.1.1
```

Create a virtual enviroment
```sh
virtualenv online-cl
```
Activating a virtual environment
```sh
source online-cl/bin/activate
```
Installing packages
```sh
pip install -r requirements.txt
```
conda create --name lsw python=3.6


## Datasets 

### Online Class Incremental
- Split CIFAR-10
- Split CIFAR-100
- Split COIL-100
- Split Mini-ImageNet

### Data preparation
- CIFAR10 & CIFAR100 will be downloaded during the first run
- Mini-ImageNet: Download from https://www.kaggle.com/whitemoon/miniimagenet/download , and place it in datasets/mini_imagenet/
- COIL-100: Download from https://www.kaggle.com/datasets/jessicali9530/coil100, and place it in datasets/coil-100/


## Algorithms 
* ER: Experience Replay [[Paper]](https://arxiv.org/abs/1902.10486)
* MIR: Maximally Interfered Retrieval [[Paper]](https://proceedings.neurips.cc/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html)
* GSS: Gradient-Based Sample Selection [[Paper]](https://arxiv.org/pdf/1903.08671.pdf)

## Run commands
Detailed descriptions of options can be found in [general_main.py](general_main.py)

### Sample commands to run algorithms on Split-CIFAR100
```shell
#ER
python general_main.py --data cifar100 --cl_type nc --agent ER --retrieve random --update random --mem_size 5000

#MIR
python general_main.py --data cifar100 --cl_type nc --agent ER --retrieve MIR --update random --mem_size 5000

#GSS
python general_main.py --data cifar100 --cl_type nc --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000

#-------λSW-------

#ER+λSW
python general_main.py --data cifar100 --cl_type nc --agent ER --retrieve random --update random --mem_size 5000

#MIR+λSW
python general_main.py --data cifar100 --cl_type nc --agent ER --retrieve MIR --update random --mem_size 5000

#GSS+λSW
python general_main.py --data cifar100 --cl_type nc --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000

```

## Repo Structure & Description
    ├──agents                       #Files for different algorithms
        ├──base.py                      #Abstract class for algorithms
        ├──exp_replay.py                #File for ER, MIR and GSS
    
    ├──continuum                    #Files for create the data stream objects
        ├──dataset_scripts              #Files for processing each specific dataset
            ├──dataset_base.py              #Abstract class for dataset
            ├──cifar10.py                   #File for CIFAR-10
            ├──cifar100,py                  #File for CIFAR-100
            ├──coil100.py                    #File for COIL-100
            ├──mini_imagenet.py             #File for Mini_ImageNet
        ├──continuum.py             
        ├──data_utils.py
    
    ├──models                       #Files for backbone models
        ├──pretrained.py                #Files for pre-trained models
        ├──resnet.py                    #Files for ResNet
    
    ├──utils                        #Files for utilities
        ├──buffer                       #Files related to buffer
            ├──buffer.py                    #Abstract class for buffer
            ├──buffer_utils.py              #General utilities for all the buffer files
            ├──gss_greedy_update.py         #File for Buffer: GSS 
            ├──gss_greedy_task_boundary.py  #File for Buffer: GSS+λSW 
            ├──mir_retrieve.py              #File for Retrieval: MIR 
            ├──random_retrieve.py           #File for Retrieval: ER and GSS
            ├──reservoir_update.py          #File for Buffer: reservoir sampling ER and MIR
            ├──weighted_reservoir_update.py #File for Buffer: weighted reservoir sampling ER+λSW and MIR+λSW
        ├──io.py                        #Code related to load and store csv or yarml
        ├──name_match.py                #Match name strings to objects 
        ├──setup_elements.py            #Set up and initialize basic elements
        ├──utils.py                     #File for general utilities
    
    analysis.py                   #File used to produce results and graphs in the paper
    run_ER_batch_size.sh          #Bash script to vary the buffer replay batch size for ER
    run_ER_MIR.sh                 #Bash script to produce ER, ER+λSW, MIR, MIR+λSW results
    run_ER_MIR_hyperparameter.sh  #Bash script for λ hyper-parameter for ER+λSW, MIR+λSW 
    run_random_seeds.sh           #Bash script for random task order seed for ER and ER+λSW
    run_gss.sh                    #Bash script to produce GSS, GSS+λSW results
    run_gss_epsilon.sh            #Bash script for ε threshold evaluation on GSS+λSW Exp


[//]: # (## Acknowledgments)

[//]: # (## Note)
