#!/usr/bin/bash

MY_PYTHON="python"
nb_seeds=20
seed=0
while [ $seed -le $nb_seeds ]
do
# Mini-ImageNet
#	CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --epsilon 0.125 --policy exp --data mini_imagenet --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --epsilon 0.125 --policy linear --data mini_imagenet --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --epsilon 0.125 --policy sig --data mini_imagenet --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --epsilon 0 --data mini_imagenet --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed --random_task_order_seed 0

#	CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --epsilon 0.125 --policy exp --data mini_imagenet --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --epsilon 0.125 --policy linear --data mini_imagenet --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --epsilon 0.125 --policy sig --data mini_imagenet --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --epsilon 0 --data mini_imagenet --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --random_task_order_seed 0

#	CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --epsilon 0.125 --policy exp --data mini_imagenet --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --epsilon 0.125 --policy linear --data mini_imagenet --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --epsilon 0.125 --policy sig --data mini_imagenet --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --epsilon 0 --data mini_imagenet --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed --random_task_order_seed 0

# CIFAR-10
#	CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --epsilon 0.125 --policy exp --data cifar10 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 100 --num_tasks 5 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --epsilon 0.125 --policy linear --data cifar10 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 100 --num_tasks 5 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --epsilon 0.125 --policy sig --data cifar10 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 100 --num_tasks 5 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --epsilon 0 --data cifar10 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 100 --num_tasks 5 --seed $seed --random_task_order_seed 0

#	CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --epsilon 0.125 --policy exp --data cifar10 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 200 --num_tasks 5 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --epsilon 0.125 --policy linear --data cifar10 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 200 --num_tasks 5 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --epsilon 0.125 --policy sig --data cifar10 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 200 --num_tasks 5 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --epsilon 0 --data cifar10 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 200 --num_tasks 5 --seed $seed --random_task_order_seed 0

#	CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --epsilon 0.125 --policy exp --data cifar10 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 500 --num_tasks 5 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --epsilon 0.125 --policy linear --data cifar10 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 500 --num_tasks 5 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --epsilon 0.125 --policy sig --data cifar10 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 500 --num_tasks 5 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --epsilon 0 --data cifar10 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 500 --num_tasks 5 --seed $seed --random_task_order_seed 0

# COIL-100
#	CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --epsilon 0.125 --policy exp --data coil100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 100 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --epsilon 0.125 --policy linear --data coil100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 100 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --epsilon 0.125 --policy sig --data coil100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 100 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --epsilon 0 --data coil100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 100 --num_tasks 20 --seed $seed --random_task_order_seed 0

#	CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --epsilon 0.125 --policy exp --data coil100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 200 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --epsilon 0.125 --policy linear --data coil100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 200 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --epsilon 0.125 --policy sig --data coil100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 200 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --epsilon 0 --data coil100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 200 --num_tasks 20 --seed $seed --random_task_order_seed 0

#	CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --epsilon 0.125 --policy exp --data coil100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 500 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --epsilon 0.125 --policy linear --data coil100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 500 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --epsilon 0.125 --policy sig --data coil100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 500 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --epsilon 0 --data coil100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 500 --num_tasks 20 --seed $seed --random_task_order_seed 0

# CIFAR-100
#	CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --epsilon 0.125 --policy exp --data cifar100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --epsilon 0.125 --policy linear --data cifar100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --epsilon 0.125 --policy sig --data cifar100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --epsilon 0 --data cifar100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed --random_task_order_seed 0

#	CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --epsilon 0.125 --policy exp --data cifar100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --epsilon 0.125 --policy linear --data cifar100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --epsilon 0.125 --policy sig --data cifar100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --epsilon 0 --data cifar100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --random_task_order_seed 0

#	CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --epsilon 0.125 --policy exp --data cifar100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --epsilon 0.125 --policy linear --data cifar100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --epsilon 0.125 --policy sig --data cifar100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed --random_task_order_seed 0
#	CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --epsilon 0 --data cifar100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed --random_task_order_seed 0
