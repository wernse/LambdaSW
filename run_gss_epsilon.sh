#!/usr/bin/bash

MY_PYTHON="python"
nb_seeds=30
seed=0
while [ $seed -le $nb_seeds ]
do

#  5000 Mini-image
#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --policy exp --data mini_imagenet --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --epsilon 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --policy exp --data mini_imagenet --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --epsilon 0.5 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --policy exp --data mini_imagenet --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --epsilon 0.25 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --policy exp --data mini_imagenet --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --epsilon 0.125 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py  --data mini_imagenet --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --epsilon 1 --random_task_order_seed 0

# 500 COIL
#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --policy exp --data coil100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 500 --num_tasks 20 --seed $seed --epsilon 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --policy exp --data coil100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 500 --num_tasks 20 --seed $seed --epsilon 0.5 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --policy exp --data coil100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 500 --num_tasks 20 --seed $seed --epsilon 0.25 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --policy exp --data coil100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 500 --num_tasks 20 --seed $seed --epsilon 0.125 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data coil100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 500 --num_tasks 20 --seed $seed --epsilon 1--random_task_order_seed 0

# 5000 CIFAR100
#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --policy exp --data cifar100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --epsilon 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --policy exp --data cifar100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --epsilon 0.5 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --policy exp --data cifar100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --epsilon 0.25 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --policy exp --data cifar100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --epsilon 0.125 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --epsilon 1 --random_task_order_seed 0

# 500 CIFAR10
#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --policy exp --data cifar10 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 500 --num_tasks 5 --seed $seed --epsilon 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --policy exp --data cifar10 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 500 --num_tasks 5 --seed $seed --epsilon 0.5 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --policy exp --data cifar10 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 500 --num_tasks 5 --seed $seed --epsilon 0.25 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --policy exp --data cifar10 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 500 --num_tasks 5 --seed $seed --epsilon 0.125 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --data cifar10 --agent ER --retrieve random --update GSSTask --eps_mem_batch 10 --mem_size 500 --num_tasks 5 --seed $seed --epsilon 1 --random_task_order_seed 0
  wait

	((seed++))
done

