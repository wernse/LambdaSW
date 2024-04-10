#!/usr/bin/bash

MY_PYTHON="python"
nb_seeds=30
seed=0
while [ $seed -le $nb_seeds ]
do

### Hyper param Forgetting
#  Mini-image
#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  COIL-100
#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --data coil100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 100 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data coil100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 100 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --data coil100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 200 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data coil100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 200 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data coil100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 500 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data coil100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 500 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --data coil100 --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 100 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data coil100 --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 100 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data coil100 --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 200 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data coil100 --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 200 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data coil100 --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 500 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data coil100 --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 500 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 0


#  CIFAR-100
#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  CIFAR-10
#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --data cifar10 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 100 --num_tasks 5 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data cifar10 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 100 --num_tasks 5 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --data cifar10 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 200 --num_tasks 5 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data cifar10 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 200 --num_tasks 5 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data cifar10 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 500 --num_tasks 5 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data cifar10 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 500 --num_tasks 5 --seed $seed --lambda_param 0 --random_task_order_seed 0


#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --data cifar10 --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 100 --num_tasks 5 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data cifar10 --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 100 --num_tasks 5 --seed $seed --lambda_param 0 --random_task_order_seed 0

#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data cifar10 --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 200 --num_tasks 5 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data cifar10 --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 200 --num_tasks 5 --seed $seed --lambda_param 0 --random_task_order_seed 0
#
#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data cifar10 --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 500 --num_tasks 5 --seed $seed --lambda_param 1 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data cifar10 --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 500 --num_tasks 5 --seed $seed --lambda_param 0 --random_task_order_seed 0


#Original
#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --data coil100 --agent ER --retrieve random --update random --eps_mem_batch 10 --mem_size 100 --num_tasks 20 --seed $seed
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data coil100 --agent ER --retrieve random --update random --eps_mem_batch 10 --mem_size 200 --num_tasks 20 --seed $seed
#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data coil100 --agent ER --retrieve random --update random --eps_mem_batch 10 --mem_size 500 --num_tasks 20 --seed $seed

#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data cifar10 --agent ER --retrieve random --update random --eps_mem_batch 10 --mem_size 100 --num_tasks 5 --seed $seed
#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --data cifar10 --agent ER --retrieve random --update random --eps_mem_batch 10 --mem_size 200 --num_tasks 5 --seed $seed
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data cifar10 --agent ER --retrieve random --update random --eps_mem_batch 10 --mem_size 500 --num_tasks 5 --seed $seed

#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update random --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update random --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed
#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update random --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed

#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve random --update random --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed
#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve random --update random --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve random --update random --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed


  wait

	((seed++))
done

