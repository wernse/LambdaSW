#!/usr/bin/bash

MY_PYTHON="python"
nb_seeds=30
seed=0
while [ $seed -le $nb_seeds ]
do

#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 0
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 0

#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 1
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 1

#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 2
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 2

#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 3
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 3

#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 4
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data cifar100 --agent ER --retrieve random --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 4


#  CUDA_VISIBLE_DEVICES=0 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 1
#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 1000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 1

#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 1
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 1

#  CUDA_VISIBLE_DEVICES=1 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 2
#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 5000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 2

#  CUDA_VISIBLE_DEVICES=2 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed --lambda_param 1 --random_task_order_seed 1
#  CUDA_VISIBLE_DEVICES=3 $MY_PYTHON general_main.py --data mini_imagenet --agent ER --retrieve MIR --update weighted --eps_mem_batch 10 --mem_size 10000 --num_tasks 20 --seed $seed --lambda_param 0 --random_task_order_seed 1


  wait

	((seed++))
done

