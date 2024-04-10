import argparse
import random
import numpy as np
import torch
from experiment.run import multiple_run
from utils.utils import boolean_string
import wandb
import os

def main(args):
    os.environ["WANDB_API_KEY"] = "YOUR KEY"
    wandb.init(project=f"{args.retrieve}_{args.update}_{args.data}_New_Task", config=args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    multiple_run(args, store=args.store, save_path=args.save_path)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Online Continual Learning PyTorch")
    ########################General#########################
    parser.add_argument('--num_runs', dest='num_runs', default=1, type=int,
                        help='Number of runs (default: %(default)s)')
    parser.add_argument('--seed', dest='seed', default=0, type=int,
                        help='Random seed')
    parser.add_argument('--random_task_order_seed', dest='random_task_order_seed', default=0, type=int,
                        help='Random task shuffle seed')
    parser.add_argument('--mem_iters', dest='mem_iters', default=1, type=int,
                        help='mem_iters')
    parser.add_argument('--dynamic_batch', dest='dynamic_batch', default=False,
                        type=boolean_string,
                        help='Enable Dynamic batching')


    ########################Misc#########################
    parser.add_argument('--val_size', dest='val_size', default=0.1, type=float,
                        help='val_size (default: %(default)s)')
    parser.add_argument('--num_val', dest='num_val', default=3, type=int,
                        help='Number of batches used for validation (default: %(default)s)')
    parser.add_argument('--num_runs_val', dest='num_runs_val', default=3, type=int,
                        help='Number of runs for validation (default: %(default)s)')
    parser.add_argument('--error_analysis', dest='error_analysis', default=False, type=boolean_string,
                        help='Perform error analysis (default: %(default)s)')
    parser.add_argument('--verbose', type=boolean_string, default=True,
                        help='print information or not (default: %(default)s)')
    parser.add_argument('--store', type=boolean_string, default=False,
                        help='Store result or not (default: %(default)s)')
    parser.add_argument('--save-path', dest='save_path', default=None)

    ########################Log#########################
    parser.add_argument('--log_logits', type=boolean_string, default=False,
                        help='Log output logits')

    ########################Agent#########################
    parser.add_argument('--agent', dest='agent', default='ER',
                        choices=['ER', 'ER_ACE'],
                        help='Agent selection  (default: %(default)s)')
    parser.add_argument('--update', dest='update', default='random', choices=['random', 'GSS', 'GSSTask', 'weighted'],
                        help='Update method  (default: %(default)s)')
    parser.add_argument('--retrieve', dest='retrieve', default='random', choices=['MIR', 'random', 'adaptive'],
                        help='Retrieve method  (default: %(default)s)')

    ########################Optimizer#########################
    parser.add_argument('--optimizer', dest='optimizer', default='SGD', choices=['SGD', 'Adam'],
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.1,
                        type=float,
                        help='Learning_rate (default: %(default)s)')
    parser.add_argument('--epoch', dest='epoch', default=1,
                        type=int,
                        help='The number of epochs used for one task. (default: %(default)s)')
    parser.add_argument('--batch', dest='batch', default=10,
                        type=int,
                        help='Batch size (default: %(default)s)')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0,
                        help='weight_decay')

    ########################Data#########################
    parser.add_argument('--num_tasks', dest='num_tasks', default=10,
                        type=int,
                        help='Number of tasks (default: %(default)s), OpenLORIS num_tasks is predefined')
    parser.add_argument('--data', dest='data', default="cifar10", choices=['cifar10', 'cifar100', 'coil100', 'mini_imagenet'],
                        help='Path to the dataset. (default: %(default)s)')

    ########################ER#########################
    parser.add_argument('--mem_size', dest='mem_size', default=10000,
                        type=int,
                        help='Memory buffer size (default: %(default)s)')
    parser.add_argument('--eps_mem_batch', dest='eps_mem_batch', default=10,
                        type=int,
                        help='Episode memory per batch (default: %(default)s)')

    ########################MIR#########################
    parser.add_argument('--subsample', dest='subsample', default=50,
                        type=int,
                        help='Number of subsample to perform MIR(default: %(default)s)')

    ########################ER+λSW and MIR+λSW######################### lambda
    parser.add_argument('--lambda_param', type=float, default=1,
                        help='Lambda hyper-parameter to control level of replacement')

    ########################Adaptive######################### lambda
    parser.add_argument('--validation_split', type=float, default=0,
                        help='Lambda hyper-parameter to control level of replacement')

    ########################GSS#########################
    parser.add_argument('--gss_mem_strength', dest='gss_mem_strength', default=10, type=int,
                        help='Number of batches randomly sampled from memory to estimate score')
    parser.add_argument('--gss_batch_size', dest='gss_batch_size', default=10, type=int,
                        help='Random sampling batch size to estimate score')

    ########################GSS+λSW######################### eps
    parser.add_argument('--epsilon', type=float, default=0,
                        help='epsilon threshold for controlling diversity')
    parser.add_argument('--policy', type=str, default="", choices=["", 'sig', 'linear', 'exp'],
                        help='first subsample from recent memories')

    ####################Early Stopping######################
    parser.add_argument('--min_delta', dest='min_delta', default=0., type=float,
                        help='A minimum increase in the score to qualify as an improvement')
    parser.add_argument('--patience', dest='patience', default=0, type=int,
                        help='Number of events to wait if no improvement and then stop the training.')
    parser.add_argument('--cumulative_delta', dest='cumulative_delta', default=False, type=boolean_string,
                        help='If True, `min_delta` defines an increase since the last `patience` reset, '
                             'otherwise, it defines an increase after the last event.')


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    main(args)
