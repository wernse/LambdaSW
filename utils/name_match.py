from agents.exp_replay_ace import ExperienceReplayACE
from continuum.dataset_scripts.cifar100 import CIFAR100
from continuum.dataset_scripts.cifar10 import CIFAR10
from continuum.dataset_scripts.coil100 import CoilDataset
from continuum.dataset_scripts.mini_imagenet import Mini_ImageNet
from agents.exp_replay import ExperienceReplay
from utils.buffer.adaptive_retrieve import Adaptive_retrieve
from utils.buffer.gss_greedy_task_boundary import GSSGreedyUpdateTaskBoundary
from utils.buffer.random_retrieve import Random_retrieve
from utils.buffer.reservoir_update import Reservoir_update
from utils.buffer.mir_retrieve import MIR_retrieve
from utils.buffer.weighted_reservoir_update import Weighted_Reservoir_update

data_objects = {
    'cifar100': CIFAR100,
    'cifar10': CIFAR10,
    'mini_imagenet': Mini_ImageNet,
    'coil100': CoilDataset,
}

agents = {
    'ER': ExperienceReplay,
    'ER_ACE': ExperienceReplayACE
}

retrieve_methods = {
    'MIR': MIR_retrieve,
    'random': Random_retrieve,
    'adaptive': Adaptive_retrieve,
}

update_methods = {
    'random': Reservoir_update,
    'weighted': Weighted_Reservoir_update,
    'GSSTask': GSSGreedyUpdateTaskBoundary,
}

