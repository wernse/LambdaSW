from abc import abstractmethod
import abc
import numpy as np
import torch
from torch.nn import functional as F
from utils.utils import maybe_cuda, AverageMeter
from torch.utils.data import TensorDataset, DataLoader
import copy
import pickle
import wandb

class ContinualLearner(torch.nn.Module, metaclass=abc.ABCMeta):
    '''
    Abstract module which is inherited by each and every continual learning algorithm.
    '''

    def __init__(self, model, opt, params):
        super(ContinualLearner, self).__init__()
        self.params = params
        self.model = model
        self.opt = opt
        self.data = params.data
        self.cuda = params.cuda
        self.epoch = params.epoch
        self.batch = params.batch
        self.verbose = params.verbose
        self.old_labels = []
        self.acc_array = []
        self.new_labels = []
        self.task_seen = 0
        self.error_list = []
        self.new_class_score = []
        self.old_class_score = []
        self.fc_norm_new = []
        self.fc_norm_old = []
        self.bias_norm_new = []
        self.bias_norm_old = []
        self.lbl_inv_map = {}
        self.class_task_map = {}

    def before_train(self, x_train, y_train):
        new_labels = list(set(y_train.tolist()))
        self.new_labels += new_labels
        for i, lbl in enumerate(new_labels):
            self.lbl_inv_map[lbl] = len(self.old_labels) + i

        for i in new_labels:
            self.class_task_map[i] = self.task_seen

    @abstractmethod
    def train_learner(self, x_train, y_train):
        pass

    def after_train(self):
        self.old_labels += self.new_labels
        self.new_labels_zombie = copy.deepcopy(self.new_labels)
        self.new_labels.clear()
        self.task_seen += 1

    def criterion(self, logits, labels):
        labels = labels.clone()
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        return ce(logits, labels)

    def forward(self, x):
        return self.model.forward(x)

    def evaluate(self, test_loaders):
        self.model.eval()
        acc_array = np.zeros(len(test_loaders))

        # Log confusion matrix
        confusion = np.zeros((int(self.params.num_tasks), int(self.params.num_tasks)))
        task_class_list = [(idx, set(x.dataset.y.tolist())) for idx, x in enumerate(test_loaders)]
        task_map = {}
        for task_id, labels in task_class_list:
            for label in labels:
                task_map[label] = task_id

        with torch.no_grad():
            for task, test_loader in enumerate(test_loaders):
                acc = AverageMeter()
                for i, (batch_x, batch_y) in enumerate(test_loader):

                    batch_x = maybe_cuda(batch_x, self.cuda)
                    batch_y = maybe_cuda(batch_y, self.cuda)
                    total_correct = 0

                    logits = self.model.forward(batch_x)
                    _, pred_label = torch.max(logits, 1)
                    top_2_logits, second_highest = torch.topk(logits, 2)
                    diff = list()
                    for i in top_2_logits:
                        diff_logic = i[0] - i[1]
                        diff.append(round(diff_logic.cpu().item(), 4))

                    total_correct = total_correct + (pred_label == batch_y).sum().item()
                    for pred, true in zip(pred_label.cpu(), batch_y.cpu()):
                        task_true = task_map[true.item()]
                        task_pred = task_map[pred.item()]
                        confusion[task_true][task_pred] = confusion[task_true][task_pred] + 1

                    correct_cnt = total_correct / batch_y.size(0)
                    acc.update(correct_cnt, batch_y.size(0))
                acc_array[task] = acc.avg()
        wandb.log({"confusion_matrix": str(confusion.tolist())})
        return acc_array