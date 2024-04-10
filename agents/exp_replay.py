from collections import Counter

import torch
from torch.utils import data

from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.buffer.buffer_utils import random_retrieve
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter
import wandb
import pandas as pd

class ExperienceReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.validation_split = params.validation_split
        self.task = None
        self.data = params.data
        self.tasks = params.num_tasks
        self.count = 0
        self.validation_acc = {}
        self.validation_idxs = []


    def setup_task_map(self):
        # Generate Task Mapping
        task_classes = [[y[1] for y in x][0].unique() for x in self.test_loaders]
        task_mapping = {}
        for idx, task_class in enumerate(task_classes):
            for t_class in task_class:
                task_mapping[int(t_class)] = idx
        self.task_mapping = task_mapping

    def log_tasks(self):
        df = pd.DataFrame(self.buffer.buffer_label.cpu())
        mem_dict = {f"buffer_{idx}" : 0  for idx, i in enumerate(range(self.tasks))}
        mem_values = df[0].value_counts().to_dict()
        for i in mem_values.keys():
            if i < 0:
                continue
            task_key = f"buffer_{self.task_mapping[i]}"
            if not mem_dict.get(task_key):
                mem_dict[task_key] = 0
            mem_dict[task_key] = mem_dict[task_key] + mem_values[i]
        print("log_tasks", mem_dict)
        task_ids = [int(k.split('_')[-1]) for k, v in mem_dict.items()]
        task_ids.sort()
        # print(f"buffer_{task_ids[-1]}", mem_dict.get(f"buffer_{task_ids[-1]}"))
        # for k, v in mem_dict.items():
        #     experiment.report_scalar(title="buffer", series=f"buffer_{k}", value=v, iteration=self.count)
        wandb.log(mem_dict)

    def log_tmp_tasks(self):
        df = pd.DataFrame(self.buffer.tmp_buffer_label.cpu())
        mem_dict = {}
        mem_values = df[0].value_counts().to_dict()
        for i in mem_values.keys():
            if i < 0:
                continue
            task_key = f"buffer_{self.task_mapping[i]}"
            if not mem_dict.get(task_key):
                mem_dict[task_key] = 0
            mem_dict[task_key] = mem_dict[task_key] + mem_values[i]
        print("log_tmp_tasks", mem_dict)
        task_ids = [int(k.split('_')[-1]) for k, v in mem_dict.items()]
        task_ids.sort()
        wandb.log(mem_dict)

    def validate_images(self, indices):
        indices = [int(x) for x in indices]
        y = self.buffer.buffer_label[indices]
        task_distribution = Counter([self.task_mapping.get(int(x)) for x in y])
        print("task_distribution", task_distribution)

        # group indices by label
        idx_grouping = {}
        for label, idx in zip(y, indices):
            task_id = self.task_mapping.get(int(label))
            if idx_grouping.get(task_id) is None:
                idx_grouping[task_id] = []
            idx_grouping[task_id].append(idx)

        validation_acc = {}
        with torch.no_grad():
            for task, indices in idx_grouping.items():
                # task = f"{task}"
                batch_x = self.buffer.buffer_img[indices]
                batch_y = self.buffer.buffer_label[indices]
                total_correct = 0

                logits = self.model.forward(batch_x)
                _, pred_label = torch.max(logits, 1)

                total_correct = total_correct + (pred_label == batch_y).sum().item()
                correct_cnt = total_correct / batch_y.size(0)
                validation_acc[task] = round(correct_cnt, 2)
        print("validation_acc", validation_acc)
        wandb.log({f"v_{k}": v for k, v in validation_acc.items()})
        return validation_acc


    def get_validation_set(self, buffer, validation_size):
        if validation_size is not None:
            current_size = buffer.current_size()
            batch_size = int(current_size * validation_size)
            return random_retrieve(buffer, batch_size, return_indices=True)[2]


    def train_learner(self, x_train, y_train, task):
        if self.task is None:
            self.task = task

        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()
        self.acc_array = []

        # setup tracker
        losses_batch = AverageMeter()
        acc_batch = AverageMeter()
        self.replay = {}

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                logits = self.model.forward(batch_x)
                loss = self.criterion(logits, batch_y)
                _, pred_label = torch.max(logits, 1)
                correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
                # update tracker
                acc_batch.update(correct_cnt, batch_y.size(0))
                losses_batch.update(loss, batch_y.size(0))
                # backward
                self.opt.zero_grad()
                loss.backward()

                # 1. Randomly select validation samples 10% after the buffer retrieve, penalise the last task.
                # How does the validation split affect the buffer.

                # Fixed replay batch size? Distribution?
                mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y,
                                                    excl_indices=self.validation_idxs)
                if mem_x.size(0) > 0:
                    # mem update
                    mem_x = maybe_cuda(mem_x, self.cuda)
                    mem_y = maybe_cuda(mem_y, self.cuda)
                    mem_logits = self.model.forward(mem_x)
                    loss_mem = self.criterion(mem_logits, mem_y)
                    _, pred_label = torch.max(mem_logits, 1)

                    loss_mem.backward()

                self.opt.step()

                new_task = self.task != task
                lambda_param = 1
                if new_task:
                    self.task = task
                    print("")
                    print("")
                    print("New Task")
                    print(self.validation_acc)
                    self.validation_acc = self.validate_images(self.validation_idxs)
                    if len(self.validation_acc) == 0:
                        self.validation_acc = {1: 1}
                    lambda_param = sum(self.validation_acc.values()) / len(self.validation_acc.values())
                    self.validation_idxs = self.get_validation_set(self.buffer, validation_size=self.validation_split)
                    wandb.log({'lambda_param': lambda_param})
                    print("lambda_params", lambda_param)

                    if self.params.update == 'weighted':
                        self.log_tasks()
                self.buffer.update(batch_x, batch_y, new_task=new_task, t=task, lambda_param=lambda_param)
                if self.params.update == 'weighted':
                    if new_task:
                        print("------after update------")
                        self.log_tasks()
                        print("")
                        print("")

                if (i % 50 == 1) and self.verbose:
                    # Log accuracy and task accuracy
                    acc_array = list(self.evaluate(self.test_loaders))
                    task_acc = {f"task_{idx}": v for idx, v in enumerate(acc_array)}
                    wandb.log(task_acc)
                    self.log_tasks()
                    loss = round(losses_batch.avg(), 4)
                    print(f'=> it: {i}, avg. loss: {loss} train acc: {acc_array}')
                    wandb.log({"loss": loss})
                    # experiment.report_scalar(title="loss", series=f"loss", value=loss, iteration=self.count)
                    self.acc_array.append(acc_array)

                    # Train mode after eval
                    self.model = self.model.train()
                self.count = self.count + 1


        self.log_tasks()
        self.after_train()



