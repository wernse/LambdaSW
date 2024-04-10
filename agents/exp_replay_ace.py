from collections import Counter

import torch
from torch.utils import data

# from logger import experiment
from torchvision import transforms

from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter
import wandb
import pandas as pd

TRANSFORM = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5071, 0.4867, 0.4408),
                          (0.2675, 0.2565, 0.2761))])

transform = transforms.Compose(
    [transforms.ToPILImage(), TRANSFORM])

class ExperienceReplayACE(ContinualLearner):


    def __init__(self, model, opt, params):
        super(ExperienceReplayACE, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.validation_split = params.validation_split
        self.task = None
        self.data = params.data
        self.tasks = params.num_tasks
        self.count = 0
        self.device = 'cuda'
        self.seen_so_far = torch.tensor([]).long().to(self.device)

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
        print(f"buffer_{task_ids[-1]}", mem_dict.get(f"buffer_{task_ids[-1]}"))
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
        print(f"tmp_buffer_{task_ids[-1]}", mem_dict.get(f"buffer_{task_ids[-1]}"))
        wandb.log(mem_dict)

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

                batch_x_aug = torch.stack([transform(ee.cpu()) for ee in batch_x]).to(self.device)
                present = batch_y.unique()
                self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()


                present = batch_y.unique()
                self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

                logits = self.model.forward(batch_x_aug)
                mask = torch.zeros_like(logits)
                mask[:, present] = 1

                self.opt.zero_grad()
                # if self.seen_so_far.max() < (self.num_classes - 1):
                mask[:, self.seen_so_far.max():] = 1

                if self.task > 0:
                    logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
                loss = self.criterion(logits, batch_y)
                loss_re = torch.tensor(0.)

                if self.task > 0:
                    # sample from buffer
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y,
                                                        task_mapping=self.task_mapping)
                    mem_x = maybe_cuda(mem_x, self.cuda)
                    mem_y = maybe_cuda(mem_y, self.cuda)
                    mem_x_aug = torch.stack([transform(ee.cpu()) for ee in mem_x]).to(self.device)

                    mem_logits = self.model.forward(mem_x_aug)
                    loss_re = self.criterion(mem_logits, mem_y)
                    _, pred_label = torch.max(mem_logits, 1)

                loss += loss_re
                loss.backward()
                self.opt.step()

                # update mem
                new_task = self.task != task
                if new_task:
                    self.task = task

                self.buffer.update(batch_x, batch_y, new_task=new_task, t=task)

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


