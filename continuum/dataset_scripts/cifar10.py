import numpy as np
from torchvision import datasets
from continuum.data_utils import create_task_composition, load_task_with_labels, shuffle_data
from continuum.dataset_scripts.dataset_base import DatasetBase


class CIFAR10(DatasetBase):
    def __init__(self, params):
        dataset = 'cifar10'
        self.random_task_order_seed = params.random_task_order_seed

        num_tasks = params.num_tasks
        super(CIFAR10, self).__init__(dataset, num_tasks, params.num_runs, params)


    def download_load(self):
        dataset_train = datasets.CIFAR10(root=self.root, train=True, download=True)
        self.train_data = dataset_train.data
        self.train_label = np.array(dataset_train.targets)
        labels = []
        datas = []
        # Randomise
        filled_indices = np.arange(len(self.train_label))
        np.random.seed(self.random_task_order_seed)
        np.random.shuffle(filled_indices)
        for idx in filled_indices:
            data, label = self.train_data[idx], self.train_label[idx]
            datas.append(data)
            labels.append(label)
        self.train_label = np.array(labels)
        self.train_data = np.array(datas)
        dataset_test = datasets.CIFAR10(root=self.root, train=False, download=True)
        self.test_data = dataset_test.data
        self.test_label = np.array(dataset_test.targets)

    def setup(self):
        self.task_labels = create_task_composition(class_nums=10, num_tasks=self.task_nums,
                                                   random_task_order_seed=self.params.random_task_order_seed)
        self.test_set = []
        for labels in self.task_labels:
            x_test, y_test = load_task_with_labels(self.test_data, self.test_label, labels)
            self.test_set.append((x_test, y_test))

    def new_task(self, cur_task, **kwargs):
        labels = self.task_labels[cur_task]
        x_train, y_train = load_task_with_labels(self.train_data, self.train_label, labels)
        return x_train, y_train, labels

    def new_run(self, **kwargs):
        self.setup()
        return self.test_set
