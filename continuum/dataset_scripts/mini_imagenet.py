import pickle
import numpy as np
from continuum.data_utils import create_task_composition, load_task_with_labels, shuffle_data
from continuum.dataset_scripts.dataset_base import DatasetBase

TEST_SPLIT = 1 / 6


class Mini_ImageNet(DatasetBase):
    def __init__(self, params):
        dataset = 'mini_imagenet'
        num_tasks = params.num_tasks
        super(Mini_ImageNet, self).__init__(dataset, num_tasks, params.num_runs, params)


    def download_load(self):
        import os
        os.listdir('datasets/mini_imagenet')
        train_in = open("datasets/mini_imagenet/mini-imagenet-cache-train.pkl", "rb")
        train = pickle.load(train_in)
        train_x = train["image_data"].reshape([64, 600, 84, 84, 3])
        val_in = open("datasets/mini_imagenet/mini-imagenet-cache-val.pkl", "rb")
        val = pickle.load(val_in)
        val_x = val['image_data'].reshape([16, 600, 84, 84, 3])
        test_in = open("datasets/mini_imagenet/mini-imagenet-cache-test.pkl", "rb")
        test = pickle.load(test_in)
        test_x = test['image_data'].reshape([20, 600, 84, 84, 3])
        all_data = np.vstack((train_x, val_x, test_x))
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        for i in range(len(all_data)):
            cur_x = all_data[i]
            cur_y = np.ones((600,)) * i
            rdm_x, rdm_y = shuffle_data(cur_x, cur_y)
            x_test = rdm_x[: int(600 * TEST_SPLIT)]
            y_test = rdm_y[: int(600 * TEST_SPLIT)]
            x_train = rdm_x[int(600 * TEST_SPLIT):]
            y_train = rdm_y[int(600 * TEST_SPLIT):]
            train_data.append(x_train)
            train_label.append(y_train)
            test_data.append(x_test)
            test_label.append(y_test)
        self.train_data = np.concatenate(train_data)
        self.train_label = np.concatenate(train_label)
        self.test_data = np.concatenate(test_data)
        self.test_label = np.concatenate(test_label)

    def new_run(self, **kwargs):
        self.setup()
        return self.test_set

    def new_task(self, cur_task, **kwargs):
        labels = self.task_labels[cur_task]
        x_train, y_train = load_task_with_labels(self.train_data, self.train_label, labels)
        return x_train, y_train, labels

    def setup(self):
        self.task_labels = create_task_composition(class_nums=100, num_tasks=self.task_nums,
                                                   random_task_order_seed=self.params.random_task_order_seed)
        self.test_set = []
        for labels in self.task_labels:
            x_test, y_test = load_task_with_labels(self.test_data, self.test_label, labels)
            self.test_set.append((x_test, y_test))

