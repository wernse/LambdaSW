import numpy as np
from continuum.data_utils import create_task_composition, load_task_with_labels, shuffle_data
from continuum.dataset_scripts.dataset_base import DatasetBase
from PIL import Image
from torchvision.transforms import ToTensor
import os
import dataclasses

TEST_SPLIT = 1/10


@dataclasses.dataclass
class Params:
    num_tasks: int = 20
    fix_order: bool = False
    random_task_order_seed: int = 0


class CoilDataset(DatasetBase):
    def __init__(self, params):
        dataset = "coil-100"
        self.transform = ToTensor()
        self.root_dir = 'datasets/coil-100'
        self.images = [x for x in os.listdir('datasets/coil-100') if '.png' in x]
        self.num_tasks = params.num_tasks
        super(CoilDataset, self).__init__(dataset, self.num_tasks, 0, params)

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

    def download_load(self):
        all_data = {}
        for img in self.images:
            img_id, img_label = img, int(img.split('_')[0].replace('obj', ''))-1
            img_fname = f"{self.root_dir}/{img_id}"
            img = np.array(Image.open(img_fname))

            if all_data.get(img_label) is None:
                all_data[img_label] = []
            all_data[img_label].append(img)
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        for i in all_data:
            x = all_data[i]
            y = np.ones((len(x),)) * i
            perm_inds = np.arange(0, len(x))
            np.random.shuffle(perm_inds)
            rdm_x = [x[ind] for ind in perm_inds]
            rdm_y = y[perm_inds]
            x_test = rdm_x[: int(len(x) * TEST_SPLIT)]
            y_test = rdm_y[: int(len(x) * TEST_SPLIT)]
            x_train = rdm_x[int(len(x) * TEST_SPLIT):]
            y_train = rdm_y[int(len(x) * TEST_SPLIT):]
            train_data.append(x_train)
            train_label.append(y_train)
            test_data.append(x_test)
            test_label.append(y_test)
        self.train_data =  np.concatenate(train_data)
        self.train_label = np.concatenate(train_label)
        self.test_data =  np.concatenate(test_data)
        self.test_label = np.concatenate(test_label)
