import numpy as np
import wandb

from utils.setup_elements import input_size_match
from utils import name_match #import update_methods, retrieve_methods
from utils.utils import maybe_cuda
import torch

class Buffer(torch.nn.Module):
    def __init__(self, model, params):
        super().__init__()
        self.params = params
        self.model = model
        self.cuda = self.params.cuda
        self.current_index = 0
        self.tmp_current_index = 0
        self.n_seen_so_far = 0
        self.tmp_n_seen_so_far = 0
        self.device = "cuda" if self.params.cuda else "cpu"

        # define buffer
        buffer_size = params.mem_size

        # Modify for validation set
        buffer_size = int(buffer_size * (1 + params.validation_split))
        wandb.log({'total_buffer': buffer_size})
        self.buffer_size = buffer_size
        print('buffer has %d slots' % buffer_size)
        input_size = input_size_match[params.data]
        buffer_img = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
        buffer_label = maybe_cuda(torch.LongTensor(buffer_size).fill_(-1))
        self.buffer_weights = np.array([])
        self.tmp_buffer_weights = np.array([])

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('buffer_img', buffer_img)
        self.register_buffer('buffer_label', buffer_label)
        self.register_buffer('tmp_buffer_img', buffer_img.detach().clone())
        self.register_buffer('tmp_buffer_label', buffer_label.detach().clone())

        # define update and retrieve method
        self.update_method = name_match.update_methods[params.update](params)
        self.retrieve_method = name_match.retrieve_methods[params.retrieve](params)

    def update(self, x, y,**kwargs):
        return self.update_method.update(buffer=self, x=x, y=y, **kwargs)


    def retrieve(self, **kwargs):
        return self.retrieve_method.retrieve(buffer=self, **kwargs)

    def current_size(self):
        return len(self.buffer_label[self.buffer_label != -1])