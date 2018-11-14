#Adapting code from https://github.com/thomasjpfan/pytorch/blob/401ec389db2c9d2978917a6e4d1101b20340d7e7/torch/optim/lr_scheduler.py

import math
import numpy as np
from torch.optim.optimizer import Optimizer


class _LRScheduler(object):
    """
    Class used to link scheduler with optimizer and perform the update of the LR
    """

    def __init__(self, optimizer, last_epoch=-1):
        """
        Arguments:

        optimizer: Optimizer of Model which we want to train

        last_epoch: Set to -1 and used to keep track of previous epoch

        Initialises learning rates and makes one call to step to set it at last_epoch=0
        """
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                        "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        """
        Arguments:

        epoch: How many epochs are done

        Everytime scheduler.step() is called the learning rates are updated by a call to get_lr
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class CyclicLR(_LRScheduler):
    """
    Class that implements different policies of Cyclic LR as per Smith el. al 2015 (https://arxiv.org/abs/1506.01186)
    """

    def __init__(self, optimizer,  max_lrs = [0.01], step_size=3, gamma=0.99, mode='triangular', last_epoch=-1, n_epochs = 10, cut_frac = 0.1, ratio = 32, epoch_length = 60000, batchsize = 1000, verbose = False):
        """
        Constructor to class Cyclic LR:

        Arguments:

        optimizer: Optimizer of Model which we want to train

        max_lrs: Max Learning rate that your policy should go upto. ULMFiT returns a parameter term times this value with that parameter taking max value = 1

        step_size: For Triangular policies, How quickly the learning rate should transition from min to max value - Half of length of cycle

        gamma: For Exp_Range policy, the boundary values reduce by a factor of gamma^(iterations completed)

        mode: Can be triangular, triangular2, exp_range or ulmfit

        last_epoch: Set to -1 and used to keep track of previous epoch

        n_epochs: Number of epochs

        cut_frac: For ULMFiT policy, the fraction of iterations for which we increase the learning rate before making the cut

        ratio: For ULMFiT policy, dictates how much smaller the LR should become at min stage compared to max_lr i.e. if ratio is 32, then min_lr = max_lr/32

        epoch_length = For ULMFiT policy, this is the length of each epoch

        TODO: find a better name for this
        batchsize = For ULMFiT policy, the number of iterations between when consecutive calls to scheduler.step() is made
        """
        self.optimizer = optimizer
        self.max_lrs = max_lrs
        self.step_size = step_size
        self.gamma = gamma
        self.mode = mode
        self.n_epochs = n_epochs
        self.cut_frac = cut_frac
        self.ratio = ratio
        self.epoch_length = epoch_length
        self.cycle_iter,self.cycle_count=0,0
        self.batchsize = batchsize
        self.verbose = verbose
        assert mode in ['triangular', 'triangular2', 'exp_range', 'ulmfit']
        super(CyclicLR, self).__init__(optimizer, last_epoch)
    
    def calc_lr(self, i):
        """
        Implements policy described in ULMFiT paper for increasing the learning rate upto a cutpoint and decreasing slowly afterwards. 

        Doubts:
        1) Check the increment of cycle_iter
        """
        cut_pt = self.epoch_length * self.cut_frac
        if self.cycle_iter > cut_pt:
            p = 1 - (self.cycle_iter - cut_pt)/(self.epoch_length - cut_pt)
            #print i, "jere"
        else: 
            p = self.cycle_iter/cut_pt
        res = i * (1 + p*(self.ratio-1)) / self.ratio
        #1
        if self.epoch_length == self.cycle_iter:
            self.cycle_iter = 0
            self.cycle_count += 1
        return res
    
    def get_lr(self):
        """
        Returns the updated learning rate
        """
        new_lr = []
		# make sure that the length of base_lrs doesn't change. Dont care about the actual value
        i = 0
        self.cycle_iter += self.batchsize
		#print(self.max_lrs)
        for base_lr in self.base_lrs:
            cycle = np.floor(1 + self.last_epoch / (2 * self.step_size))
            x = np.abs(float(self.last_epoch) / self.step_size - 2 * cycle + 1)
            if self.mode == 'triangular':
                lr = base_lr + (self.max_lrs[0] - base_lr) * np.maximum(0, (1 - x))
            elif self.mode == 'triangular2':
                lr = base_lr + (self.max_lrs[0] - base_lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))
            elif self.mode == 'exp_range':
                lr = base_lr + (self.max_lrs[0] - base_lr) * np.maximum(0, (1 - x)) * (self.gamma ** (
                    self.last_epoch))
            elif self.mode == 'ulmfit':
                lr = self.calc_lr(self.max_lrs[i]);
            i+=1
            new_lr.append(lr)
        if self.verbose:
            print(new_lr)
	    return new_lr
