import torch

class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DistributedShufle:
    @staticmethod
    def forward_shuffle(x, epoch):
        """ forward shuffle, return shuffled batch of x from all processes.
        epoch is used as manual seed to make sure the shuffle id in all process is same.
        """
        forward_inds, backward_inds = DistributedShufle.get_shuffle_ids(x.shape[0], epoch)
        return x[forward_inds], backward_inds


    @staticmethod
    def backward_shuffle(x, backward_inds):
        """ backward shuffle, return data which have been shuffled back
        x is the shared data, should be local data
        """
        return x[backward_inds]   

    @staticmethod
    def get_shuffle_ids(bsz, epoch):
        """generate shuffle ids for ShuffleBN"""
        torch.manual_seed(epoch)
        # global forward shuffle id  for all process
        forward_inds = torch.randperm(bsz).long().cuda()

        # global backward shuffle id
        backward_inds = torch.zeros(forward_inds.shape[0]).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)

        return forward_inds, backward_inds