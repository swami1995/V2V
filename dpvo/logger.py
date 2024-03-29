
import torch
from torch.utils.tensorboard import SummaryWriter


SUM_FREQ = 100

class Logger:
    def __init__(self, name, scheduler, args, start_step=0):
        self.total_steps = start_step
        self.running_loss = {}
        self.writer = None
        self.name = name
        self.scheduler = scheduler
        self.test = args.test
        self.save_dir = args.save_dir

    def _print_training_status(self):
        if self.writer is None and not self.test:
            self.writer = SummaryWriter("{}/{}".format(self.save_dir, self.name))
            print([k for k in self.running_loss])

        if self.scheduler is not None:
            lr = self.scheduler.get_lr().pop()
        else:
            lr = 0.0
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in self.running_loss.keys()]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, lr)
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        if self.test:
            print(training_str + metrics_str)

        if not self.test:
            for key in self.running_loss:
                val = self.running_loss[key] / SUM_FREQ
                self.writer.add_scalar(key, val, self.total_steps)
                self.running_loss[key] = 0.0

    def push(self, metrics):

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

        self.total_steps += 1

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter("{}/{}".format(self.save_dir,self.name))
            print([k for k in self.running_loss])
            
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

