import os
import sys

sys.path.append(os.path.abspath("."))

from args import args
from main import main as run

if __name__ == "__main__":

    for seed in range(3):
        args.seed = seed
        args.label_noise = 0.0
        args.workers = 8
        args.batch_size = 128
        args.output_size = 100
        args.trainswa = False
        args.resume = []
        args.label_smoothing = 0.05
        args.num_models = 2
        args.trainer = "ensemble"
        args.optimizer = "sgd"
        args.lr = 0.1
        args.momentum = 0.9
        args.wd = 1e-4
        args.lr_policy = None
        args.log_interval = 10
        args.set = "CIFAR10"
        # this says to use GPU 0.
        args.multigpu = [0]
        args.model = "CIFARResNet"
        args.model_name = "cifar_resnet_20"
        args.conv_type = "StandardConv"
        args.bn_type = "StandardBN"
        args.conv_init = "kaiming_normal"
        args.epochs = 160
        args.warmup_length = 5
        args.data_seed = 0
        args.name = f"id=base+ln={args.label_noise}+seed={args.seed}"
        args.save = True
        args.save_epochs = []
        args.save_iters = []

        # TODO: change these paths -- this is an example.
        args.data = "/home/sahma61/learning-subspaces/data"
        args.log_dir = "learning-subspaces-results/cifar/train-ensemble-members"

        run()
