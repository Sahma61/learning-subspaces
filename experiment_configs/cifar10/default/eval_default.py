import os
import sys

sys.path.append(os.path.abspath("."))


from args import args
from main import main as run

if __name__ == "__main__":

    # TODO: change these paths -- this is an example.
    args.data = "/home/sahma61/learning-subspaces/data"
    args.log_dir = (
        "learning-subspaces-results/cifar/eval-default"
    )

    for seed in range(2):
        args.epochs = 160
        args.seed = seed
        args.lr = 0.1
        args.label_noise = 0.0
        args.beta = 1.0
        args.label_smoothing = 0.05
        args.optimizer = "sgd"
        args.momentum = 0.9
        args.wd = 1e-4
        args.lr_policy = None
        #args.pretrained = True
        args.layerwise = True
        args.num_samples = 1
        args.workers = 8
        args.batch_size = 128
        args.output_size = 100
        args.trainswa = False
        args.test_freq = 10
        args.set = "CIFAR10"
        args.multigpu = [0]
        args.model = "CIFARResNet"
        args.model_name = "cifar_resnet_20"
        args.conv_type = "StandardConv"
        args.bn_type = "StandardBN"
        args.conv_init = "kaiming_normal"

        name_string = (
            f"id=default"
            f"+num_samples={args.num_samples}"
            f"+seed={args.seed}"
        )

        # Now, analyze.
        args.resume = [
            f"learning-subspaces-results/cifar/default/{name_string}+try=0/"
            f"epoch_{args.epochs}_iter_{args.epochs * round(50000 / 128)}.pt"
        ]
        #args.resume = []
        args.num_models = 1
        args.save = False
        args.save_data = True
        args.pretrained = True
        args.epochs = 0
        args.trainer = "default"
        args.update_bn = True
        acc_data = {}
        args.name = (
                f"{name_string}"
                )
        args.save_epochs = []
        run()
