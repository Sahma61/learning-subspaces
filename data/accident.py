import os

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch
import torch.multiprocessing
from torchvision import datasets
from torchvision import transforms

from args import args as args

torch.multiprocessing.set_sharing_strategy("file_system")


class CIFAR10:
    def __init__(self):
        super(CIFAR10, self).__init__()
        
        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = (
            {"num_workers": args.workers, "pin_memory": True}
            if use_cuda
            else {}
        )

        data_root = os.path.join(args.data, "Accident")
        traindir = os.path.join(data_root, "train")
        testdir = os.path.join(data_root, "test")

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(720),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        
        test_dataset = datasets.ImageFolder(
            testdir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(720),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        

        if args.label_noise is not None:
            print(f"==> Using label noising proportion {args.label_noise}")

            pfile_train = "imagenet_train"
            pfile_test = "imagenet_test"
            m = len(train_dataset.targets)  # 1281167
            n = len(test_dataset.targets) 

            if not os.path.isfile(pfile_train + ".npy"):
                perm = np.random.permutation(m)
                train_labels = np.random.randint(2, size=(m,))
                np.save(pfile_train, perm)
                np.save(pfile_train + "_labels", train_labels)
            else:
                perm = np.load(pfile_train + ".npy")
                train_labels = np.load(pfile_train + "_labels.npy")

            for k in range(int(m * args.label_noise)):
                train_dataset.samples[perm[k]] = (
                    train_dataset.samples[perm[k]][0],
                    train_labels[k],
                )
   
            train_dataset.targets = [s[1] for s in train_dataset.samples]
            
            if not os.path.isfile(pfile_test + ".npy"):
                perm = np.random.permutation(n)
                test_labels = np.random.randint(2, size=(n,))
                np.save(pfile_test, perm)
                np.save(pfile_test + "_labels", test_labels)
            else:
                perm = np.load(pfile_test + ".npy")
                test_labels = np.load(pfile_test + "_labels.npy")

            for k in range(int(n * args.label_noise)):
                test_dataset.samples[perm[k]] = (
                    test_dataset.samples[perm[k]][0],
                    test_labels[k],
                )
   
            test_dataset.targets = [s[1] for s in test_dataset.samples]
            
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            [test_dataset[i] for i in range(9)], batch_size=args.batch_size, shuffle=True, **kwargs
        )
