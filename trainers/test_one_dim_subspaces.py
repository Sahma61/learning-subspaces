import torch
import math
import torch.nn as nn

import utils
from args import args

import matplotlib.pyplot as plt

def init(models, writer, data_loader):
    return


def train(models, writer, data_loader, optimizers, criterion, epoch):
    return


def test(models, writer, criterion, data_loader, epoch):

    model = models[0]
    model_0 = models[1]
    model_0.eval()
    model_0.zero_grad()

    model.apply(lambda m: setattr(m, "return_feats", True))
    model_0.apply(lambda m: setattr(m, "return_feats", True))

    model.zero_grad()
    model.eval()
    test_loss = 0
    correct = 0
    ensemble_correct = 0
    m0_correct = 0
    tv_dist = 0.0
    test_loader = data_loader.test_loader
    feat_cosim = 0

    model.apply(lambda m: setattr(m, "alpha", args.alpha1))
    model_0.apply(lambda m: setattr(m, "alpha", args.alpha0))

    if args.update_bn:
        utils.update_bn(data_loader.train_loader, model, device=args.device)
        utils.update_bn(data_loader.train_loader, model_0, device=args.device)

    with torch.no_grad():

        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)

            output, feats = model(data)
            test_loss += criterion(output, target).item()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

            # get model 0
            model_0_output, model_0_feats = model_0(data)
            ensemble_pred = (model_0_output + output).argmax(
                dim=1, keepdim=True
            )
            ensemble_correct += (
                ensemble_pred.eq(target.view_as(pred)).sum().item()
            )
                        
            m0_pred = model_0_output.argmax(dim=1, keepdim=True)
            
            class_names = ['Non-accident', 'Accident']
            
            n = int(math.sqrt(data.shape[0]))
            figure, axis = plt.subplots(n, n)

            data = data.to("cpu")

            for i in range(n):
                for j in range(n):
                    index = i*n + j
                    axis.imshow(torch.squeeze(data[index], 0).permute(1, 2, 0))
                    axis.set_title(f"Predicted:{class_names[int(pred[index])]} Actual:{class_names[int(target.view_as(pred)[index])]}")
            
            #print(f"Predicted, Expected: {class_names[int(m0_pred)]}, {class_names[target]}")

            plt.show()
