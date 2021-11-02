#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os
import sys

sys.path.append(os.path.abspath("."))

import matplotlib.pyplot as plt

import utils as utils
from config import fs_helper
from config import helper


def lvh_helper(ax, setxax=True, setyax=True):

    one_dim_subspace_data = utils.read_csv_files(
        [
            f"/home/sahma61/learning-subspaces/learning-subspaces-results/accident/eval-one-dimesnional-subspaces/results.csv",
        ],
        ["curr_acc1", "ensemble_acc", "m0_acc"],
    )

    ensemble_data = utils.read_csv_files(
        [
            f"/home/sahma61/learning-subspaces/learning-subspaces-results/accident/eval-one-dimesnional-subspaces/results.csv",
        ],
        ["curr_acc1"],
    )
    
    default_data = utils.read_csv_files(
        [
            f"/home/sahma61/learning-subspaces/learning-subspaces-results/accident/eval-default/results.csv",
        ],
        ["curr_acc1"],
    )
    
    xarr = [round(i * 0.05, 2) for i in range(0, 21)]
    dh = utils.DataHolder(xarr, r"\alpha", "one-dim-subspaces")
    
    utils.add_data_helper(
        dh,
        ax,
        one_dim_subspace_data,
        "alpha1",
        "ensemble_acc",
        id="lines",
        **helper("ens_6"),
    )
    
    
    utils.add_data_helper(
        dh,
        ax,
        default_data,
        "alpha1",
        "ensemble_acc",
        id="default",
        **helper("standard_training"),
    )

    
    utils.add_data_helper(
        dh,
        ax,
        one_dim_subspace_data,
        "alpha1",
        "curr_acc1",
        id="lines",
        **helper("t_6"),
    )

    utils.add_data_helper(
        dh,
        ax,
        one_dim_subspace_data,
        "alpha1",
        "ensemble_acc",
        id="lines-layerwise",
        **helper("ens_4"),
    )
    utils.add_data_helper(
        dh,
        ax,
        one_dim_subspace_data,
        "alpha1",
        "curr_acc1",
        id="lines-layerwise",
        **helper("t_4"),
    )

    utils.add_data_helper(
        dh,
        ax,
        one_dim_subspace_data,
        "alpha1",
        "ensemble_acc",
        id="curves",
        **helper("ens_1"),
    )
    utils.add_data_helper(
        dh,
        ax,
        one_dim_subspace_data,
        "alpha1",
        "curr_acc1",
        id="curves",
        **helper("t_1"),
    )
    
    baselines = utils.query(
        default_data,
        x="num_samples",
        y="curr_acc1",
        outlier=lambda x, y, d: False,
    )
    
    
    dh.add(
        *utils.add_baseline(
            ax, xarr, baselines[1], **helper("standard_training")
        )
    )
    
    
    '''
    dh.add(
        *utils.add_baseline(
            ax, xarr, baselines[1], **helper("standard_ensemble")
        )
    )
    '''
    
    if setxax:
        ax.set_xlabel(r"$\alpha$", fontsize=fs_helper("xlabel"))
    if setyax:
        ax.set_ylabel("Accuracy", fontsize=fs_helper("ylabel"))

    ax.set_title(f"Accident Dataset - cResNet20 (CIFAR10)", fontsize=fs_helper("title"))
    ax.set_ylim([0.5, 0.935])


if __name__ == "__main__":

    save = None

    utils.format(plt)

    fig, axlist = plt.subplots(1, 1, figsize=(6, 6))
    ax = axlist

    lvh_helper(ax)

    fig.subplots_adjust(
        top=0.97, left=0.07, right=0.9, bottom=0.3, wspace=0.15, hspace=0.23
    )

    legend = ax.legend(
        loc="upper center",
        # bbox_to_anchor = (0.5, -0.2),
        bbox_to_anchor=(0.425, -0.25),
        ncol=4,
        fontsize=fs_helper("legend"),
    )

    # plt.show()
    plt.savefig("one_dimensional_subspaces.pdf", bbox_inches="tight")
