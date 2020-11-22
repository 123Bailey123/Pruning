# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import typing
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def vectorize(state_dict: typing.Dict[str, torch.Tensor]):
    """Convert a state dict into a single column Tensor in a repeatable way."""

    return torch.cat([state_dict[k].reshape(-1) for k in sorted(state_dict.keys())])


def unvectorize(vector: torch.Tensor, reference_state_dict: typing.Dict[str, torch.Tensor]):
    """Convert a vector back into a state dict with the same shapes as reference state_dict."""

    if len(vector.shape) > 1: raise ValueError('vector has more than one dimension.')

    state_dict = {}
    for k in sorted(reference_state_dict.keys()):
        if vector.nelement() == 0: raise ValueError('Ran out of values.')

        size, shape = reference_state_dict[k].nelement(), reference_state_dict[k].shape
        this, vector = vector[:size], vector[size:]
        state_dict[k] = this.reshape(shape)

    if vector.nelement() > 0: raise ValueError('Excess values.')
    return state_dict


def perm(N, seed: int = None):
    """Generate a tensor with the numbers 0 through N-1 ordered randomly."""

    gen = torch.Generator()
    if seed is not None: gen.manual_seed(seed)
    perm = torch.normal(torch.zeros(N), torch.ones(N), generator=gen)
    return torch.argsort(perm)


def shuffle_tensor(tensor: torch.Tensor, seed: int = None):
    """Randomly shuffle the elements of a tensor."""

    shape = tensor.shape
    return tensor.reshape(-1)[perm(tensor.nelement(), seed=seed)].reshape(shape)


def shuffle_state_dict(state_dict: typing.Dict[str, torch.Tensor], seed: int = None):
    """Randomly shuffle each of the tensors in a state_dict."""

    output = {}
    for i, k in enumerate(sorted(state_dict.keys())):
        output[k] = shuffle_tensor(state_dict[k], seed=None if seed is None else seed+i)
    return output

def prune(tensor: torch.Tensor, prune_fraction: float, min = True, mask: torch.Tensor = None):
    """Prune the remaining prune_fraction weights from the mak based on the scores in tensor"""
    if mask is None: mask = torch.ones_like(tensor)
    num_to_prune = np.ceil(torch.sum(mask).item() * prune_fraction).astype(int)
    sorted_tensor = torch.sort(tensor[mask == 1].reshape(-1)).values

    
    if min:
       threshold = sorted_tensor[num_to_prune].double()
       return torch.where(tensor.double() > threshold, (mask.double())*1.3, torch.zeros_like(tensor).double())
    else:
       threshold = sorted_tensor[len(sorted_tensor) - num_to_prune]
       return torch.where(tensor.double() < threshold, mask.double(), torch.zeros_like(tensor).double()).int()


def plot_distribution(scores, strategy, prune_fraction, prune_iterations):

    
    # create a PdfPages object
    file_name = strategy.capitalize() +" Pruning"
    result_folder = "Data_Distribution/"

    pdf = PdfPages(result_folder+file_name+".pdf")

    for name, score in scores.items(): 
        info = title = "\n"+strategy.capitalize() +" Pruning | "+name+ "\n"+ "Prune %: "+ str(prune_fraction) + " | Prune Iterations: "+ str(prune_iterations)

        fig, axs = plt.subplots(2)
        fig.suptitle(title)
        layer_score = (torch.flatten(score)).data.numpy()

        info += """
        Layer Name: {}, Layer Size: {}
        Min Score: {}
        Average Score: {}
        Median Score: {}
        Max Score: {}  
        """.format(
            name, len(layer_score), min(layer_score), np.average(layer_score), np.median(layer_score), max(layer_score))

        print (info)
        
        axs[0].hist(layer_score, bins=100)
        axs[0].set_xlabel("Scores")
        axs[0].set_ylabel("Frequency")
        # axs[0].set_title(name)

        sns.distplot(layer_score, ax=axs[1])
        axs[1].set_xlabel("Scores")
        axs[1].set_ylabel("Density")
        # axs[1].set_title(name)

        fig.set_figheight(10)
        fig.set_figwidth(10)
        
        pdf.savefig(fig)
        
    # remember to close the object to ensure writing multiple plots
    pdf.close()
        
        # result_folder = "Data_Distribution/"
        # fig_name = result_folder+strategy+name+".pdf"
        # plt.savefig(fig_name, bbox_inches='tight')


        # f = open(result_folder+"Plot Details.txt", "a")
        # f.write(info)
        # f.close()

    # plt.show()



def plot_distribution2(x):
    fig, axs = plt.subplots(len(scores.keys()))
    info = title = "\n"+strategy +" pruning | Prune %: "+ str(prune_fraction) + "| Prune Iterations: "+ str(prune_iterations)
    fig.suptitle(title)
    

    i = 0    
    for name, score in scores.items(): 
        layer_score = (torch.flatten(score)).data.numpy()

        axs[i].hist(layer_score, bins=100)
        axs[i].set_xlabel("Scores")
        axs[i].set_ylabel("Frequency")
        axs[i].set_title(name)
        fig.set_figheight(5)
        fig.set_figwidth(5)
        i+=1

    result_folder = "Data_Distribution/"
    fig_name = result_folder+strategy+".pdf"
    plt.savefig(fig_name, bbox_inches='tight')


    # print ("Inside Tensorboard plot")
    # from torch.utils.tensorboard import SummaryWriter

    # writer = SummaryWriter()

    # for n_iter in range(len(x)):
    #     writer.add_scalar('Loss/train', x[n_iter], n_iter)

    # writer.flush()
