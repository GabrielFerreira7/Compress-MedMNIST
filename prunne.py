

import torch.nn.utils.prune as prune
import torch.quantization

# Carregamento de Dados
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from torch import optim

# Plots e análises
from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")


import torch
import torch.nn.utils.prune as prune
import numpy as np


def eval_compression(x, Net, device, flops):
    print("nessa aqui")
    masks = {}
    Net = Net.to(device)
    tamanho = 0
    poda = 0
    layer_index = 0

    for name, module in Net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            pesos = np.prod(np.array(module.weight.shape))
            bias = 0#np.prod(np.array(module.bias.shape))
            tamanho = tamanho + pesos + bias

            if (x[layer_index]) > 0:  # Poda para a camada convolucional
                prune.l1_unstructured(module, name='weight', amount=x[layer_index])
                prune.remove(module, 'weight')
                poda = poda + (pesos * x[layer_index])
            mask = (module.weight.data != 0).float()
            masks[name + '.weight'] = mask
            layer_index += 1

        elif isinstance(module, torch.nn.Linear):
            pesos = np.prod(np.array(module.weight.shape))
            bias = 0#np.prod(np.array(module.bias.shape))
            tamanho = tamanho + pesos + bias

            if (x[layer_index]) > 0:  # Poda para a camada linear
                prune.l1_unstructured(module, name='weight', amount=x[layer_index])
                prune.remove(module, 'weight')
                poda = poda + (pesos * x[layer_index])
            #print("entroi na linear")
            layer_index += 1
            mask = (module.weight.data != 0).float()
            masks[name + '.weight'] = mask

    redeComPoda = (tamanho - poda) / tamanho

    #criterion = nn.CrossEntropyLoss().to(device)
    #print("finalizou a compressão")

    return masks, flops*redeComPoda



def eval_compression_nvar(x, Net, device, flops):
    masks = {}
    Net = Net.to(device)
    tamanho = 0
    poda = 0
    layer_index = 0

    for name, module in Net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            pesos = np.prod(np.array(module.weight.shape))
            bias = 0#np.prod(np.array(module.bias.shape))
            tamanho = tamanho + pesos + bias

            if (x[layer_index]) > 0:  # Poda para a camada convolucional
                prune.l1_unstructured(module, name='weight', amount=x[layer_index])
                prune.remove(module, 'weight')
                poda = poda + (pesos * x[layer_index])
            mask = (module.weight.data != 0).float()
            masks[name + '.weight'] = mask
            layer_index += 1

        elif isinstance(module, torch.nn.Linear):
            pesos = np.prod(np.array(module.weight.shape))
            bias = 0#np.prod(np.array(module.bias.shape))
            tamanho = tamanho + pesos + bias

            if (x[layer_index]) > 0:  # Poda para a camada linear
                prune.l1_unstructured(module, name='weight', amount=x[layer_index])
                prune.remove(module, 'weight')
                poda = poda + (pesos * x[layer_index])
            #print("entroi na linear")
            layer_index += 1
            mask = (module.weight.data != 0).float()
            masks[name + '.weight'] = mask

    redeComPoda = (tamanho - poda) / tamanho

    #criterion = nn.CrossEntropyLoss().to(device)
    #print("finalizou a compressão")

    return masks, flops*redeComPoda


def eval_compression_2var(x, Net, device, flops):
    print(x)
    masks = {}
    Net = Net.to(device)
    tamanho = 0
    poda = 0
    layer_index = 0

    for name, module in Net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            pesos = np.prod(np.array(module.weight.shape))
            bias = 0#np.prod(np.array(module.bias.shape))
            tamanho = tamanho + pesos + bias

            if (x[layer_index]) > 0:  # Poda para a camada convolucional
                prune.l1_unstructured(module, name='weight', amount=x[layer_index])
                prune.remove(module, 'weight')
                poda = poda + (pesos * x[layer_index])
            mask = (module.weight.data != 0).float()
            masks[name + '.weight'] = mask
            #layer_index += 1

        elif isinstance(module, torch.nn.Linear):
            pesos = np.prod(np.array(module.weight.shape))
            bias = 0#np.prod(np.array(module.bias.shape))
            tamanho = tamanho + pesos + bias

            if (x[layer_index + 1]) > 0:  # Poda para a camada linear
                prune.l1_unstructured(module, name='weight', amount=x[layer_index + 1])
                prune.remove(module, 'weight')
                poda = poda + (pesos * x[layer_index + 1])
            #print("entroi na linear")
            #layer_index += 1
            mask = (module.weight.data != 0).float()
            masks[name + '.weight'] = mask

    redeComPoda = (tamanho - poda) / tamanho

    #criterion = nn.CrossEntropyLoss().to(device)
    #print("finalizou a compressão")

    return masks, flops*redeComPoda