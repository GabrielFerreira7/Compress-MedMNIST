
'''
Arquivo destinado a funções utilizadas para compressão de redes neurais 
função de retreino e de compressão 

'''

import torch
from torch import nn, optim
import torch.profiler as profiler

from torchvision import models
#from torchsummary import summary

import pickle
import copy
import random
import os
#from _typeshed import NoneType

import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.quantization

# Carregamento de Dados
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


# Plots e análises
from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
import numpy as np
import time, os
#from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


import torch
import torch.nn.utils.prune as prune
import numpy as np
import random

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
#from pymoo.visualization.scatter import Scatter
from pymoo.termination import get_termination
#from compression import result_final
import pickle


import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem
from medmnist import PathMNIST, OCTMNIST, ChestMNIST
#from compression import objs


def save_data(filename, data):
    # Extrair o diretório do caminho do arquivo
    directory = os.path.dirname(filename)
    
    # Se o diretório não existir, criar
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Abrir o arquivo e salvar os dados
    with open(filename, 'wb') as outfile:
        pickle.dump(data, outfile)



x = [1, 2, 3, 4, 5, 6, 7]
caminho = "NSGA/teste/teste/Cifar10/"

filenameTestes = caminho + "xx.pkl"

save_data(filenameTestes, x)
print("aaaaaaaaaaaaaaaaa")