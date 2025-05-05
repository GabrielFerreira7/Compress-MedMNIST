
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

from torch import optim

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

from data import dados_otimizacao


#derma = acuracia do melhor:  75.80645161290323  Produzido na epoca:  13 
#retina = acuracia do melhor:  55.0  Produzido na epoca:  0
flops = 7446000.0
weight_decay = 1e-3
epoch_num = 10
lr = 1e-3
cont = 0

train_loader, val_loader, test_loader = dados_otimizacao()
net = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)


def eval_compression(x, Net, device):
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
    #tempo, test_losses = validate(test_loader, Net, 1, criterion, device)


def train(train_loader, net, epoch, criterion, optimizer, device, masks):

  # Training mode
  net.train()
  i = 0

  start = time.time()

  epoch_loss  = []
  pred_list, rotulo_list = [], []
  for batch in train_loader:

    dado, rotulo = batch
    shape = np.shape(rotulo)
    if shape[0] == args['batch_size']:
      rotulo = rotulo.view(shape[0])

      # Cast do dado na GPU
      dado = dado.to(device)
      rotulo = rotulo.to(device)

      # Forward
      ypred = net(dado)
      loss = criterion(ypred, rotulo)
      epoch_loss.append(loss.cpu().data)

      _, pred = torch.max(ypred, axis=1)
      pred_list.append(pred.cpu().numpy())
      rotulo_list.append(rotulo.cpu().numpy())

      # Backpropagation
      optimizer.zero_grad()
      loss.backward()
      for name, param in net.named_parameters():
        if name in masks:
            param.grad *= masks[name]
            i += 1


      optimizer.step()

  epoch_loss = np.asarray(epoch_loss)
  pred_list  = np.asarray(pred_list).ravel()
  rotulo_list  = np.asarray(rotulo_list).ravel()

  acc = accuracy_score(pred_list, rotulo_list)

  end = time.time()
  #print('#################### Train ####################')
  #print('Epoch %d, Loss: %.4f +/- %.4f, Acc: %.2f, Time: %.2f' % (epoch, epoch_loss.mean(), epoch_loss.std(), acc*100, end-start))
  #print("valor do i;", i)

  return (acc*100)
  

def validate(val_loader, net, epoch, criterion, device):

  # Evaluation mode
  net.eval()

  start = time.time()

  epoch_loss  = []
  pred_list, rotulo_list = [], []
  with torch.no_grad():
    for batch in val_loader:

      dado, rotulo = batch
      shape = np.shape(rotulo)
      if shape[0] == args['batch_size']:
        rotulo = rotulo.view(shape[0])
      # Cast do dado na GPU
        dado = dado.to(device)
        rotulo = rotulo.to(device)

        # Forward
        ypred = net(dado)
        loss = criterion(ypred, rotulo)
        epoch_loss.append(loss.cpu().data)

        _, pred = torch.max(ypred, axis=1)
        pred_list.append(pred.cpu().numpy())
        rotulo_list.append(rotulo.cpu().numpy())

  epoch_loss = np.asarray(epoch_loss)
  pred_list  = np.asarray(pred_list).ravel()
  rotulo_list  = np.asarray(rotulo_list).ravel()

  acc = accuracy_score(pred_list, rotulo_list)

  end = time.time()
  print('********** Validate **********')
  print('Epoch %d, Loss: %.4f +/- %.4f, Acc: %.2f, Time: %.2f\n' % (epoch, epoch_loss.mean(), epoch_loss.std(), acc*100, end-start))

  return acc*100


def calculo_zeros(Net):
      tamanho = 0
      zerados = 0
      layer_index = 0
      zerados = 0
      for name, module in Net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            pesos = np.prod(np.array(module.weight.shape))
            bias = 0#np.prod(np.array(module.bias.shape))
            tamanho = tamanho + pesos + bias
            zerados = zerados + float(torch.sum(module.weight == 0))




            layer_index += 1

        elif isinstance(module, torch.nn.Linear):
            pesos = np.prod(np.array(module.weight.shape))
            bias = 0#np.prod(np.array(module.bias.shape))
            tamanho = tamanho + pesos + bias
            zerados = zerados + float(torch.sum(module.weight == 0))

            layer_index += 1
      return zerados

def objs(x):
  net2 = copy.deepcopy(net)
  net2.load_state_dict(torch.load('mobilinet_v240.pt', map_location=torch.device('cpu')))

  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')
    
  print("tipo de device:", device)
  mask, flops = eval_compression(x, net2, device)


  criterion = nn.CrossEntropyLoss().to(device)
  optimizer = optim.Adam(net2.parameters(), lr=lr, weight_decay=weight_decay)

  for epoch in range(epoch_num):
    train_losses = train(train_loader, net2, epoch, criterion, optimizer, device, mask)

  val_losses = validate(val_loader, net2, 10, criterion, device)
  print("resultado apos a validacao: ", val_losses)

  return flops, -val_losses


def result_final(ResX):


  acc = []

  #print(ResX)
  

  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  criterion = nn.CrossEntropyLoss().to(device)
  
  resultados = []
  epoch = 10
  for x in ResX:
    net2 = copy.deepcopy(net)
    net2.load_state_dict(torch.load('mobilinet_v240.pt', map_location=torch.device('cpu')))
    optimizer = optim.Adam(net2.parameters(), lr=lr, weight_decay=weight_decay)
    mask, flops = eval_compression(x, net2, device)
    train_losses = train(train_loader, net2, epoch, criterion, optimizer, device, mask)
    val_losses = validate(test_loader, net2, epoch, criterion, device)
    #resultado = {'acc': val_losses, 'flops': flops}
    #resultados.append(resultado)
    resultados.append([flops, val_losses])


  return resultados




#result_final()

def original():
    if torch.cuda.is_available():
      device = torch.device('cuda')
    else:
      device = torch.device('cpu')

    criterion = nn.CrossEntropyLoss().to(device)
    epoch = 10
    net2 = copy.deepcopy(net)
    net2.load_state_dict(torch.load('mobilinet_v240.pt', map_location=torch.device('cpu')))
    val_losses = validate(val_loader, net2, epoch, criterion, device)
    print(val_losses)

#original()


def teste():
  vetor = np.random.rand(53)
  objs(vetor)


