
'''
Arquivo destinado a funções utilizadas para compressão de redes neurais 
função de retreino e de compressão 

'''

import torch
from torch import nn, optim
import copy
import torch.quantization
from torch import optim
from sklearn.metrics import accuracy_score
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.utils.prune as prune
import numpy as np
from data import dados_otimizacaoCifar10, retina, blood, derma
from prunne import eval_compression


weight_decay = 1e-3
epoch_num = 1
lr = 1e-3
cont = 0


net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
net3 = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

def train(train_loader, net, criterion, optimizer, device, masks, inverte, batch_size):

  # Training mode
  net.train()
  i = 0

  start = time.time()

  epoch_loss  = []
  pred_list, rotulo_list = [], []
  for batch in train_loader:

    dado, rotulo = batch
    shape = np.shape(rotulo)
    # Cast do dado na GPU
    if batch_size == shape[0]:
      if inverte:
         rotulo = rotulo.view(shape[0]) #apenas para a medMnist
         rotulo = rotulo.long()

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
  

def validate(val_loader, net, epoch, criterion, device, inverte, batch_size):

  # Evaluation mode
  net.eval()

  start = time.time()

  epoch_loss  = []
  pred_list, rotulo_list = [], []
  with torch.no_grad():
    for batch in val_loader:

      dado, rotulo = batch
      shape = np.shape(rotulo)
    # Cast do dado na GPU
      if batch_size == shape[0]:
        if inverte:
         rotulo = rotulo.view(shape[0]) #apenas para a medMnist
         rotulo = rotulo.long()

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

def objs(x, net2, train_loader, val_loader, batch_size, flops, inverte):


  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')
    
  print("tipo de device:", device)
  mask, flopsP = eval_compression(x, net2, device, flops)


  criterion = nn.CrossEntropyLoss().to(device)
  optimizer = optim.Adam(net2.parameters(), lr=lr, weight_decay=weight_decay)

  for epoch in range(epoch_num):
    train_losses = train(train_loader, net2, criterion, optimizer, device, mask, inverte, batch_size)

  val_losses = validate(val_loader, net2, 1, criterion, device, inverte, batch_size)
  print("resultado apos a validacao: ", val_losses)

  return flopsP, -val_losses


def result_final(ResX, net, train_loader, test_loader, batch_size, flops, inverte):

    if torch.cuda.is_available():
      device = torch.device('cuda')
    else:
      device = torch.device('cpu')

    criterion = nn.CrossEntropyLoss().to(device)
    
    resultados = []
    epoch = 10
    for x in ResX:
      net2 = copy.deepcopy(net)
      optimizer = optim.Adam(net2.parameters(), lr=lr, weight_decay=weight_decay)
      mask, flopsP = eval_compression(x, net2, device, flops)
      train_losses = train(train_loader, net2, criterion, optimizer, device, mask, inverte, batch_size)
      val_losses = validate(test_loader, net2, epoch, criterion, device, inverte, batch_size)
      resultados.append([flopsP, val_losses])


    return resultados

def quantizacao(Net, device):
    Net.to(device)
    model_int8 = torch.quantization.quantize_dynamic(
    Net,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights
    return model_int8
    

def result_finalQuantizacao(ResX, net, train_loader, test_loader, batch_size, flops, inverte):

    # if torch.cuda.is_available():
    #   device = torch.device('cuda')
    # else:
    #   device = torch.device('cpu')
    # device = torch.device('cpu')
    # criterion = nn.CrossEntropyLoss().to(device)
    
    resultados = []
    epoch = 10
    for x in ResX:
      net2 = copy.deepcopy(net)
      if torch.cuda.is_available():
        device = torch.device('cuda')
      else:
        device = torch.device('cpu')

      criterion = nn.CrossEntropyLoss().to(device)

      optimizer = optim.Adam(net2.parameters(), lr=lr, weight_decay=weight_decay)
      mask, flopsP = eval_compression(x, net2, device, flops)
      print("comprimiu")
      train_losses = train(train_loader, net2, criterion, optimizer, device, mask, inverte, batch_size)
      print("treinou")

      device = torch.device('cpu')
      criterion = nn.CrossEntropyLoss().to(device)
      
      net5 = quantizacao(net2, device)
      net5.to(device)
      print("quantizou")
      val_losses = validate(test_loader, net5, epoch, criterion, device, inverte, batch_size)
      print("testou")
      resultados.append([flopsP, val_losses])


    return resultados


# def original():
#     if torch.cuda.is_available():
#       device = torch.device('cuda')
#     else:
#       device = torch.device('cpu')

#     criterion = nn.CrossEntropyLoss().to(device)
#     epoch = 10
#     net2 = copy.deepcopy(net)
#     net2.load_state_dict(torch.load('mobilinet_v240.pt', map_location=torch.device('cpu')))
#     val_losses = validate(val_loader, net2, epoch, criterion, device, False)
#     print(val_losses)



def teste():
  vetor = np.random.rand(53)
  objs(vetor)




def compress(x, argumento, final = False):

   
  if argumento == '1':
    net2 = copy.deepcopy(net)
    net2.load_state_dict(torch.load('ResnetCifar10.pt', map_location=torch.device('cpu')))
    train_loader, val_loader, test_loader, batch_size = dados_otimizacaoCifar10()
    flops = 86370304.0
    inverte = False
    
  elif argumento == '2':
    net2 = copy.deepcopy(net)
    net2.load_state_dict(torch.load('Resnet50Blood.pt', map_location=torch.device('cpu')))
    train_loader, val_loader, test_loader, batch_size = blood()
    flops = 82011904.0
    inverte = True
  elif argumento == '3':
    net2 = copy.deepcopy(net)
    net2.load_state_dict(torch.load('Resnet50Derma.pt', map_location=torch.device('cpu')))
    train_loader, val_loader, test_loader, batch_size = derma()
    flops = 82011904.0
    inverte = True

  elif argumento == '4':
    net2 = copy.deepcopy(net)
    net2.load_state_dict(torch.load('Resnet50Retina.pt', map_location=torch.device('cpu')))
    train_loader, val_loader, test_loader, batch_size = retina()
    flops = 82011904.0
    inverte = True

  elif argumento == '5':
    net2 = copy.deepcopy(net3)
    net2.load_state_dict(torch.load('mobilinet_v240.pt', map_location=torch.device('cpu')))
    train_loader, val_loader, test_loader, batch_size = dados_otimizacaoCifar10()
    flops = 7937280.0
    inverte = False

  elif argumento == '6':
    net2 = copy.deepcopy(net3)
    net2.load_state_dict(torch.load('mobileNetBlood.pt', map_location=torch.device('cpu')))
    train_loader, val_loader, test_loader, batch_size = blood()
    flops = 7446000.0
    inverte = True
    
  elif argumento == '7':
    net2 = copy.deepcopy(net3)
    net2.load_state_dict(torch.load('mobileNetDerma.pt', map_location=torch.device('cpu')))
    train_loader, val_loader, test_loader, batch_size = derma()
    flops = 7446000.0
    inverte = True

  elif argumento == '8':
    net2 = copy.deepcopy(net3)
    net2.load_state_dict(torch.load('mobileNetRetina.pt', map_location=torch.device('cpu')))
    train_loader, val_loader, test_loader, batch_size = retina()
    flops = 7446000.0
    inverte = True


  else:
    print("nenhuma combinação para os parametros informados, execute novamente")

  #para quando tem quantização 
  if argumento == '2' or argumento == '3' or argumento == '4':
    flops = 7.92e+07

  elif argumento == '6' or argumento == '7' or argumento == '8':
    flops = 5.73e+06

  elif argumento == '1':
    flops = 8.35e+07

  elif argumento == '5':
    flops = 6.15e+06

  if final:
    print("Resultado final:")
    return result_finalQuantizacao(x, net2, train_loader, test_loader, batch_size, flops, inverte)

  return objs(x, net2, train_loader, val_loader, batch_size, flops, inverte)


def compress_poda(x, argumento, final = False):

  if argumento == '1':
    net2 = copy.deepcopy(net)
    net2.load_state_dict(torch.load('ResnetCifar10.pt', map_location=torch.device('cpu')))
    train_loader, val_loader, test_loader, batch_size = dados_otimizacaoCifar10()
    flops = 86370304.0
    inverte = False
    
  elif argumento == '2':
    net2 = copy.deepcopy(net)
    net2.load_state_dict(torch.load('Resnet50Blood.pt', map_location=torch.device('cpu')))
    train_loader, val_loader, test_loader, batch_size = blood()
    flops = 82011904.0
    inverte = True
  elif argumento == '3':
    net2 = copy.deepcopy(net)
    net2.load_state_dict(torch.load('Resnet50Derma.pt', map_location=torch.device('cpu')))
    train_loader, val_loader, test_loader, batch_size = derma()
    flops = 82011904.0
    inverte = True

  elif argumento == '4':
    net2 = copy.deepcopy(net)
    net2.load_state_dict(torch.load('Resnet50Retina.pt', map_location=torch.device('cpu')))
    train_loader, val_loader, test_loader, batch_size = retina()
    flops = 82011904.0
    inverte = True

  elif argumento == '5':
    net2 = copy.deepcopy(net3)
    net2.load_state_dict(torch.load('mobilinet_v240.pt', map_location=torch.device('cpu')))
    train_loader, val_loader, test_loader, batch_size = dados_otimizacaoCifar10()
    flops = 7937280.0
    inverte = False

  elif argumento == '6':
    net2 = copy.deepcopy(net3)
    net2.load_state_dict(torch.load('mobileNetBlood.pt', map_location=torch.device('cpu')))
    train_loader, val_loader, test_loader, batch_size = blood()
    flops = 7446000.0
    inverte = True
    
  elif argumento == '7':
    net2 = copy.deepcopy(net3)
    net2.load_state_dict(torch.load('mobileNetDerma.pt', map_location=torch.device('cpu')))
    train_loader, val_loader, test_loader, batch_size = derma()
    flops = 7446000.0
    inverte = True

  elif argumento == '8':
    net2 = copy.deepcopy(net3)
    net2.load_state_dict(torch.load('mobileNetRetina.pt', map_location=torch.device('cpu')))
    train_loader, val_loader, test_loader, batch_size = retina()
    flops = 7446000.0
    inverte = True


  else:
    print("nenhuma combinação para os parametros informados, execute novamente")

  #para quando tem quantização 

  if final:
    print("Resultado final:")
    return result_final(x, net2, train_loader, test_loader, batch_size, flops, inverte)

  return objs(x, net2, train_loader, val_loader, batch_size, flops, inverte)




def cont_camadas(Net):
  layer_index = 0    
  for name, module in Net.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        layer_index += 1

    elif isinstance(module, torch.nn.Linear):
        layer_index += 1

  print(layer_index)

#net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
#cont_camadas(net)


    #ResNet50
    #  cifar = 8.35e+07
    # medMnist = 7.92e+07

    #mobilenet
    # cifar = 6.15e+06
    # MedMnist = 5.73e+06