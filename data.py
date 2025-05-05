

# Carregamento de Dados
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms
import os 



# Plots e análises
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


from medmnist import RetinaMNIST, BloodMNIST, DermaMNIST

root_dir = './data/medmnist/'

args = {
'epoch_num': 40,     # Número de épocas.
'lr': 1e-3,          # Taxa de aprendizado.
'weight_decay': 1e-3,# Penalidade L2 (Regularização).
'batch_size': 32,    # Tamanho do batch.
}

def dadosMedMNIST():
    None



def dados_otimizacaoCifar10():

    data_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    # Carregando o conjunto de dados CIFAR-10
    train_set = datasets.CIFAR10('.',
                                 train=True,
                                 transform=data_transform,
                                 download=True)

    test_set = datasets.CIFAR10('.',
                                train=False,
                                transform=data_transform,
                                download=False)


    train_size = int(0.85 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = random_split(train_set, [train_size, val_size])
    #train_set, val_set, x  = random_split(train_set, [400, 100, len(train_set)-500]) #para testes local
    

    # Criando os DataLoaders
    train_loader = DataLoader(train_set,
                              batch_size=args['batch_size'],
                              shuffle=True)

    val_loader = DataLoader(val_set,
                            batch_size=args['batch_size'],
                            shuffle=True)

    test_loader = DataLoader(test_set,
                             batch_size=args['batch_size'],
                             shuffle=True)


    return train_loader, val_loader, test_loader, args['batch_size']




def retina():
    data_transform = transforms.Compose([
                                     transforms.Resize(28),
                                     #transforms.Grayscale(3),
                                     transforms.ToTensor(),])

    train_set = RetinaMNIST(split="train",transform = data_transform, download=True, root=root_dir)
    test_set = RetinaMNIST(split="test", transform = data_transform, download=True, root=root_dir)
    val_set = RetinaMNIST(split="val", transform = data_transform, download=True, root=root_dir)

    train_loader = DataLoader(train_set,
                          batch_size=args['batch_size'],
                          shuffle=True)

    test_loader = DataLoader(test_set,
                          batch_size=args['batch_size'],
                          shuffle=True)
    val_loader = DataLoader(val_set,
                          batch_size=args['batch_size'],
                          shuffle=True)
    
    return train_loader, val_loader, test_loader, 32 

 
def derma():
    data_transform = transforms.Compose([
                                     transforms.Resize(28),
                                     #transforms.Grayscale(3),
                                     transforms.ToTensor(),])

    train_set = DermaMNIST(split="train",transform = data_transform, download=True, root=root_dir)
    test_set = DermaMNIST(split="test", transform = data_transform, download=True, root=root_dir)
    val_set = DermaMNIST(split="val", transform = data_transform, download=True, root=root_dir)

    train_loader = DataLoader(train_set,
                          batch_size=args['batch_size'],
                          shuffle=True)

    test_loader = DataLoader(test_set,
                          batch_size=args['batch_size'],
                          shuffle=True)
    val_loader = DataLoader(val_set,
                          batch_size=args['batch_size'],
                          shuffle=True)
    
    return train_loader, val_loader, test_loader, 32


def blood():
    # Defina o diretório onde os dados serão salvos
    root_dir = './data/medmnist/'
    
    # Verifique se o diretório existe, caso contrário, crie-o
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    # Defina as transformações para o dataset
    data_transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
    ])

    # Carregar o dataset com o root especificado
    train_set = BloodMNIST(split="train", transform=data_transform, download=True, root=root_dir)
    test_set = BloodMNIST(split="test", transform=data_transform, download=True, root=root_dir)
    val_set = BloodMNIST(split="val", transform=data_transform, download=True, root=root_dir)

    # Criar os DataLoaders
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=True)

    return train_loader, val_loader, test_loader, 32



'''
Resultado das arquiteturas treinadas origiana trteinada nas bases 
Mobinet
    Cifar10: 81.5 restricao = 77.5 
    Blood: 94.60 restricao = 89.8
    Derma: 76.5 restricao = 71.5
    Retina: 51.3 restricao = 50


Resnet50: 
    Cifar10: 78.41 restricao = 74.5
    Blood: 93.13 restricao = 88.5
    Derma: 76.56 restricao = 72.8
    Retina: 51.1 restricao = 48.5
    
'''


'''
    Para execução, necessário fornecer o parametro de teste
    1 cifar10 e resnet50
    2 Blood e resnet50
    3 Derma e resnet50
    4 Retina e resnet50
    5 Cifar10 e Mobilenet_V2
    6 Blood e Mobilenet_V2
    7 Derma e Mobilenet_V2
    8 Retina e Mobilenet_V2
'''