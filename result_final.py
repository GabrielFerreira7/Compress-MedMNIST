from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
#from pymoo.visualization.scatter import Scatter
from pymoo.termination import get_termination
from compression import result_final
import pickle
import time
import os 
from compression import compress
from problem import problem_compress
import sys

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

def save_data(filename, data):
    # Extrair o diretório do caminho do arquivo
    directory = os.path.dirname(filename)
    
    # Se o diretório não existir, criar
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Abrir o arquivo e salvar os dados
    with open(filename, 'wb') as outfile:
        pickle.dump(data, outfile)



if __name__ == "__main__":
    param = sys.argv
    argumento = param[1]
    print("olha o argumento: ", argumento)
    base = 'Cifar10'
    rede = 'Resnet50'
    if argumento == '1' or argumento == '2' or argumento == '3' or argumento == '4':
        n_camadas = 54
    else:
        n_camadas = 53

    problem = problem_compress(n_camadas, argumento)


    caminho = "Resultados/Batelada/10epocas/"
    caminho2 = "Resultados/Batelada/10epocas/final/"

    filenameTestes = caminho + argumento + "Res.pkl"
    filenameTestesF = caminho + argumento + "ResF.pkl"
    filenameTestesX = caminho + argumento + "ResX.pkl"
    filenameTestesTempo = caminho + argumento + "tempo.pkl"
    filenameTestesFinal = caminho2 + argumento + "final.pkl"

    infile = open(filenameTestes,'rb')
    res = pickle.load(infile)



    #final = result_final(res.X)
    final = compress(res.X, argumento, True)
    end_time = time.time()

    save_data(filenameTestesFinal, final)






    #problem = get_problem("dtlz2")

    # algorithm = MOEAD(
    #     n_neighbors=20,  # número de vizinhos
    #     decomposition="pbi",  # método de decomposição (penalty boundary intersection)
    #     prob_neighbor_mating=0.9,  # probabilidade de cruzamento com vizinhos
    #     sampling=get_sampling("real_random"),
    #     crossover=get_crossover("real_sbx", prob=1.0, eta=20),
    #     mutation=get_mutation("real_pm", eta=20),
    #     eliminate_duplicates=True
    # )