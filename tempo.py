from prunne import eval_compression


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
from compression import compress, compress_poda
from problem import problem_compress
import sys
import numpy as np

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

    if argumento == '1':
        caminho = "Batelada/2var/1epocas/"
        caminhoInserir = "Batelada/2var/1epocas/Quantizacao/"
        funcao = compress

    elif argumento == '2':
        caminho = "NSGA/Resultados/Batelada/2var/1epocas/"
        caminhoInserir = "NSGA/Resultados/Batelada/2var/1epocas/Poda/"
        funcao = compress_poda

    elif argumento == '3':
        caminho = "Batelada/1epoca/"
        caminhoInserir = "Batelada/1epoca/Quantizacao/"
        funcao = compress

    elif argumento == '4':
        caminho = "Batelada/1epoca/"
        caminhoInserir = "Batelada/1epoca/Poda/"
        funcao = compress_poda

    elif argumento == '5':
        caminho = "NSGA/Resultados/Batelada/10epocas/"
        caminhoInserir = "NSGA/Resultados/Batelada/10epocas/Quantizacao/"
        funcao = compress

    elif argumento == '6':
        caminho = "NSGA/Resultados/Batelada/10epocas/"
        caminhoInserir = "NSGA/Resultados/Batelada/10epocas/Poda/"
        funcao = compress_poda




    for i in range(2, 9):  # De 1 a 8
        # Construir o caminho do arquivo com o número correspondente
        if i != 5:
            print("valor do i:", i)
            arquivo_pickle = f"{caminho}{i}ResX.pkl"
            arquivo_pickle_solucao = f"{caminhoInserir}{i}Final.pkl"
            arquivo_pickleSalvar = f"{caminhoInserir}{i}Tempo.pkl"
            
            with open(arquivo_pickle_solucao, "rb") as file:  # 'rb' significa leitura em modo binário
                final = pickle.load(file)
            final = np.array(final)
            indice_maior = np.argmax(final[:, 1])
            print("solução escolhida:", final[indice_maior])

            with open(arquivo_pickle, "rb") as file:  # 'rb' significa leitura em modo binário
                res = pickle.load(file)
            print(type(res))
            res = np.array(res)
            res = res[indice_maior]
            res.reshape(1, -1) 

            tempo = funcao(res, str(i), True)
            print(tempo)
            save_data(arquivo_pickleSalvar, tempo)






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