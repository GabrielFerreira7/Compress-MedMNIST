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
from compression import compress_poda, compress
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

    #n_camadas = 2
    problem = problem_compress(n_camadas, argumento)


    ref_dirs = get_reference_directions("uniform", 2, n_partitions=49)

    algorithm_moead = MOEAD(
        ref_dirs,
        n_neighbors=15,
        prob_neighbor_mating=0.7,
    )
    algorithm = NSGA3(pop_size=50,
                    ref_dirs=ref_dirs)
    # Critério de término
    termination = get_termination("n_gen", 90)

    # Resolvendo o problema
    start_time = time.time()
    res = minimize(problem,
                algorithm,
                termination,
                seed=1,
                save_history=True,
                verbose=True)

    caminho = "NSGA/Resultados/sn_resticao/1epocas/"

    filenameTestes = caminho + argumento + "Res.pkl"
    filenameTestesF = caminho + argumento + "ResF.pkl"
    filenameTestesX = caminho + argumento + "ResX.pkl"
    filenameTestesTempo = caminho + argumento + "tempo.pkl"
    filenameTestesFinal = caminho + argumento + "final.pkl"
    filenameTestesFinalQuant = caminho + argumento + "finalQuant.pkl"

    save_data(filenameTestes, res)
    save_data(filenameTestesF, res.F)
    save_data(filenameTestesX, res.X)



    #final = result_final(res.X)
    final = compress_poda(res.X, argumento, True)
    end_time = time.time()
    execution_time = end_time - start_time
    save_data(filenameTestesFinal, final)
    save_data(filenameTestesTempo, execution_time)

    finalQuant = compress(res.X, argumento, True)
    save_data(filenameTestesFinalQuant, finalQuant)


