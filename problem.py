import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem
from compression import objs, compress



class problem_compress(ElementwiseProblem,  ):

    def __init__(self, n_camadas = 49, configTeste = '1'):
        super().__init__(n_var=n_camadas,  
                         n_obj=2,   
                         n_constr=0,  
                         xl=0.0,   
                         xu=1.0)   
        self.configTeste = configTeste


    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2 = compress(x, self.configTeste)

        if self.configTeste == '5':
            rest = 77.5
        
        elif self.configTeste == '1':
            rest = 74.5
        elif self.configTeste == '2':
            rest = 88.5
        elif self.configTeste == '3':
            rest = 72.8
        elif self.configTeste == '4':
            rest = 48.5
        elif self.configTeste == '6':
            print("teste teste")
            rest = 89.8
        elif self.configTeste == '7':
            rest = 71.5
        elif self.configTeste == '8':
            rest = 48.5


        #g1 = f2 + rest
        out["F"] = [f1, f2]
        #out["G"] = [g1]


#problem = problem_compress()