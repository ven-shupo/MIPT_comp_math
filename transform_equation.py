import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *

class convection_equation():
    """
    class 'solver of difference task for convection equation'
    methods:
         class.fit()
         class.get_U()
         class.get_X()
         class.get_T()
    """
    
    def __init__(self):
        self.all_methods = { #dir with all methods and function for solution them
            "against_stream": self.against_stream,
            "LV": self.LV
        }
        
    def get_U(self):
        """
        returns matrix with value of U at all knots
        """
        return self.U
    
    def get_X(self):
        """
        returns all knots of coordinates
        """
        return self.X
    
    def get_T(self):
        """
        returns all knots of time
        """
        return self.T    
    
    def packer1(self, n):
        """
        Private function | packer for against stream method
        """        
        if self.frequency == True: # if border condition have frequency
            self.U[-n][1:] = list(map(lambda m: self.f(self.X[m],self.T[-n+1])*self.dt +\
                                      self.U[-n+1][m] - (self.sigma)*(self.U[-n+1][m] -\
                                                                                  self.U[-n+1][m-1]) , range(1, len(self.X))))
            self.U[-n][0] = self.U[-n][-1]
            
        if self.frequency == False: # if border condition have not frequency
            self.U[-n][1:] = list(map(lambda m: self.f(self.X[m],self.T[-n+1])*self.dt +\
                                      self.U[-n+1][m] - (self.sigma)*(self.U[-n+1][m] -\
                                                                                  self.U[-n+1][m-1]) , range(1, len(self.X))))
    def packer2(self, n):
        """
        Private function | packer for method Laksa-Vendrofa
        """
        if self.frequency == True: # if border condition have frequency
            self.U[-n][1:-1] = list(map(lambda m: self.f(self.X[m],self.T[-n+1])*self.dt + self.U[-n+1][m]\
                                      - self.a*self.dt*(self.U[-n+1][m+1] - self.U[-n+1][m-1])/(2*self.dx)\
                                      + ((self.a*self.dt)**2)*(self.U[-n+1][m+1] - 2*self.U[-n+1][m]\
                                                               + self.U[-n+1][m-1])/(2*(self.dx**2)), range(1, len(self.X)-1)))
            self.U[-n][0] = self.U[-n][-2]
            
        if self.frequency == False: # if border condition have not frequency
            self.U[-n][1:-1] = list(map(lambda m: self.f(self.X[m],self.T[-n+1])*self.dt + self.U[-n+1][m]\
                                      - self.a*self.dt*(self.U[-n+1][m+1] - self.U[-n+1][m-1])/(2*self.dx)\
                                      + ((self.a*self.dt)**2)*(self.U[-n+1][m+1] - 2*self.U[-n+1][m]\
                                                               + self.U[-n+1][m-1])/(2*self.dx**2), range(1, len(self.X)-1)))
        
    def LV(self, a, U1, U2, x_start, x_end, knot_in_X, t_start, t_end, knot_in_T):
        """
        Private function | method Laksa-Vendrofa
        """
        #preproc
        self.preproc(a, x_start, x_end, knot_in_X, t_start, t_end, knot_in_T, U1, U2)
        
        #check stable
        assert abs(self.sigma - 1) < 0.05, '!SOLUTION IS NOT STABLE!\nsigma must be about 1+-(0.05)\nsigma =' f"{self.sigma}" 
        
        #fill U | using packer1
        list((map(self.packer2, range(2, len(self.T) + 1))))
        
    def preproc(self, a, x_start, x_end, knot_in_X, t_start, t_end, knot_in_T, U1, U2):
        """
        Private function | preprocessing data
        """
        #create X, T, U and steps  
        X = np.linspace(x_start, x_end, knot_in_X)
        T = np.linspace(t_start, t_end, knot_in_T)  
        dt = T[1] - T[0]
        dx = X[1] - X[0]
        U = np.zeros([len(T),len(X)])
        sigma = a*dt/dx
        
        #fill U | set border and initial conditions
        U[-1] = np.array(list(map(lambda i: U1(i),X))) #fill bottom
        U[:,0] = np.array(list(map(lambda i: U2(i), reversed(T)))) #fill left board
        
        #remember this elem
        self.U, self.X, self.T, self.dx, self.dt, self.a, self.sigma = U, X, T, dx, dt, a, sigma
    
    def against_stream(self, a, U1, U2, x_start, x_end, knot_in_X, t_start, t_end, knot_in_T):
        """
        Private function | method against_stream
        """
        #preproc
        self.preproc(a, x_start, x_end, knot_in_X, t_start, t_end, knot_in_T, U1, U2)
        
        #check stable
#         assert self.sigma <= 1, '!SOLUTION IS NOT STABLE!\nsigma must not be more 1\nsigma =' f"{self.sigma}" 
        
        #fill U | using packer1
        list((map(self.packer1, range(2, len(self.T) + 1))))
         
    
    def fit(self, a, t_end, x_end, knot_in_T, knot_in_X, x_start = 0, t_start = 0,\
                        U1 = lambda x: 0, U2 = lambda x: 0, f = lambda x, t: 0, method = 'against_stream', frequency = False):
        """
        solver of difference task for convection equation
        in:  a - coef
             U1 - function setting the border condition
                int: the point
                out: value of func in this point
             U2 - function setting the initial condition
                int: the point
                out: value of func in this point
             t_end - the end point of time mesuarements
             x_end - the end point of coordinates measurements
             t_start - the start point of time measuarements
             x_start - the start point of coordinates  measurements
             knot_in_T - value of knot in T
             knot_in_X - value of knot in X
             method - is solution method, could be next ['against_stream', 'Laks-Vendrof']
        after fit filled in self.U, self.X, self.T, you can get them with method: class.get_@()
        """    
        # remember method and frequency
        self.method, self.frequency, self.f = method, frequency, f

        # ran the solver for this method
        self.all_methods[method](a, U1, U2, x_start, x_end, knot_in_X, t_start, t_end, knot_in_T) 
                
        #output
        print("Computational is succeeded\n...",'\nsigma =', f"{format(self.sigma, '.6f')}", '\ndt = ',format(self.dt, '.3f'), '\ndx = ', format(self.dx, '.3f'))   
        print('method: 'f"{method}")
        
if __name__ == "__main__":
    print("This is solver for transform equation")