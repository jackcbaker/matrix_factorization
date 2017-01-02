import numpy as np


class StochasticGradientDescent:
    """
    Container which holds data necessary for a stochastic gradient descent update 
    using ADADELTA for probabilistic matrix factorization

    References:
        1. http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
        2. http://www.ics.uci.edu/~welling/publications/papers/kdd15_dbmf_v0.07_submitted_arXiv.pdf
    """
    
    def __init__(self,pmf,stepsize):
        """
        Initialize the container for SGD

        Parameters:
        pmf - matrix factorization object see matrix_factorization.py
        stepsize - the stepsize to perform SGD at
        """
        self.stepsize = stepsize
        
        # Hold number of iterations so far
        self.iter = 1

        # Controls ADADELTA fixed window
        self.ada = 0.01

        # Containers for ADADELTA G matrices
        self.G_U = np.zeros( pmf.U.shape )
        self.G_V = np.zeros( pmf.V.shape )
        self.G_a = np.zeros( pmf.a.shape )
        self.G_b = np.zeros( pmf.b.shape )


    def update(self,pmf):
        """
        Update one step of stochastic gradient descent

        Parameters:
        pmf - matrix factorization object see matrix_factorization.py
        """
        
