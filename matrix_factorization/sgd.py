import numpy as np


class StochasticGradientDescent:
    """
    Container which holds data necessary for a stochastic gradient descent update 
    using ADADELTA for probabilistic matrix factorization

    References:
        1. http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
        2. http://www.ics.uci.edu/~welling/publications/papers/kdd15_dbmf_v0.07_submitted_arXiv.pdf
    """
    
    def __init__(self,pmf,stepsize,minibatch_size,window_size):
        """
        Initialize the container for SGD

        Parameters:
        pmf - matrix factorization object see matrix_factorization.py
        stepsize - the stepsize to perform SGD at
        minibatch_size - size of the minibatch used at each iteration
        window_size - window size used in ADADELTA
        """
        self.stepsize = stepsize

        # Set the minibatch size
        self.minibatch_size = minibatch_size
        
        # Hold number of iterations so far
        self.iter = 1

        # Controls ADADELTA fixed window
        self.ada = 1 / float( window_size )

        # Containers for ADADELTA G matrices
        self.G_U = np.ones( pmf.U.shape )
        self.G_V = np.ones( pmf.V.shape )
        self.G_a = np.ones( pmf.a.shape )
        self.G_b = np.ones( pmf.b.shape )


    def update(self,pmf):
        """
        Update one step of stochastic gradient descent, using ADADELTA for tuning

        Parameters:
        pmf - matrix factorization object see matrix_factorization.py
        """
        # Sample the next minibatch
        self.minibatch = np.random.choice( range( pmf.N ), self.minibatch_size, replace = False )

        # Calculate gradients at current point
        dlogU, dlogV, dloga, dlogb = pmf.dloglik(self)

        # Update sum of squares used for ADADELTA
        self.G_U += self.ada * ( np.square( dlogU ) - self.G_U )
        self.G_V += self.ada * ( np.square( dlogV ) - self.G_V )
        self.G_a += self.ada * ( np.square( dloga ) - self.G_a )
        self.G_b += self.ada * ( np.square( dlogb ) - self.G_b )

        # Update parameters using SGD
        pmf.U += self.stepsize * np.divide( dlogU, np.sqrt( self.G_U ) )
        pmf.V += self.stepsize * np.divide( dlogV, np.sqrt( self.G_V ) )
        pmf.a += self.stepsize * np.divide( dloga, np.sqrt( self.G_a ) )
        pmf.b += self.stepsize * np.divide( dlogb, np.sqrt( self.G_b ) )
