import numpy as np
from sgd import StochasticGradientDescent


class MatrixFactorization:
    """
    Methods for performing matrix factorization, training using stochastic gradient descent.

    Once trained the MatrixFactorization object can be used as a recommender system.
    We use notation as in the top reference
    References: 
        1. http://www.ics.uci.edu/~welling/publications/papers/kdd15_dbmf_v0.07_submitted_arXiv.pdf
        2. http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
    """


    def __init__(self,train,test,d=20):
        """
        Initialise the MatrixFactorization object

        Parameters:
        train - the training dataset, array of user, object, rating.
        test - the test dataset, array of user, object, rating.
        d - the dimensionality of the factorization (default 20)
        """
        self.train = train
        self.test = test
        self.n_users = np.max( train[:,0] )
        self.n_objects = np.max( train[:,1] )
        self.d = d
        self.N = self.train.shape[1]

        # Initialize level parameters
        self.U = np.ones( ( self.d, self.n_users ) )
        self.V = np.ones( ( self.d, self.n_objects ) )
        self.a = np.ones( self.n_users )
        self.b = np.ones( self.n_objects )

        # Initialize hyperparameters
        self.lambda_U = np.ones( self.d )
        self.lambda_V = np.ones( self.d )
        self.lambda_a = 1.0
        self.lambda_b = 1.0
        self.tau = 1.0

        # Initialize parameters for hyperprior
        self.alpha = 1.0
        self.beta = 1.0


    def dloglik(self,sgd):
        """
        Calculate gradient of the log likelihood wrt each of the parameters using a minibatch of data

        Parameters:
        sgd - a stochastic gradient descent object, used to specify the minibatch

        Returns:
        dlogU - gradient of the log likelihood wrt U
        dlogV - gradient of the log likelihood wrt V
        dloga - gradient of the log likelihood wrt a
        dlogb - gradient of the log likelihood wrt b
        """
        dlogU = np.zeros( self.U.shape )
        dlogV = np.zeros( self.V.shape )
        dloga = np.zeros( self.a.shape )
        dlogb = np.zeros( self.b.shape )

        # Calculate sum of gradients at each point in the minibatch
        for i in sgd.minibatch:
            ( user, item, rating ) = self.train[i,:]

            # Calculate gradient of the log density at x, update each dlog* accordingly
            dlogU[:,user] += self.τ * ( rating - np.dot( self.U[:,user], self.V[:,item] ) -
                    self.a[user] - self.b[item] )*self.V[:,item]
            dlogV[:,item] += self.τ * ( rating - np.dot( self.U[:,user], self.V[:,item] ) - 
                self.a[user] - self.b[item] )*self.U[:,user]
            dloga[user] += self.τ * ( rating - np.dot( self.U[:,user], self.V[:,item] ) - 
                self.a[user] - self.b[item] )
            dlogb[item] += self.τ * ( rating - np.dot( self.U[:,user], self.V[:,item] ) - 
                self.a[user] - self.b[item] )

        # Adjust log density gradients so they're unbiased
        dlogU *= self.N / sgd.minibatch_size
        dlogV *= self.N / sgd.minibatch_size
        dloga *= self.N / sgd.minibatch_size
        dlogb *= self.N / sgd.minibatch_size

        return dlogU, dlogV, dloga, dlogb


    def fit(self,stepsize):
        """
        Fit the Matrix Factorization algorithm using stochastic gradient descent

        Sensitivity to the stepsize is reduced using ADADELTA (see reference 2)
        """
        sgd = StochasticGradientDescent(self,stepsize)
