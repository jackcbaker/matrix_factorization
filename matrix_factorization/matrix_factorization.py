import numpy as np


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

        # Initialize parameters for hyperprior
        self.alpha = 1.0
        self.beta = 1.0


    def fit(self,stepsize):
        """
        Fit the Matrix Factorization algorithm using stochastic gradient descent

        Sensitivity to the stepsize is reduced using ADADELTA (see reference 2)
        """
