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
        self.N = self.train.shape[0]
        self.test_size = self.test.shape[0]

        # Check train and test sets start from 0
        if np.amin( self.train[:,0] ) == 1:
            self.train[:,0] -= 1
        if np.amin( self.train[:,1] ) == 1:
            self.train[:,1] -= 1
        if np.amin( self.test[:,0] ) == 1:
            self.test[:,0] -= 1
        if np.amin( self.test[:,1] ) == 1:
            self.test[:,1] -= 1

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
        sgd - a stochastic gradient descent object, used to specify the minibatch (see sgd.py)

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
            dlogU[:,user] += self.tau * ( rating - np.dot( self.U[:,user], self.V[:,item] ) -
                    self.a[user] - self.b[item] )*self.V[:,item]
            dlogV[:,item] += self.tau * ( rating - np.dot( self.U[:,user], self.V[:,item] ) - 
                self.a[user] - self.b[item] )*self.U[:,user]
            dloga[user] += self.tau * ( rating - np.dot( self.U[:,user], self.V[:,item] ) - 
                self.a[user] - self.b[item] )
            dlogb[item] += self.tau * ( rating - np.dot( self.U[:,user], self.V[:,item] ) - 
                self.a[user] - self.b[item] )

        # Adjust log density gradients so they're unbiased
        dlogU *= self.N / sgd.minibatch_size
        dlogV *= self.N / sgd.minibatch_size
        dloga *= self.N / sgd.minibatch_size
        dlogb *= self.N / sgd.minibatch_size

        return dlogU, dlogV, dloga, dlogb


    def fit(self,stepsize,n_iters=10**4,minibatch_size=4000,window_size=100):
        """
        Fit probabilistic matrix factorization model using train and test set.

        Uses stochastic gradient descent algorithm

        Parameters:
        stepsize - stepsize to use in stochastic gradient descent
        n_iters - number of iterations of stochastic gradient descent
        minibatch_size - minibatch size in stochastic gradient descent (optional)
        window_size - size of window used in ADADELTA (optional)
        """
        sgd = StochasticGradientDescent(self,stepsize,minibatch_size,window_size)
        for i in range(n_iters):
            # Every so often output RMSE on test set and progress, update hyperparameters
            if i % 10 == 0:
                print "{0}\t{1}".format( i, self.rmse() )
                self.update_hyperparams()
            sgd.update(self)


    def update_hyperparams(self):
        """Update hyperparameters using Gibbs sampling"""
        for d in range(self.d):
            # Update U
            shape = self.alpha + self.n_users / 2.0
            rate = self.beta + 1/2.0 * np.square( self.U[d,:] ).sum()
            self.lambda_U[d] = np.random.gamma( shape, 1 / rate )
            # Update V
            shape = self.alpha + self.n_objects / 2.0
            rate = self.beta + 1/2.0 * np.square( self.V[d,:] ).sum()
            self.lambda_V[d] = np.random.gamma( shape, 1 / rate )
        # Update a
        shape = self.alpha + self.n_users / 2.0
        rate = self.beta + 1/2.0 * np.square( self.a ).sum()
        self.lambda_a = np.random.gamma( shape, 1 / rate )
        # Update b
        shape = self.alpha + self.n_objects / 2.0
        rate = self.beta + 1/2.0 * np.square( self.a ).sum()
        self.lambda_b = np.random.gamma( shape, 1 / rate )


    def rmse(self):
        """Calculate the RMSE on the test set, used to check convergence"""
        rmse = 0
        for i in range(self.test_size):
            ( user, item, rating ) = self.test[i,:]
            rmse += ( rating - np.dot( self.U[:,user], self.V[:,item] - 
                      self.a[user] - self.b[item] ) )**2
        rmse = np.sqrt( rmse / self.test_size )
        return rmse
