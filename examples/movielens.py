# Example training of a matrix factorization algorithm which can be used to build a recommender system.
# This example uses the movielens dataset: http://grouplens.org/datasets/movielens/
# Equations for the matrix factorization algorithm can be found here: http://www.ics.uci.edu/~welling/publications/papers/kdd15_dbmf_v0.07_submitted_arXiv.pdf


import os
import defaults
import urllib
import zipfile
import numpy as np
from ..matrix_factorization.matrix_factorization import MatrixFactorization


def load_dataset():
    """
    Download and unzip the movielens dataset. Make sure you have a valid internet connection!

    Outputs: movielens directory to ../data
    """
    if not os.path.exists( defaults.data_dir ):
        os.makedirs( defaults.data_dir )
    urllib.urlretrieve("http://files.grouplens.org/datasets/movielens/ml-100k.zip",
            defaults.data_dir + "movielens.zip")
    zip_ref = zipfile.ZipFile( defaults.data_dir + 'movielens.zip', 'r')
    zip_ref.extractall( defaults.data_dir )
    zip_ref.close()
    os.remove( defaults.data_dir + 'movielens.zip' )


def movielens():
    """
    Train matrix factorization algorithm on movielens 100k dataset
    """
    # Load data, download if necessary
    try:
        train = np.loadtxt( defaults.data_dir + 'ml-100k/u1.base', dtype = int)
        test = np.loadtxt( defaults.data_dir + 'ml-100k/u1.test', dtype = int)
    except IOError:
        print "Downloading movielens dataset, make sure you have a valid connection..."
        load_dataset()
        train = np.loadtxt( defaults.data_dir + 'ml-100k/u1.base', dtype = int)
        test = np.loadtxt( defaults.data_dir + 'ml-100k/u1.test', dtype = int)
    # Remove last nuisance row
    train = train[:,:3]
    test = test[:,:3]
    # Initialise model
    pmf = MatrixFactorization( train, test, 10 )
    pmf.fit(0.005,10**3)


if __name__ == '__main__':
    movielens()
