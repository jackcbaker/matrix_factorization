# Example training of a matrix factorization algorithm which can be used to build a recommender system.
# This example uses the movielens dataset: http://grouplens.org/datasets/movielens/
# Equations for the matrix factorization algorithm can be found here: http://www.ics.uci.edu/~welling/publications/papers/kdd15_dbmf_v0.07_submitted_arXiv.pdf


import os
import urllib
import zipfile
import numpy as np
from ..matrix_factorization.matrix_factorization import MatrixFactorization


def load_dataset():
    """
    Download and unzip the movielens dataset. Make sure you have a valid internet connection!

    Outputs: movielens directory to ../data
    """
    if not os.path.exists('../data'):
        os.makedirs('../data')
    urllib.urlretrieve ("http://files.grouplens.org/datasets/movielens/ml-100k.zip", 
            "../data/movielens.zip")
    zip_ref = zipfile.ZipFile('../data/movielens.zip', 'r')
    zip_ref.extractall('../data/')
    zip_ref.close()
    os.remove( '../data/movielens.zip' )


def movielens():
    """
    Train matrix factorization algorithm on movielens 100k dataset
    """
    # Load data, download if necessary
    try:
        train = np.loadtxt('../data/ml-100k/u1.base', dtype = int)
    except IOError:
        print "Downloading movielens dataset, make sure you have a valid connection..."
        load_dataset()
    test = np.loadtxt('../data/ml-100k/u1.test', dtype = int)
    # Remove last nuisance row
    train = train[:,:3]
    test = test[:,:3]
    # Initialise model
    pmf = MatrixFactorization( train, test, 10 )

if __name__ == '__main__':
    movielens()
