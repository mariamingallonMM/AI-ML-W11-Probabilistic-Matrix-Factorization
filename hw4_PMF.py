"""
This code implements a probabilistic matrix factorization (PMF) per weeks 10 and 11 assignment of the machine learning module part of Columbia University Micromaster programme in AI. 
Written using Python 3.7 for running on Vocareum. Version run on Vocareum for grading (all docs removed to avoid issues with grading platform)

Execute as follows:
$ python3 hw4_PMF.py ratings.csv
"""

from __future__ import division

# builtin modules
import sys
import os
import math
from random import randrange
import functools
import operator
import requests
import psutil

# 3rd party modules
import numpy as np
import pandas as pd
import scipy as sp
from scipy.cluster.vq import kmeans2
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy import stats

def PMF(train_data, headers = ['user_id', 'movie_id'], lam:int = 2, sigma2:float = 0.1, d:int = 5):
    """
    Implements Probabilistic Matrix Factorization.

    ------------
    Parameters:

    - data: dataset used for training (e.g. the ratings.csv dataset with missing values for users and movies).
    - headers: title of the headers in the dataset for the 'users id' and 'movie id' values.
    - lam: lambda value to initialise the Gaussian zero mean distribution (default lam = 2 for this assignment).
    - sigma2: covariance of the Gaussian (default sigma2 = 0.1 for this assignment).
    - d: number of dimensions for the ranking, (default d = 5 for this assignment).
    
    ------------
    Returns:

    - L: Loss
    - U_matrices: 
    - V_matrices: 

    """

    L = []
    U_matrices = []
    V_matrices = []

    # first convert dataframe to the ratings matrix as a sparse matrix
    M, n, m = df_to_ratings_matrix(train_data, headers = headers)

    initial_parameters = initialize_parameters(lam, n, m, d)

    new_parameters = update_parameters(M, initial_parameters, lam, n, m, d)


    return L, U_matrices, V_matrices


def initialize_parameters(lam, n, m, d):
    """
    Initializes our parameters. First the V matrix as a random Gaussian zero mean distribution from a given lambda.
    
    ------------
    Parameters:

    - lam: dataframe used for training (e.g. the ratings.csv dataset with missing values for users and movies).
    - n: number of users in dataset
    - m: number of movies in dataset
    - d: number of dimensions for the ranking, (default d = 5 for this assignment).

    ------------
    Returns:
    It updates the 'parameters' dictionary with values for:

    - U
    - V
    - lambda_U
    - lambda_V

    """

    U = np.zeros((d, n), dtype=np.float64)
    V = np.random.normal(0.0, 1.0 / lam, (d, m))
    
    parameters['U'] = U
    parameters['V'] = V
    parameters['lambda_U'] = lam
    parameters['lambda_V'] = lam
    
    return parameters


def get_data(filename, **kwargs):
    """
    Read data from a file given its name. Option to provide the path to the file if different from: [./datasets/in].
    ------------
    Parameters:
    - filename: name of the file to be read, to get the data from.
    - kwargs (optional): 
        - 'headers': list of str for the headers to include in the outputs file created
        - 'path': str of the path to where the file is read, specified if different from default ([./datasets/in])
    ------------
    Returns:
    - df: a dataframe of the data

    """
    
    # Define input filepath
    if 'path' in kwargs:
        filepath = kwargs['path']
    else:
        filepath = os.path.join(os.getcwd(),'datasets','out')

    input_path = os.path.join(filepath, filename)

    # If provided, use the title of the columns in the input dataset as headers of the dataframe
    if 'headers' in kwargs:
        # Read input data
        df = pd.read_csv(input_path, names = kwargs['headers'])
    else:
        # Read input data
        df = pd.read_csv(input_path)
       
    return df


def df_to_ratings_matrix(df, **kwargs):
    """
    Converts a given dataframe to a sparse matrix, in this case the M ratings matrix.

    ------------
    Parameters:

    - df: dataframe used for training (e.g. the ratings.csv dataset with missing values for users and movies).
    - headers (optional): title of the headers in the dataset for the 'users id' and 'movie id' values.
   
    ------------
    Returns:

    - M: the ratings matrix, as sparse (zeros used to fill the nan, missing values)
    """

    df = df.dropna(how='all')

    if 'headers' in kwargs:
        headers = kwargs['headers']
        users_header = headers[0]
        movies_header = headers[1]
    else:
        users_header = 'user_id'
        movies_header = 'movie_id'

    users = df[users_header].unique()
    movies = df[movies_header].unique()
    df_values = df.values
   
    # initialise M ratings matrix as a sparse matrix of zeros
    M = np.zeros((len(users), len(movies)))

    rows = {}
    cols = {}

    for i, user_id in enumerate(users):
        rows[user_id] = i

    for j, movie_id in enumerate(movies):
        cols[movie_id] = j
    
    for index, row in df.iterrows():
        i = rows[row.user_id]
        j = cols[row.movie_id]
        M[i, j] = row.rating
    
    n = len(users) #number of rows
    m = len(movies) #number of columns
    
    return M, n, m



def update_parameters(M, parameters, lam, n, m, d):
    """
    Implements the function that updates U and V.

    ------------
    Parameters:
   
    ------------
    Returns:

    It updates the 'parameters' dictionary with values for:

    - U
    - V
    - lambda_U
    - lambda_V

    """

    U = parameters['U']
    V = parameters['V']
    lambda_U = parameters['lambda_U']
    lambda_V = parameters['lambda_V']
    
    for i in range(n):
        V_j = V[:, M[i, :] > 0]
        U[:, i] = np.dot(np.linalg.inv(np.dot(V_j, V_j.T) + lambda_U * np.identity(d)), np.dot(M[i, M[i, :] > 0], V_j.T))
        
    for j in range(m):
        U_i = U[:, M[:, j] > 0]
        V[:, j] = np.dot(np.linalg.inv(np.dot(U_i, U_i.T) + lambda_V * np.identity(d)), np.dot(M[M[:, j] > 0, j], U_i.T))
        
    parameters['U'] = U
    parameters['V'] = V
    
    return parameters



def log_a_posteriori(parameters, M):
    """
    #TODO: add docs
    Implements the Log-a posteriori with equation as follows:
    
    L=-\frac 1 2 \left(\sum_{i=1}^N\sum_{j=1}^M(R_{ij}-U_i^TV_j)_{(i,j) \in \Omega_{R_{ij}}}^2+\lambda_U\sum_{i=1}^N\|U_i\|_{Fro}^2+\lambda_V\sum_{j=1}^M\|V_j\|_{Fro}^2\right)

    ------------
    Parameters:
   
    ------------
    Returns:


    """


    lambda_U = parameters['lambda_U']
    lambda_V = parameters['lambda_V']
    U = parameters['U']
    V = parameters['V']
    
    UV = np.dot(U.T, V)
    M_UV = (M[M > 0] - UV[M > 0])
    
    return -0.5 * (np.sum(np.dot(M_UV, M_UV.T)) + lambda_U * np.sum(np.dot(U, U.T)) + lambda_V * np.sum(np.dot(V, V.T)))

def predict(user_id, movie_id):
    """
    Predicts the rating value. Note the value has been scaled within the range 0-5.

    ------------
    Parameters:

    - user_id:
    - movie_id:
   
    ------------
    Returns:


    """


    U = parameters['U']
    V = parameters['V']
    
    r_ij = U[:, user_to_row[user_id]].T.reshape(1, -1) @ V[:, movie_to_column[movie_id]].reshape(-1, 1)

    max_rating = parameters['max_rating']
    min_rating = parameters['min_rating']

    return 0 if max_rating == min_rating else ((r_ij[0][0] - min_rating) / (max_rating - min_rating)) * 5.0




def save_outputs_txt(data, iter_list:list = [5, 10, 25]):
    """
    Write the outputs to csv files.

    ------------
    Parameters:

    - data: a list of the resulting matrixes to write as outputs.
    - iter_list: the iterations to store as output csv files for the U and V matrixes.
    
    ------------
    Returns:

    - csv files with the output data

    """

    L_results = data[0]
    np.savetxt("objective.csv", L_results, delimiter=",")

    U_results = data[1]
    V_results = data[2]

    for i in iter_list:
        filename = "U-" + str(i) + ".csv"
        np.savetxt(filename, U_results[i-1], delimiter=",")
        filename = "V-" + str(i) + ".csv"
        np.savetxt(filename, V_results[i-1], delimiter=",")

    return




def main():

    #Uncomment next line when running in Vocareum
    #train_data=np.genfromtxt(sys.argv[1], delimiter = ',', skip_header=1)
    train_data = get_data('ratings_sample.csv', path = os.path.join(os.getcwd(), 'datasets'), headers=['user_id', 'movie_id', 'rating'])

    # Assuming the PMF function returns Loss L, U_matrices and V_matrices
    L, U_matrices, V_matrices = PMF(train_data, headers = ['user_id', 'movie_id'], lam = 2, sigma2 = 0.1, d = 5)

    save_outputs_txt([L, U_matrices, V_matrices], [5, 10, 25])


if __name__ == '__main__':
    main()



