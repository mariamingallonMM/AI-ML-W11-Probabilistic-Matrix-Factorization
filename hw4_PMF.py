"""
This code implements a probabilistic matrix factorization (PMF) per weeks 10 and 11 assignment of the machine learning module part of Columbia University Micromaster programme in AI. 

Written using Python 3.7 and adjusted to ensure it runs on Vocareum.

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


def PMF(train_data, headers = ['user_id', 'movie_id'], lam:int = 2, sigma2:float = 1/10, d:int = 5, iterations:int = 50, output_iterations:list=[10,25,50]):
    """
    Implements Probabilistic Matrix Factorization.

    ------------
    Parameters:

    - data: dataset used for training (e.g. the ratings.csv dataset with missing values for users and movies).
    - headers: title of the headers in the dataset for the 'users id' and 'movie id' values.
    - lam: lambda value to initialise the Gaussian zero mean distribution (default lam = 2 for this assignment).
    - sigma2: covariance of the Gaussian (default sigma2 = 0.1 for this assignment).
    - d: number of dimensions for the ranking, (default d = 5 for this assignment).
    - iterations: number of iterations to run PMF for (default, 50 iterations).
    
    ------------
    Returns:

    - L_results: results from calculating the objective function ('L')
    - U_matrices: matrices of users
    - V_matrices: matrices of objects

    """

    L_results = []
    U_matrices = {}
    V_matrices = {}
    log_aps = []

    # add a header row to the train_data plain csv input file
    train_data = pd.DataFrame(train_data, columns = ['user_id', 'movie_id', 'rating'])

    # first convert dataframe to the ratings matrix as a sparse matrix
    M, n, m, users, objects, rows, cols = df_to_ratings_matrix(train_data, headers = headers)

    parameters = initialize_parameters(lam, n, m, d)

    for i in range(1, iterations + 1):
        new_parameters = update_parameters(M, parameters, lam, n, m, d)
        L = objective_function(M, sigma2, lam, parameters)
        L_results.append(L)

        # TODO: check results from U_matrices and V_matrices
        # U_matrices are exporting all same for all iterations
        # V_matrices do not show the correct size of 5 columns, 20 rows
        if i in output_iterations:
            print('Objective function L at iteration ', i, ':', L)
            U_matrices[i] = new_parameters['U']
            V_matrices[i] = new_parameters['V']

    return L_results, U_matrices, V_matrices, users, objects, new_parameters, M, rows, cols



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

    - parameters: a dictionary with the values for: 
        - U: matrix of users
        - V: matrix of objects (movies in this case)
        - lambda_U: value of lambda, per the inputs
        - lambda_V: value of lambda, per the inputs

    """

    U = np.zeros((d, n), dtype=np.float64)
    V = np.random.normal(0.0, 1.0 / lam, (d, m))
    
    parameters = {}

    parameters['U'] = U
    parameters['V'] = V
    parameters['lambda_U'] = lam
    parameters['lambda_V'] = lam
    
    return parameters


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
    - n: number of rows
    - m: number of columns
    - users: list of unique users
    - movies: list of unique movies
    - rows: rows of the matrix M
    - cols: columns of the matrix M

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
    
    return M, n, m, users, movies, rows, cols


def update_parameters(M, parameters, lam, n, m, d):
    """
    Implements the function that updates U and V.

    ------------
    Parameters:
    
    - M: the ratings matrix, as sparse (zeros used to fill the nan, missing values)   
    - parameters: a dictionary with the values for: 
        - U: matrix of users
        - V: matrix of objects (movies in this case)
        - lambda_U: value of lambda, per the inputs
        - lambda_V: value of lambda, per the inputs    
    - lam: lambda value to initialise the Gaussian zero mean distribution (default lam = 2 for this assignment).
    - n: number of users in dataset
    - m: number of movies in dataset
    - d: number of dimensions for the ranking, (default d = 5 for this assignment).
    
    ------------
    Returns:

    - parameters: a dictionary with the values for: 
        - U: matrix of users
        - V: matrix of objects (movies in this case)
        - lambda_U: value of lambda, per the inputs
        - lambda_V: value of lambda, per the inputs
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

    min_rating = np.min(M)
    max_rating = np.max(M)

    return parameters


def objective_function(M, sigma2, lam, parameters):
    """
    Calculates the result of the objective function 'L' with equation as follows:
    L = − ∑(i,j)∈Ω12σ2(Mij−uTivj)2 − ∑Nui=1λ2∥ui∥2 − ∑Nvj=1λ2∥vj∥2
    ------------
    Parameters:
    
    - M: the ratings matrix, as sparse (zeros used to fill the nan, missing values)
    - sigma2:
    - lam: 
    - parameters: a dictionary with the values for: 
        - U: matrix of users
        - V: matrix of objects (movies in this case)
        - lambda_U: value of lambda, per the inputs
        - lambda_V: value of lambda, per the inputs
        
    ------------
    Returns:
    
    - L: the resulting float number from calculating the objective function based on the above equation of 'L'

    """

    lambda_U = parameters['lambda_U']
    lambda_V = parameters['lambda_V']
    U = parameters['U']
    V = parameters['V']

    # We divide L equation into its three main summands

    UV = np.dot(U.T, V) # uTivj
    M_UV = (M[M > 0] - UV[M > 0]) # (Mij−uTivj)

    L1 = - (1 / (2 * sigma2)) * (np.sum((M_UV)**2))
    L2 = - (lambda_U / 2 ) * (np.sum(np.linalg.norm(U)**2))
    L3 = - (lambda_V / 2 ) * (np.sum(np.linalg.norm(V)**2))

    L = L1 + L2 + L3
    #L = -0.5 * (sigma2)* (np.sum(np.dot(M_UV, M_UV.T)) + lambda_U * np.sum(np.dot(U, U.T)) + lambda_V * np.sum(np.dot(V, V.T)))
    
    return L


def save_outputs_txt(data, output_iterations:list = [5, 10, 25]):
    """
    Write the outputs to csv files.

    ------------
    Parameters:

    - data: a list of the resulting matrixes to write as outputs.
    - output_iterations: the iterations to store as output csv files for the U and V matrixes.
    
    ------------
    Returns:

    - csv files with the output data

    """

    L_results = data[0]
    np.savetxt("objective.csv", L_results, delimiter=",")

    U_results = data[1]
    V_results = data[2]

    for i in output_iterations:
        filename = "U-" + str(i) + ".csv"
        np.savetxt(filename, U_results[i].T, delimiter=",")
        filename = "V-" + str(i) + ".csv"
        np.savetxt(filename, V_results[i].T, delimiter=",")

    return


def main():

    train_data=np.genfromtxt(sys.argv[1], delimiter = ',', skip_header=1)

    # Call the PMF function to return the outputs
    L_results, U_matrices, V_matrices, users, movies, new_parameters, M, rows, cols = PMF(train_data, headers = ['user_id', 'movie_id'], lam = 2, sigma2 = 0.1, d = 5, iterations = 50, output_iterations = [10, 25, 50])

    save_outputs_txt(data = [L_results, U_matrices, V_matrices], output_iterations = [10, 25, 50])


if __name__ == '__main__':
    main()

