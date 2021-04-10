"""
This code implements a probabilistic matrix factorization (PMF) per weeks 10 and 11 assignment of the machine learning module part of Columbia University Micromaster programme in AI. 
Written using Python 3.7.

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
    - users: list of the users ids
    - objects: list of the objects ids

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


def PMF(train_data, headers = ['user_id', 'movie_id'], lam:int = 2, sigma2:float = 0.1, d:int = 5, iterations:int = 50, output_iterations:list=[10,25,50]):
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

    - L: Loss
    - U_matrices: matrices of users
    - V_matrices: matrices of objects

    """

    L_results = []
    U_matrices = {}
    V_matrices = {}
    log_aps = []

    # first convert dataframe to the ratings matrix as a sparse matrix
    M, n, m, users, objects, rows, cols = df_to_ratings_matrix(train_data, headers = headers)

    parameters = initialize_parameters(lam, n, m, d)

    for i in range(1, iterations + 1):
        new_parameters = update_parameters(M, parameters, lam, n, m, d)
        log_ap = log_a_posteriori(M, parameters)
        L_results.append(log_ap)

        if i in output_iterations:
            print('Log p a-posteriori at iteration ', i, ':', log_ap)
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


def log_a_posteriori(M, parameters):
    """
    Implements the Log-a posteriori with equation as follows:
    
    L=-\frac 1 2 \left(\sum_{i=1}^N\sum_{j=1}^M(R_{ij}-U_i^TV_j)_{(i,j) \in \Omega_{R_{ij}}}^2+\lambda_U\sum_{i=1}^N\|U_i\|_{Fro}^2+\lambda_V\sum_{j=1}^M\|V_j\|_{Fro}^2\right)

    ------------
    Parameters:
    
    - M: the ratings matrix, as sparse (zeros used to fill the nan, missing values)
    - parameters: a dictionary with the values for: 
        - U: matrix of users
        - V: matrix of objects (movies in this case)
        - lambda_U: value of lambda, per the inputs
        - lambda_V: value of lambda, per the inputs
        
    ------------
    Returns:
    
    - L: the resulting float number from the above equation of 'L'

    """

    lambda_U = parameters['lambda_U']
    lambda_V = parameters['lambda_V']
    U = parameters['U']
    V = parameters['V']
    
    UV = np.dot(U.T, V)
    M_UV = (M[M > 0] - UV[M > 0])
    
    return -0.5 * (np.sum(np.dot(M_UV, M_UV.T)) + lambda_U * np.sum(np.dot(U, U.T)) + lambda_V * np.sum(np.dot(V, V.T)))


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


def predict(M, rows, cols, parameters, user_id, movie_id):
    """
    Predicts the rating value. Note the value has been scaled within the range 0-5.

    ------------
    Parameters:

    - M: the ratings matrix, as sparse (zeros used to fill the nan, missing values)   
    - rows: rows of the matrix M
    - cols: columns of the matrix M
    - parameters: a dictionary with the values for: 
        - U: matrix of users
        - V: matrix of objects (movies in this case)
        - lambda_U: value of lambda, per the inputs
        - lambda_V: value of lambda, per the inputs
    - user_id: id of the users being examined
    - movie_id: id of the objects being rated
   
    ------------
    Returns:
    
    - rating: a float number of the predicted rating for the object and user pair

    """

    U = parameters['U']
    V = parameters['V']
    
    M_ij = U[:, rows[user_id]].T.reshape(1, -1) @ V[:, cols[movie_id]].reshape(-1, 1)

    min_rating = np.min(M)
    max_rating = np.max(M)

    return 0 if max_rating == min_rating else ((M_ij[0][0] - min_rating) / (max_rating - min_rating)) * 5.0


def get_prediction(user_id, movies, M, rows, cols, parameters):
    """
    Obtain a dataframe of users Ids, movies Ids and the predicted rating for a given user Id.
    
    ------------
    Parameters:

    - user_id: the id of the user being examined
    - movies: the list of unique movie Ids
    - M: the ratings matrix, as sparse (zeros used to fill the nan, missing values)   
    - rows: rows of the matrix M
    - cols: columns of the matrix M
    - parameters: a dictionary with the values for: 
        - U: matrix of users
        - V: matrix of objects (movies in this case)
        - lambda_U: value of lambda, per the inputs
        - lambda_V: value of lambda, per the inputs

    ------------
    Returns:
    
    - df_result: a dataframe of users Ids, movies Ids and the predicted rating for a given user Id

    """

    predictions = np.zeros((len(movies), 1))
    df_result = pd.DataFrame(columns=['UserID', 'MovieID', 'Prediction'])

    for i, movie_id in enumerate(movies):
        predictions[i] = predict(M, rows, cols, new_parameters, user_id, movie_id)
        df_row = pd.DataFrame({
            'UserID': user_id,
            'MovieID': movie_id,
            'Prediction': predictions[i]
            })
        df_result = df_result.append(df_row, sort=False)
    
    return df_result

def main():

    #Uncomment next line when running in Vocareum
    #train_data=np.genfromtxt(sys.argv[1], delimiter = ',', skip_header=1)
    train_data = get_data('ratings_sample.csv', path = os.path.join(os.getcwd(), 'datasets'), headers=['user_id', 'movie_id', 'rating'])
    out_iterations = [10, 25, 50]

    # Assuming the PMF function returns Loss L, U_matrices and V_matrices
    L_results, U_matrices, V_matrices, users, movies, new_parameters, M, rows, cols = PMF(train_data, headers = ['user_id', 'movie_id'], lam = 2, sigma2 = 0.1, d = 5, iterations = 50, output_iterations = out_iterations)

    save_outputs_txt(data = [L_results, U_matrices, V_matrices], output_iterations = out_iterations)

    # Not required in Vocareum
    df_results = get_prediction(user_id = 15, movies = movies, M = M, rows = rows, cols = cols, parameters = new_parameters)


if __name__ == '__main__':
    main()




