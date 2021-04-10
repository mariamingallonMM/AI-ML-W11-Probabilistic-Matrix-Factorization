# AI-ML-W11-Probabilistic-Matrix-Factorization
ColumbiaX CSMM.102x Machine Learning Course. Week 11 Assignment to implement the probabilistic matrix factorization (PMF) model. 


## Instructions

In this assignment, we will implement the probabilistic matrix factorization (PMF) model. PMF works as follows:
* fills in the values of a missing matrix M, 
* where Mij is an observed value if (i,j)∈Ω and
* where Ω contains the measured pairs. 

The goal of PMF is to factorize this matrix into a product between vectors such that Mij ≈ uTi vj, where each ui,vj∈Rd. The modeling problem is to learn ui for i=1,…,Nu and vj for j=1,…,Nv by maximizing the objective function, which can be formulated as follows:

![equation 1: L=−∑(i,j)∈Ω12σ2(Mij−uTivj)2−∑Nui=1λ2∥ui∥2−∑Nvj=1λ2∥vj∥2](./ref/eq1.JPG?raw=true)

For this assignment we are asked to set d, sigma and lambda to a particular value as follows:
* d=5, dimensions of the rank
* σ2=110, covariance of Gaussian distribution
* λ=2, lambda of Gaussian distribution 

## Execute the program 

The following command will execute your program:

`$ python3 hw4_PMF.py ratings.csv`

The input csv file ('ratings.csv') is a comma separated file containing three columns: user_index, object_index, and rating.

## Expected outputs from the program

The PMF algorithm must learn 5 dimensions (d=5) and shall be run for 50 iterations. 

When executed, the program writes several output files each described below.

* objective.csv: This is a comma separated file containing the PMF objective function given above along each row. There should be 50 rows (one per iteration) and each row should have one value.

* U-[iteration].csv: This is a comma separated file containing the locations corresponding to the rows, or "users", of the missing matrix M. The ith row should contain the ith user's vector (5 values as d=5). We only need to create this file for iteration number 10, 25, and 50. For example, the 10th iteration will produce file U-10.csv

* V-[iteration].csv: This is a comma separated file containing the locations corresponding to the columns, or "objects", of the missing matrix M. The jth row should contain the jth object's vector (5 values as d=5). We only need to create this file for iteration number 10, 25, and 50. For example, the 10th iteration will produce file V-10.csv

## How it works

Probabilistic Matrix Factorization (PMF) is used commonly for **collaborative filtering**. The latter is used as an alternative to **content-based filtering** when there is not enough information provided by a user to make suggestions. While **content-based filtering makes use of the user's explicitly expressed preferences, **collaborative filtering** uses the history data provided by a group of users with similar preferences to make elicit recommendations. 

Whereas in content-based filering we expect for a given user to build a profile that clearly states preferences, in collaborative-filtering this information may not be fully available, but we expect our system to still be able to make recommendations based on evidence that similar users provide. 

The following is how we have implemented Probabilistic Matrix Factorization for building a movie recommendation system using collaborative filtering:

1. Transform input ratings.csv to **M matrix**, of n rows and m columns, where each row is a user and each column is a movie. Where we don't have data, we will use a '0' instead of NaN. Users and movies shall be indexed from 1 (not '0').
2. We estimate the M matrix by using **two low-rank matrices U and V** as: M = UT x V, where:
	a. UT is the transposed matrix of U. UT is an n x d matrix, where n is the number of users (rows of M), and d is the rank (d fixed to 5 in this assignment).
	b. V is a d x m matrix, where m is the number of movies to rate (columns in M).
3. We will use MAP inference coordinate ascent algorithm to estimate the missing ratings of 5 users for 5 movies not already rated in the starting dataset.
4. First, we will initialize each vj with a normal distribution of zero mean and covariance equal to the inverse of lambda multiplied by the identity matrix.
5. For each iteration, we update ui and then vj.


## Notes on data repositories

- We are using the **[ratings_sample.csv](./datasets/ratings_sample.csv)** dataset provided with the assignment.

## Citations & References

- [PMF for Recommender Systems](https://towardsdatascience.com/pmf-for-recommender-systems-cbaf20f102f0) by [Oscar Contreras Carrasco](https://medium.com/@OscarContrerasC)
