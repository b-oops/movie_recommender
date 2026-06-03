import pandas as pd
import numpy as np
from numpy import linalg as la
import math

############################## helper functions ######################################


############################### similarity measures ####################################

def pearsonSim(inA, inB):
    """Return Pearson similarity between two vectors, ignoring NaNs."""
    # works for row or column vectors
    a = np.ravel(inA).astype(float)
    b = np.ravel(inB).astype(float)

    # mask out missing values
    mask = ~np.isnan(a) & ~np.isnan(b)
    a = a[mask]
    b = b[mask]

    # not enough overlap → neutral similarity
    if a.size < 3:
        return 0.5

    corr = np.corrcoef(a, b)[0, 1]
    return 0.5 + 0.5 * corr


def cosineSim(inA, inB):
    """Return Cosine similarity between two vectors, ignoring NaNs."""
    # works for row or column vectors
    a = np.ravel(inA).astype(float)
    b = np.ravel(inB).astype(float)

    # mask out missing values
    mask = ~np.isnan(a) & ~np.isnan(b)
    a = a[mask]
    b = b[mask]

    # no overlap → neutral similarity
    if a.size == 0:
        return 0.5

    denom = la.norm(a) * la.norm(b)
    if denom == 0:
        return 0.5

    cos = np.dot(a, b) / denom
    return 0.5 + 0.5 * cos

########################### Estimate rating functions ##################################

def standItemEst(matrix, user, simMatrix, item):
    """Returns a predicted rating from user (u) on item (i) based on user's average rating on other films (weighted by film similarity)"""
    # dataMat is assumed to be 2d Numpy array, e.g., representing a user-item rating matrix
    # user is the index of a single user (a row) in the dataMat
    # item is the index of a single item (a colums) in the dataMat
    
    n = np.shape(matrix)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for i in range(n):
        userRating = matrix[user,i]
        if userRating == 0 or math.isnan(userRating): 
            continue
        else: 
            similarity = simMatrix[item, i]
        #print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating

    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

def standUserEst(matrix, user, simMatrix, item): 
    """Returns a predicted rating from user (u) on item (i) based on other users' average rating on that films (weighted by user similarity)"""
    # dataMat is assumed to be 2d Numpy array, e.g., representing a user-item rating matrix
    # user is the index of a single user (a row) in the dataMat
    # item is the index of a single item (a colums) in the dataMat
    
    n = np.shape(matrix)[0]
    simTotal = 0.0; ratSimTotal = 0.0
    for u in range(n):
        userRating = matrix[u,item]
        if userRating == 0 or math.isnan(userRating): 
            continue
        else: 
            similarity = simMatrix[user, u]
        #print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

############################## recommend functions #####################################

def weighted_recommend(matrix, user, simMatrix, N=3, estMethod=standItemEst):
    """Returns n top recommendations for user based on the estimation method and helper similarity matrix provided"""
    unratedItems = np.nonzero(matrix[user,:]==0)[0] #find unrated items 
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(matrix, user, simMatrix, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


# def user_based_recommend(dataMat, user, N=3, simMeas=pearsonSim, estMethod=standUserEst):
#     unratedItems = np.nonzero(dataMat[user,:]==0)[0] #find unrated items 
#     if len(unratedItems) == 0: return 'you rated everything'
#     itemScores = []
#     for item in unratedItems:
#         estimatedScore = estMethod(dataMat, user, simMeas, item)
#         itemScores.append((item, estimatedScore))
#     return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

def simple_recommend(df: pd.DataFrame, unrated: set, n: int) -> pd.DataFrame:
    """
    Simply returns the n top unrated films based on the user ratings matrix.
    """
    means = df.mean(axis=0)                       # get mean ratings
    print(type(means))
    means = means[means.index.isin(unrated)]          # filter out user rated movies
    top = means.nlargest(n)                       # get sorted df

    return top

