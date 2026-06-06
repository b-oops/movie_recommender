import pandas as pd
import numpy as np
from numpy import linalg as la
import math

############################## helper functions ######################################


############################### similarity measures ####################################

def pearsonSim(inA, inB):
    """Return Pearson similarity between two vectors, ignoring NaNs."""
    a = np.ravel(inA).astype(float)
    b = np.ravel(inB).astype(float)

    mask = ~np.isnan(a) & ~np.isnan(b)

    a = a[mask]
    b = b[mask]

    ## pearson breaks if there are not enough points
    if len(a) < 3:
        return 0.5

    ## pearson breaks if there is no variance for a vector
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.5

    corr = np.corrcoef(a,b)[0,1]

    # significance weighting
    corr *= len(a)/(len(a)+10)

    return 0.5 + 0.5*corr


def cosineSim(inA, inB):
    """Return Cosine similarity between two vectors, ignoring NaNs."""
    # works for row or column vectors
    a = np.ravel(inA).astype(float)
    b = np.ravel(inB).astype(float)

    mask = ~np.isnan(a) & ~np.isnan(b)

    a = a[mask]
    b = b[mask]

    if len(a) < 3:
        return 0.5


    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.5

    # significance weighted
    cos = np.dot(a,b)/denom
    cos *= len(a)/(len(a)+10)

    return 0.5 + 0.5*cos

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

def kNearestItemEst_topn(matrix, user, topn, item, k=30):
    """
    Predict rating using only the precomputed Top-N neighbors for each item.
    """
    neighbors = []

    for i, similarity in topn[item]:
        userRating = matrix[user, i]
        if userRating == 0 or np.isnan(userRating):
            continue

        neighbors.append((similarity, userRating, i))

    if not neighbors:
        return 0

    # Sort and keep top-k
    neighbors.sort(key=lambda x: x[0], reverse=True)
    neighbors = neighbors[:k]

    if len(neighbors) < 5: # need at least 5 neighbors to feel good about predicting with them
        return 5 # neutral 

    simTotal = sum(sim for sim, _, _ in neighbors)
    if simTotal == 0:
        return 0

    ratSimTotal = sum(sim * rating for sim, rating, _ in neighbors)
    return ratSimTotal / simTotal

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




