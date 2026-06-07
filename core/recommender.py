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

def kNearestItemEst_topn(matrix, user, simMatrix, item, k=20, verbose=False, user_labels=None, movie_labels=None):
    """Returns a predicted rating from user (u) on item (i) based on user's average rating on k most similar films (weighted by film similarity)"""
    neighbors = []
    row = matrix[user]
    rated = row[row != 0]
    user_mean = rated.mean()

    for i, similarity in simMatrix[item]:
        userRating = matrix[user, i]
        if userRating == 0 or np.isnan(userRating):
            continue
        neighbors.append((similarity, userRating, i))

        if verbose:
            print(f"User: {user_labels[user]}, item: {movie_labels[i]}, "
                  f"sim: {similarity}, rating: {userRating}")

    if not neighbors:
        return user_mean

    neighbors.sort(key=lambda x: x[0], reverse=True)
    neighbors = neighbors[:k]

    if len(neighbors) < 5:
        return user_mean

    simTotal = sum(sim for sim, _, _ in neighbors)
    if simTotal == 0:
        return user_mean

    ratSimTotal = sum(sim * rating for sim, rating, _ in neighbors)
    return ratSimTotal / simTotal

def kNearestUserEst(matrix, user, simMatrix, item, k=20, min_sim=0.3, verbose: bool=False, user_labels=None, movie_labels=None): 
    """Returns a predicted rating from user (u) on item (i) based on k most similar other users' average rating on that film (weighted by user similarity)"""
    # dataMat is assumed to be 2d Numpy array, e.g., representing a user-item rating matrix
    # user is the index of a single user (a row) in the dataMat
    # item is the index of a single item (a colums) in the dataMat
    
    n = np.shape(matrix)[0]
    simTotal = 0.0; ratSimTotal = 0.0

    neighbors = []
    for u in range(n):
        userRating = matrix[u,item]
        if u == user:
            continue
        if userRating == 0 or np.isnan(userRating):
            continue
        similarity = simMatrix[user, u]
        if similarity > min_sim:
            if verbose:
                print(f"User: {user_labels[u]}, item: {movie_labels[item]}, user_sim: {similarity}, rating: {userRating}")
            neighbors.append((similarity, userRating))

    if not neighbors:
        return 0
    
    # keep only top-k most similar items
    neighbors.sort(key=lambda x: x[0], reverse=True)
    neighbors = neighbors[:k]

    simTotal = sum(sim for sim, _ in neighbors)

    if simTotal == 0: 
        return 0

    ratSimTotal = sum(sim * rating for sim, rating in neighbors)

    return ratSimTotal/simTotal

############################## recommend functions #####################################

def weighted_recommend(matrix, user, simMatrix, estMethod, N=3, ratio=.5, simMatrix2=None, estMethod2=None, rewatch=False):
    """Returns n top recommendations for user based on the estimation method(s) and helper similarity matrix(es) specified"""
    
    ## what movies are we picking from?
    itemsToRate = np.nonzero(matrix[user,:]==0)[0] #find unrated items 
    if rewatch:
        itemsToRate = np.nonzero(matrix[user,:]>0)[0]
        if len(itemsToRate) == 0: return "you haven't rated anything"
    else:
        itemsToRate = np.nonzero(matrix[user,:]==0)[0] #find unrated items 
        if len(itemsToRate) == 0: return 'you rated everything: try a rewatch!'

    ## which movies should we pick?
    itemScores = []
    for item in itemsToRate:
        ## just one estimator provided
        estimatedScore = estMethod(matrix, user, simMatrix, item)
        ## two estimation methods and a ratio to split them
        if simMatrix2 is not None:
            estimatedScore2 = estMethod2(matrix, user, simMatrix2, item)
            estimatedScore = estimatedScore*(ratio) + estimatedScore2*(1-ratio)
        itemScores.append((item, estimatedScore))
        
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]




