"""
LHS implementation. This code was originally published by the following individuals for use with
Scilab:
    Copyright (C) 2012 - 2013 - Michael Baudin
    Copyright (C) 2012 - Maria Christopoulou
    Copyright (C) 2010 - 2011 - INRIA - Michael Baudin
    Copyright (C) 2009 - Yann Collette
    Copyright (C) 2009 - CEA - Jean-Marc Martinez
The present code is a modification of the pyDOE package implementation with some extras, including 
an optional seed argument for experiments reproduction.
"""

import numpy as np
from scipy import spatial
from scipy import stats
from scipy import linalg
from numpy import ma
from typing import Optional

__all__ = ['lhs']


def lhs(n: int, k: int, method: Optional[str]=None, iter:Optional[int]=None, seed: Optional[int]=None, corr_matrix :Optional[np.ndarray]= None) -> np.ndarray:
    """
    Generate a latin-hypercube design
    Parameters
    ----------
    n : int
        The number of factors to generate samples for
    k : int
        The number of samples to generate for each factor
    Optional
    --------
    method : str
        Allowable values are "center" or "c", "maximin" or "m",
        "centermaximin" or "cm", and "correlation" or "corr". If no value
        given, the design is simply randomized.
    iter : int
        The number of iterations in the maximin and correlations algorithms
        (Default: 5).
    seed : np.random.RandomState, int
         The seed and random draws
    corr_matrix : ndarray
         Enforce correlation between factors (only used in lhsmu)
    Returns
    -------
    H : 2d-array
        An n-by-k design matrix that has been normalized so factor values
        are uniformly spaced between zero and one.
    """
    H = None

    if seed is None:
        seed = np.random.RandomState()
    elif not isinstance(seed, np.random.RandomState):
        seed = np.random.RandomState(seed)

    if method is not None:
        if not method.lower() in ('center', 'c', 'maximin', 'm',
                                     'centermaximin', 'cm', 'correlation',
                                     'corr','lhsmu'):
            raise ValueError('Invalid value for "method": {}'.format(method))
    else:
        H = _lhsclassic(n, k, seed)

    if method is None:
        method = 'center'
    if iter is None:
        iter = 5

    if H is None:
        if method.lower() in ('center', 'c'):
            H = _lhscentered(n, k, seed)
        elif method.lower() in ('maximin', 'm'):
            H = _lhsmaximin(n, k, iter, 'maximin', seed)
        elif method.lower() in ('centermaximin', 'cm'):
            H = _lhsmaximin(n, k, iter, 'centermaximin', seed)
        elif method.lower() in ('correlation', 'corr'):
            H = _lhscorrelate(n, k, iter, seed)
        elif method.lower() in ('lhsmu'):
            H = _lhsmu(n, k, corr_matrix, seed)

    return H


def _lhsclassic(n, k, randomstate):
    # generate the intervals
    cut = np.linspace(0, 1, k + 1)    
    
    # fill points uniformly in each interval
    u = randomstate.rand(k, n)
    a = cut[:k]
    b = cut[1:k + 1]
    rdpoints = np.zeros_like(u)
    for j in range(n):
        rdpoints[:, j] = u[:, j]*(b-a) + a
    
    # make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(n):
        order = randomstate.permutation(range(k))
        H[:, j] = rdpoints[order, j]
    
    return H
    

def _lhscentered(n, k, randomstate):
    # generate the intervals
    cut = np.linspace(0, 1, k + 1)    
    
    # fill points uniformly in each interval
    u = randomstate.rand(k, n)
    a = cut[:k]
    b = cut[1:k + 1]
    _center = (a + b)/2
    
    # make the random pairings
    H = np.zeros_like(u)
    for j in range(n):
        H[:, j] = randomstate.permutation(_center)
    
    return H
    

def _lhsmaximin(n, k, iter, lhstype, randomstate):
    maxdist = 0
    
    # maximize the minimum distance between points
    for i in range(iter):
        if lhstype=='maximin':
            Hcandidate = _lhsclassic(n, k, randomstate)
        else:
            Hcandidate = _lhscentered(n, k, randomstate)
        
        d = spatial.distance.pdist(Hcandidate, 'euclidean')
        if maxdist<np.min(d):
            maxdist = np.min(d)
            H = Hcandidate.copy()
    
    return H


def _lhscorrelate(n, k, iter, randomstate):
    mincorr = np.inf
    
    # minimize the components correlation coefficients
    for i in range(iter):
        # generate a random LHS
        Hcandidate = _lhsclassic(n, k, randomstate)
        R = np.corrcoef(Hcandidate.T)
        if np.max(np.abs(R[R!=1]))<mincorr:
            mincorr = np.max(np.abs(R-np.eye(R.shape[0])))
            H = Hcandidate.copy()
    
    return H
 

def _lhsmu(n, k, corr=None, seed=None):

    if seed is None:
        seed = np.random.RandomState()
    elif not isinstance(seed, np.random.RandomState):
        seed = np.random.RandomState(seed)

    I = 5*k

    rdpoints = seed.uniform(size=(I, n))

    dist = spatial.distance.cdist(rdpoints, rdpoints, metric='euclidean')
    D_ij = ma.masked_array(dist, mask=np.identity(I))

    index_rm = np.zeros(I-k, dtype=int)
    i = 0
    while i < I-k:
        order = ma.sort(D_ij, axis=1)

        avg_dist = ma.mean(order[:, 0:2], axis=1)
        min_l = ma.argmin(avg_dist)

        D_ij[min_l, :] = ma.masked
        D_ij[:, min_l] = ma.masked

        index_rm[i] = min_l
        i += 1

    rdpoints = np.delete(rdpoints, index_rm, axis=0)

    if(corr is not None):
        # check if covariance matrix is valid
        assert type(corr) == np.ndarray
        assert corr.ndim == 2
        assert corr.shape[0] == corr.shape[1]
        assert corr.shape[0] == n

        norm_u = stats.norm().ppf(rdpoints)
        L = linalg.cholesky(corr, lower=True)

        norm_u = np.matmul(norm_u, L)

        H = stats.norm().cdf(norm_u)
    else:
        H = np.zeros_like(rdpoints, dtype=float)
        rank = np.argsort(rdpoints, axis=0)

        for l in range(k):
            low = float(l)/k
            high = float(l+1)/k

            l_pos = rank == l
            H[l_pos] = seed.uniform(low, high, size=n)
    return H
