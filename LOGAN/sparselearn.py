import numpy as np
import scipy
import numpy.random as nr
import LOGAN.notears as nt
from numpy import linalg as LA
import scipy.linalg as slin
import pycasso

## define penalized regression with mcp penalty
def _mcp_reg(y, x, n_sample, p): # p is the number of columns in x
    if p==1:
        x = x.reshape((n_sample,1))
    lambda_list = np.exp(np.arange(-5,3,0.1))
    for j in range(p):
        x[:,j] = x[:,j] - np.mean(x[:,j])
    std = np.sqrt(np.sum(x * x, axis=0))/np.sqrt(n_sample)
    x = x/std
    mcp = pycasso.Solver(x, y-np.mean(y), penalty="mcp", gamma=1.25, prec=1e-4, lambdas=lambda_list)
    mcp.train()
    BIC = np.zeros(len(lambda_list))
    for k in range(len(lambda_list)):
        BIC[k] = np.sum(np.square(y - np.mean(y) - x @ mcp.coef()['beta'][k])) + \
                        sum(mcp.coef()['beta'][k]!=0)*np.log(n_sample)
    return mcp.coef()['beta'][np.argmin(BIC)]/std

#reestimate the coefficients based on mcp penalized regression
def _refit(x, n_sample, p, W_est): # p is the number of columns in x
    W_est = W_est.transpose()
    W_refit = np.zeros((p,p))
    for j in np.arange(1,p-1):
        Indices = W_est[j,:]!=0
        Indices[0] = True
        Indices = np.append(Indices, False)
        W_refit[j,Indices] = _mcp_reg(x[:,j], x[:,Indices], n_sample, sum(Indices))
    W_refit[p-1,0:-1] = _mcp_reg(x[:,p-1], x[:,0:-1], n_sample, p-1)
    return W_refit

## determine the set of ancestors for each node
def _ancestor(W_est): 
    W_est = W_est.transpose()
    p = len(W_est)
    B0 = np.abs(W_est)>0
    B = B0.copy()
    B_tem = B0.copy()
    for j in range(p):
        B = np.dot(B, B0)
        B_tem = np.maximum(B_tem, B)
    B_final = np.full((p+1,p+1), False)
    B_final[0:-1,0:-1] = B_tem
    B_final[-1,0:-1] = True
    B_final[0:-1,0] = True
    return B_final

## calculate the decorrelated score statistic
def _decor_score(x, W_refit, B, L=1000):
    # B denote the set of ancestors
    n_sample, p = np.shape(x)
    W_ds = W_refit.copy()
    Boot_W = np.zeros((p, p, L))
    for j in range(p):
        x[:,j] = x[:,j] - np.mean(x[:,j])
    for i in np.arange(1,p):
        for j in np.arange(0,p-1):
            if W_refit[i,j]!=0:
                Indices = B[i,:].copy()
                Indices[0] = True
                Indices[i] = False
                Indices[j] = False
                Indices[p-1] = False
                gamma = np.zeros(p)
                if (sum(Indices)>0):
                    gamma[Indices] = _mcp_reg(x[:,j], x[:,Indices], n_sample, sum(Indices))
                tem_vec1 = x[:,j] - x @ gamma
                tem_vec2 = x[:,i] - x @ W_refit[i,:] + x[:,j]*W_refit[i,j]
                W_ds[i,j] = np.dot(tem_vec1, tem_vec2)/np.dot(x[:,j], tem_vec1)
                for l in range(L):
                    Boot_W[i,j,l] = np.dot(tem_vec1, nr.normal(size=n_sample))/np.dot(x[:,j], tem_vec1)
    return W_ds, Boot_W

## calculate W_star
def _W_star(W, p):
    W_star = abs(W).copy()
    W_tem = W_star.copy()
    for j in range(p+1):
        for i in range(len(W)):
            W_star[i, :] = np.amax(np.minimum(np.outer(W_star[i, :], np.ones(len(W))), abs(W)), axis=0)
        W_tem = np.maximum(W_tem, W_star)
    return W_tem.copy()
