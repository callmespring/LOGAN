import numpy as np
from scipy.special import expit as sigmoid
import random
import numpy.random as nr


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W

# adj_mat and G_true are lower triangular matrice, the off-diaognal part has a link with probability prob.  
def generate_dag_exam1(num_nodes, prob=[0.02, 0.2], edge_coefficient_range=[0.5, 2.0]):
    adj_mat = np.zeros([num_nodes+2, num_nodes+2])
    G_true = adj_mat
    
    # generate random graph between mediators
    for i in np.arange(2,num_nodes+1):
        adj_mat[i,1:(i+1)] = nr.binomial(1, prob[0], i)
        G_true[i,1:(i+1)] = adj_mat[i,1:(i+1)]*nr.uniform(low=edge_coefficient_range[0], high=edge_coefficient_range[1], size=i)*(2*nr.binomial(1, 0.5, i)-1)
        
    # generate weights for treatment-mediator and mediator-responses    
    adj_mat[1:,0] = nr.binomial(1, prob[1], num_nodes+1)
    G_true[1:,0] = adj_mat[1:,0]*nr.uniform(low=edge_coefficient_range[0], high=edge_coefficient_range[1], size=num_nodes+1)*(2*nr.binomial(1, 0.5, num_nodes+1)-1)
    adj_mat[-1,:-1] = nr.binomial(1, prob[1], num_nodes+1)
    G_true[-1,:-1] = adj_mat[-1,:-1]*nr.uniform(low=edge_coefficient_range[0], high=edge_coefficient_range[1], size=num_nodes+1)*(2*nr.binomial(1, 0.5, num_nodes+1)-1)   
        
    return G_true, adj_mat

def simulate_from_dag_lg(tam, n_sample, mean=0, variance=1):
    num_nodes = len(tam)

    def get_value(i, e):
        if values[i] == None:
            val = e[i]
            for j in range(num_nodes):
                if tam[j][i] != 0.0:
                    val += get_value(j, e) * tam[j][i]
            values[i] = val
            return val
        else:
            return values[i]
    
    simulation_data = []
    for i in range(n_sample):
        errors = np.random.normal(mean, variance, num_nodes)
        values = [None for _ in range(num_nodes)]
        for i in range(num_nodes):
            values[i] = get_value(i, errors)
            
        simulation_data.append(values)
        
    return simulation_data


def simulate_linear_sem0(mu, W, n, sem_type='gauss', noise_scale=1.0):
    """Simulate samples from linear SEM with specified type of noise.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, logistic, poisson
        noise_scale (float): scale parameter of additive noise

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """
    def _simulate_single_equation(X, w):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=noise_scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=noise_scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=noise_scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if np.isinf(n):
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * noise_scale * np.linalg.pinv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    X = np.zeros([n, d])
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j])
    return X+mu

def simulate_linear_sem(mu, W, n, sem_type='gauss', noise_scale=1.0):
    """Simulate samples from linear SEM with specified type of noise.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, logistic, poisson
        noise_scale (float): scale parameter of additive noise

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """
    def _simulate_single_equation(X, w):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=noise_scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=noise_scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=noise_scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if np.isinf(n):
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * noise_scale * np.linalg.pinv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    X = np.zeros([n, d])
    for j in range(0,d):
        X[:, j] = _simulate_single_equation(X[:, 0:j], W[0:j, j])
    return X+mu

def count_accuracy(W_true, W, W_und=None):
    """Compute FDR, TPR, and FPR for B, or optionally for CPDAG = B + B_und.

    Args:
        W_true (np.ndarray): [d, d] ground truth graph
        W (np.ndarray): [d, d] predicted graph
        W_und (np.ndarray): [d, d] predicted undirected edges in CPDAG, asymmetric

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    d = W_true.shape[0]
    # convert to binary adjacency matrix
    B_true = (W_true != 0)
    B = (W != 0)
    B_und = None if W_und is None else (W_und != 0)
    # linear index of nonzeros
    pred_und = None
    if B_und is not None:
        pred_und = np.flatnonzero(B_und)
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    if B_und is not None:
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    if B_und is not None:
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    if B_und is not None:
        pred_size += len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)
    if B_und is not None:
        B_lower += np.tril(B_und + B_und.T)
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}
