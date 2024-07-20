import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from IPython.display import clear_output
import random
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.pipeline import make_pipeline
import timeit
np.random.seed(90)
random.seed(90)

def bound(p, m, n, k, s, sigma):
    term1 = 8 * sigma * np.sqrt(k * np.log(p) / m) + 8 * sigma * np.sqrt(k * np.log(p) / n) 
    term2 = 4 * sigma * np.sqrt(s * np.log(n) / n)

    return max(term1, term2)

def sgn(u):
    res = 2 * ((u > 0) - 0.5)
    return res

def sparsePermutation(N, m, noPermuted):
    '''
    Returns a permutation matrix of size (N, N) with 
    top m rows same as that of an identity matrix.
    The permutation is in the lower rows.
    '''
    to_be_permuted = random.sample(range(m, N), noPermuted)
    P = np.eye(N)
    for i in range(noPermuted // 2):
        a, b = to_be_permuted[2*i], to_be_permuted[2*i+1] 
        temp = P[a].copy()
        P[a] = P[b].copy()
        P[b] = temp.copy()

    return P

def addNoise(y, noiseFrac):
      
    '''
    y: (N, 1)
    Adds Gaussian noise to the measurements vector y 
    and returns the noisy measurement
    '''
    N = y.shape[0]
    noise_std = noiseFrac * np.mean(np.abs(y))
    noise = np.random.normal(0, noise_std, size = (N, 1))
    noisy_y = y + noise
    return noisy_y

def meanNormalizedError(x, xhat):
    '''
    x, xhat: (p, 1)
    '''
    return np.linalg.norm(x - xhat)/np.linalg.norm(x)

def RLWKC_without_constraint(A, y, A_cv, y_cv, m, r1 = None, r2 = None):
    '''
    Implements Robust Lasso with known correspondences.
    A: (m + n, p) and y: (m + n, 1), where N = m + n
    m: number of known correspondences
    n: number of unknown correspondences

    A_cv: (cv, p)  y_cv: (cv, 1)
    '''

    N = A.shape[0]
    p = A.shape[1]
    n = N - m

    A1 = A[:m].copy()
    y1 = y[:m].copy()

    A2 = A[m:].copy()
    y2 = y[m:].copy()
    
    currLeastError  = float('inf')
    candidateLambda1 = np.array([r1]) if r1 is not None else np.arange(0.001, 0.900, 0.020) 
    candidateLambda2 = np.array([r2]) if r2 is not None else np.arange(0.001, 0.900, 0.020)

    crossvalidationErrorList = np.zeros(shape = (candidateLambda1.shape[0], candidateLambda2.shape[0]))

    for i, lambda1 in enumerate(candidateLambda1):
        for j, lambda2 in enumerate(candidateLambda2):

            t_0 = timeit.default_timer()
            # print(i, j, candidateLambda1.shape[0])
            x = cp.Variable(shape = (p, 1))
            z = cp.Variable(shape = (n, 1))
            objective = cp.Minimize(cp.sum_squares(y1 - A1 @ x) + cp.sum_squares(y2 - A2 @ x - z) + lambda1*cp.norm1(x) + lambda2*cp.norm1(z))
            prob = cp.Problem(objective)
            result = prob.solve(solver = cp.ECOS, verbose = False)
            x = x.value
            x = x.reshape((p, 1))
            z = z.value
            z = z.reshape((n, 1))

            t_1 = timeit.default_timer()
            
            crossValidationError = np.linalg.norm(A_cv @ x - y_cv)
            crossvalidationErrorList[i][j] = crossValidationError
            if crossValidationError < currLeastError:
                currLeastError = crossValidationError
                xhat = x
                zhat = z
                usedlambda1 = lambda1
                usedlambda2 = lambda2
                t = t_1 - t_0
    
    if usedlambda1 <= 0.004 or usedlambda1 >= 0.870:
        print("Choose lambda1 properly")
    if usedlambda2 <= 0.004 or usedlambda2 >= 0.870:
        print("Choose lambda2 properly")
    return xhat, usedlambda1, usedlambda2, t

def RLWKC_with_constraint(A, y, A_cv, y_cv, m, r1 = None, r2 = None):
    '''
    Implements Robust Lasso with known correspondences.
    A: (m + n, p) and y: (m + n, 1), where N = m + n
    m: number of known correspondences
    n: number of unknown correspondences

    A_cv: (cv, p)  y_cv: (cv, 1)
    '''

    N = A.shape[0]
    p = A.shape[1]
    n = N - m

    A1 = A[:m].copy()
    y1 = y[:m].copy()

    A2 = A[m:].copy()
    y2 = y[m:].copy()
    
    currLeastError  = float('inf')
    candidateLambda1 = np.array([r1]) if r1 is not None else np.arange(0.001, 0.900, 0.020) 
    candidateLambda2 = np.array([r2]) if r2 is not None else np.arange(0.001, 0.900, 0.020)

    crossvalidationErrorList = np.zeros(shape = (candidateLambda1.shape[0], candidateLambda2.shape[0]))

    for i, lambda1 in enumerate(candidateLambda1):
        for j, lambda2 in enumerate(candidateLambda2):
            # print(i, j, candidateLambda1.shape[0])
            x = cp.Variable(shape = (p, 1))
            z = cp.Variable(shape = (n, 1))
            objective = cp.Minimize(cp.sum_squares(y1 - A1 @ x) + cp.sum_squares(y2 - A2 @ x - z) + lambda1*cp.norm1(x) + lambda2*cp.norm1(z))
            constraint = [cp.sum(z) == 0]
            prob = cp.Problem(objective, constraint)
            result = prob.solve(solver = cp.ECOS, verbose = False)
            x = x.value
            x = x.reshape((p, 1))
            z = z.value
            z = z.reshape((n, 1))
            
            crossValidationError = np.linalg.norm(A_cv @ x - y_cv)
            crossvalidationErrorList[i][j] = crossValidationError
            if crossValidationError < currLeastError:
                currLeastError = crossValidationError
                xhat = x
                zhat = z
                usedlambda1 = lambda1
                usedlambda2 = lambda2
    
    if usedlambda1 <= 0.004 or usedlambda1 >= 0.870:
        print("Choose lambda1 properly")
    if usedlambda2 <= 0.004 or usedlambda2 >= 0.870:
        print("Choose lambda2 properly")
    return xhat, usedlambda1, usedlambda2


def RL_without_constraint(A, y, A_cv, y_cv, r1 = None, r2 = None):
    '''
    Implements Robust Lasso with known correspondences.
    A: (m + n, p) and y: (m + n, 1), where N = m + n
    m: number of known correspondences
    n: number of unknown correspondences

    A_cv: (cv, p)  y_cv: (cv, 1)
    '''

    N = A.shape[0]
    p = A.shape[1]
    
    currLeastError  = float('inf')
    candidateLambda1 = np.array([r1]) if r1 is not None else np.arange(0.001, 0.900, 0.020) 
    candidateLambda2 = np.array([r2]) if r2 is not None else np.arange(0.001, 0.900, 0.020)

    crossvalidationErrorList = np.zeros(shape = (candidateLambda1.shape[0], candidateLambda2.shape[0]))

    for i, lambda1 in enumerate(candidateLambda1):
        for j, lambda2 in enumerate(candidateLambda2):
            # print(i, j, candidateLambda1.shape[0])

            t_0 = timeit.default_timer()
            x = cp.Variable(shape = (p, 1))
            z = cp.Variable(shape = (N, 1))
            objective = cp.Minimize(cp.sum_squares(y - A @ x - z) + lambda1*cp.norm1(x) + lambda2*cp.norm1(z))
            prob = cp.Problem(objective)
            result = prob.solve(solver = cp.ECOS, verbose = False)
            x = x.value
            x = x.reshape((p, 1))
            z = z.value
            z = z.reshape((N, 1))

            t_1 = timeit.default_timer()
            
            crossValidationError = np.linalg.norm(A_cv @ x - y_cv)
            crossvalidationErrorList[i][j] = crossValidationError
            if crossValidationError < currLeastError:
                currLeastError = crossValidationError
                xhat = x
                zhat = z
                usedlambda1 = lambda1
                usedlambda2 = lambda2
                t = t_1 - t_0
    
    if usedlambda1 <= 0.004 or usedlambda1 >= 0.870:
        print("Choose lambda1 properly")
    if usedlambda2 <= 0.004 or usedlambda2 >= 0.870:
        print("Choose lambda2 properly")
    return xhat, usedlambda1, usedlambda2, t


def RL_with_constraint(A, y, A_cv, y_cv, r1 = None, r2 = None):
    '''
    Implements Robust Lasso with known correspondences.
    A: (m + n, p) and y: (m + n, 1), where N = m + n
    m: number of known correspondences
    n: number of unknown correspondences

    A_cv: (cv, p)  y_cv: (cv, 1)
    '''

    N = A.shape[0]
    p = A.shape[1]
    
    currLeastError  = float('inf')
    candidateLambda1 = np.array([r1]) if r1 is not None else np.arange(0.001, 0.900, 0.020) 
    candidateLambda2 = np.array([r2]) if r2 is not None else np.arange(0.001, 0.900, 0.020)

    crossvalidationErrorList = np.zeros(shape = (candidateLambda1.shape[0], candidateLambda2.shape[0]))

    for i, lambda1 in enumerate(candidateLambda1):
        for j, lambda2 in enumerate(candidateLambda2):
            # print(i, j, candidateLambda1.shape[0])
            x = cp.Variable(shape = (p, 1))
            z = cp.Variable(shape = (N, 1))
            objective = cp.Minimize(cp.sum_squares(y - A @ x - z) + lambda1*cp.norm1(x) + lambda2*cp.norm1(z))
            constraint = [cp.sum(z) == 0]
            prob = cp.Problem(objective, constraint)
            result = prob.solve(solver = cp.ECOS, verbose = False)
            x = x.value
            x = x.reshape((p, 1))
            z = z.value
            z = z.reshape((N, 1))
            
            crossValidationError = np.linalg.norm(A_cv @ x - y_cv)
            crossvalidationErrorList[i][j] = crossValidationError
            if crossValidationError < currLeastError:
                currLeastError = crossValidationError
                xhat = x
                zhat = z
                usedlambda1 = lambda1
                usedlambda2 = lambda2
    
    if usedlambda1 <= 0.004 or usedlambda1 >= 0.870:
        print("Choose lambda1 properly")
    if usedlambda2 <= 0.004 or usedlambda2 >= 0.870:
        print("Choose lambda2 properly")
    return xhat, usedlambda1, usedlambda2

def RobustRegression(A, y):
    '''
    Implements Robust Regression
    A: (N, p) and y: (N, 1)
    '''

    N = A.shape[0]
    p = A.shape[1]
    
    x = cp.Variable(shape = (p, 1))   
    objective = cp.Minimize(cp.norm1(y - A @ x))
    prob = cp.Problem(objective)
    result = prob.solve(solver = cp.ECOS, verbose = False)
    x = x.value
    x = x.reshape((p))
      
    return x # [p, 1]


def HTP(A, y, A_cv, y_cv, n_interations = 200):

    p = A.shape[1]
    currLeastError  = float('inf')
    res = None

    # k_range = [i for i in range(1, 25)] 
    mu_range = [i * (1e-1) for i in range(1, 10, 2)] + [i * (1e-2) for i in range(1, 10, 2)] + [i * (1e-3) for i in range(1, 10, 2)] 


    for k in [28]:
        for mu in mu_range:

            t_0 = timeit.default_timer()

            x_hat = np.zeros(shape = (p, 1))

            f_best = float('inf')
            x_best = np.zeros(shape = (p, 1))

            
            for iteration in range(n_interations):

                if iteration == 0:
                    x_hat = mu * np.transpose(A) @ sgn(y) 
                else:
                    x_hat = x_hat - mu * np.transpose(A) @ sgn(A[:, non_zero_support] @ x_hat[non_zero_support, :] - y) 

                x_hat = x_hat.reshape((-1))

                zero_support = np.argpartition(np.abs(x_hat), p - k)[:(p - k)] # minimum (p - k)
                non_zero_support = np.argpartition(np.abs(x_hat), -k)[-k: ]    # maximum k

                x_hat[zero_support] = 0
                x_hat[non_zero_support] = RobustRegression(np.copy(A[:, non_zero_support]), y)

                x_hat = x_hat.reshape((p, 1))

                f_hat = np.linalg.norm((A[:, non_zero_support] @ x_hat[non_zero_support , :] - y).reshape((-1)), ord = 1)
                if f_hat < f_best:
                    f_best = f_hat
                    x_best = np.copy(x_hat)

            t_1 = timeit.default_timer()
                        
            crossValidationError = np.linalg.norm(A_cv @ x_best - y_cv)
            if crossValidationError < currLeastError:
                currLeastError = crossValidationError
                res = x_best
                used_k = k
                used_mu = mu
                t = t_1 - t_0

    return res, used_k, used_mu, t

def L1_L2_regression(A1, y1, A2, y2):

    '''
    Returns x = arg min ||y1 - A1 x||_2^2 + ||y2 - A2 x||_1 
    x = (p, 1)

    A1 = (m, p)
    y1 = (m, 1)

    A2 = (n, p)
    y2 = (n, 1)
    '''

    p = A1.shape[1]

    x = cp.Variable(shape = (p, 1))   
    objective = cp.Minimize(cp.sum_squares(y1 - A1 @ x) + cp.norm1(y2 - A2 @ x))
    prob = cp.Problem(objective)
    result = prob.solve(solver = cp.ECOS, verbose = False)
    x = x.value
    x = x.reshape((p))
      
    return x # [p]




# def HTP_with_priors(A, y, A_cv, y_cv, m, k_range, mu_range, n_interations = 600):

#     A1 = A[:m].copy()
#     y1 = y[:m].copy()

#     A2 = A[m:].copy()
#     y2 = y[m:].copy()

#     p = A.shape[1]
#     currLeastError  = float('inf')
#     res = None

#     k_range = [i for i in range(1, 25)] if k_range is None else [k_range]
#     mu_range = [i * (1) for i in range(1, 3)] + [i * (1e-1) for i in range(1, 10)] + [i * (1e-2) for i in range(1, 10)] + [i * (1e-3) for i in range(1, 10)] + [i * (1e-4) for i in range(1, 10)] if mu_range is None else [mu_range] 


#     for k in [18]:
#         for mu in [1e-2]:

#             x_hat = np.zeros(shape = (p, 1))

#             f_best = float('inf')
#             x_best = np.zeros(shape = (p, 1))

            
#             for iteration in range(n_interations):

#                 if iteration == 0:
#                     x_hat = mu * (np.eye(p) - np.transpose(A1) @ np.linalg.inv(A1 @ np.transpose(A1)) @ A1) @ (np.transpose(A2) @ sgn(y2))
#                 else:
#                     x_hat = x_hat - mu * (np.eye(p) - np.transpose(A1) @ np.linalg.inv(A1 @ np.transpose(A1)) @ A1) @ (np.transpose(A2) @ sgn(A2[:, non_zero_support] @ x_hat[non_zero_support, :] - y2))

#                 x_hat = x_hat.reshape((-1))

#                 zero_support = np.argpartition(np.abs(x_hat), p - k)[:(p - k)] # minimum (p - k)
#                 non_zero_support = np.argpartition(np.abs(x_hat), -k)[-k: ]    # maximum k

#                 x_hat[zero_support] = 0
#                 x_hat[non_zero_support] = RobustRegression(np.copy(A[:, non_zero_support]), y)

#                 x_hat = x_hat.reshape((p, 1))

#                 f_hat = np.linalg.norm((A[:, non_zero_support] @ x_hat[non_zero_support , :] - y).reshape((-1)), ord = 1)
#                 if f_hat < f_best:
#                     f_best = f_hat
#                     x_best = np.copy(x_hat)

                        
#             crossValidationError = np.linalg.norm(A_cv @ x_best - y_cv)
#             if crossValidationError < currLeastError:
#                 currLeastError = crossValidationError
#                 res = x_best
#                 used_k = k
#                 used_mu = mu

#     return res, used_k, used_mu

# def OMP(A, y, m):

#     A1 = A[:m].copy()
#     y1 = y[:m].copy()

#     A2 = A[m:].copy()
#     y2 = y[m:].copy()

    

#     p = A.shape[1]
#     n = A2.shape[0]
#     currLeastError  = float('inf')
#     res = None
#     W = np.concatenate((np.concatenate((A1, np.zeros((m, n))), axis = 1), np.concatenate((A2, np.eye(n)), axis = 1)), axis = 0)

#     omp = OrthogonalMatchingPursuit(n_nonzero_coefs = 14 + 16 + 30)
#     model = make_pipeline(StandardScaler(with_mean=False), OrthogonalMatchingPursuit(n_nonzero_coefs = 14 + 16 + 30))
#     model.fit(W, y)
#     return omp.coef_[:p]


def HTP_with_priors(A, y, A_cv, y_cv, m, n_interations = 100):

    A1 = A[:m].copy()
    y1 = y[:m].copy()

    A2 = A[m:].copy()
    y2 = y[m:].copy()

    

    p = A.shape[1]
    n = A2.shape[0]
    currLeastError  = float('inf')
    res = None
    W = np.concatenate((np.concatenate((A1, np.zeros((m, n))), axis = 1), np.concatenate((A2, np.eye(n)), axis = 1)), axis = 0)

    # k_range = [i for i in range(1, 25, 2)] if k_range is None else [k_range]
    # s_range = [i for i in range(1, 51, 2)] if s_range is None else [s_range]

    for s in [32]:
        for k in [28]:
            for mu in [i * (1e-3) for i in range(1, 101, 2)]:

                t_0 = timeit.default_timer()

                h_hat = np.zeros(shape = (p + n, 1))
                f_best = float('inf')
                h_best = np.zeros(shape = (p + n, 1))
                errors = []
                
                for iteration in range(n_interations):
                    if iteration == 0:
                        h_hat = mu * np.transpose(W) @ y
                    else:
                        h_hat = h_hat - mu * np.transpose(W) @ (W[:, non_zero_support] @ h_hat[non_zero_support, :] - y)

                    h_hat = h_hat.reshape((-1))

                    x_non_zero_support = np.argpartition(np.abs(h_hat[:p]), -k)[-k: ]
                    x_zero_support = np.argpartition(np.abs(h_hat[:p]), p - k)[:(p - k)]

                    e_non_zero_support = np.argpartition(np.abs(h_hat[p:]), -s)[-s: ]
                    e_zero_support = np.argpartition(np.abs(h_hat[p:]), n - s)[:(n - s)]

                    e_non_zero_support += p
                    e_zero_support += p

                    non_zero_support = np.concatenate((x_non_zero_support, e_non_zero_support))            # maximum t
                    zero_support = np.concatenate((x_zero_support, e_zero_support))
                    non_zero_support = np.sort(non_zero_support)
                    zero_support = np.sort(zero_support)
                    
                    h_hat[non_zero_support] = (np.linalg.pinv(W[:, non_zero_support]) @ y).reshape((-1))
                    h_hat[zero_support] = 0
                    h_hat = h_hat.reshape((p + n, 1))

                    f_hat = np.linalg.norm((W[:, non_zero_support] @ h_hat[non_zero_support, :] - y).reshape((-1)), ord = 2)
                    if f_hat < f_best:
                        f_best = f_hat
                        h_best = np.copy(h_hat)
                    if f_hat <= (1e-4) * np.linalg.norm(y.reshape((-1)), ord = 2):
                        break

                    errors.append(f_hat)

                t_1 = timeit.default_timer()
                            
                crossValidationError = np.linalg.norm(A_cv @ h_best[:p, :] - y_cv)
                if crossValidationError < currLeastError:
                    currLeastError = crossValidationError
                    res = np.copy(h_best)
                    t = t_1 - t_0

    return res[:p, :], t


            

            



def one_norm(A, y, A_cv, y_cv, r1 = None):
    '''
    Implements One norm optimizer.
    A: (m + n, p) and y: (m + n, 1), where N = m + n
    '''

    N = A.shape[0]
    p = A.shape[1]
    
    currLeastError  = float('inf')
    candidateLambda1 = np.array([r1]) if r1 is not None else np.arange(0.001, 0.900, 0.020) 

    crossvalidationErrorList = np.zeros(shape = (candidateLambda1.shape[0]))

    for i, lambda1 in enumerate(candidateLambda1):

        x = cp.Variable(shape = (p, 1))
        objective = cp.Minimize(cp.norm1(y - A @ x) + lambda1*cp.norm1(x))    
        prob = cp.Problem(objective)
        result = prob.solve(solver = cp.ECOS, verbose = False)
        x = x.value
        x = x.reshape((p, 1))
            
        crossValidationError = np.linalg.norm(A_cv @ x - y_cv)
        crossvalidationErrorList[i] = crossValidationError
        if crossValidationError < currLeastError:
            currLeastError = crossValidationError
            xhat = x    
            usedlambda1 = lambda1
              
    
    if usedlambda1 <= 0.004 or usedlambda1 >= 0.870:
        print("Choose lambda1 properly")

    return xhat, usedlambda1

def only_priors(A, y, A_cv, y_cv, m, r1 = None):
    '''
    Implements One norm optimizer.
    A: (m + n, p) and y: (m + n, 1), where N = m + n
    '''

    N = A.shape[0]
    p = A.shape[1]

    A1 = np.copy(A[:m, :])
    y1 = np.copy(y[:m, :])
    
    currLeastError  = float('inf')
    candidateLambda1 = np.array([r1]) if r1 is not None else np.arange(0.001, 0.900, 0.020) 

    crossvalidationErrorList = np.zeros(shape = (candidateLambda1.shape[0]))

    for i, lambda1 in enumerate(candidateLambda1):

        x = cp.Variable(shape = (p, 1))
        objective = cp.Minimize(cp.norm1(y1 - A1 @ x) + lambda1*cp.norm1(x))    
        prob = cp.Problem(objective)
        result = prob.solve(solver = cp.ECOS, verbose = False)
        x = x.value
        x = x.reshape((p, 1))
            
        crossValidationError = np.linalg.norm(A_cv @ x - y_cv)
        crossvalidationErrorList[i] = crossValidationError
        if crossValidationError < currLeastError:
            currLeastError = crossValidationError
            xhat = x    
            usedlambda1 = lambda1
              
    
    if usedlambda1 <= 0.004 or usedlambda1 >= 0.870:
        print("Choose lambda1 properly")

    return xhat, usedlambda1













