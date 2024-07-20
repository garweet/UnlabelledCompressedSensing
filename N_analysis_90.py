from utils import RL_with_constraint, RL_without_constraint, RLWKC_with_constraint, RLWKC_without_constraint, HTP, HTP_with_priors, sparsePermutation, addNoise, meanNormalizedError, sgn, one_norm, only_priors
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from IPython.display import clear_output
import random  
import scipy
np.random.seed(90)
random.seed(90)

n_Permuted = 16                                          # number of permutation
cv_measurements = 5
m = 32                                                    # number of known correspondences
noise_fraction = 0.02
permutation_runs, noise_runs = 50, 50
p = 240                                                   # dimension of x
k = 14                                                    # sparsity of x
# N_Metrics = np.zeros(shape = (permutation_runs, noise_runs, 9))
x = np.load("x_sparsity_" + str(k) + ".npy")
A_cv =  np.load("A_cv_fixed.npy")  
y_cv = addNoise(A_cv @ x, noise_fraction)     


for N in [90]:
    N_Metrics = np.load('N_Metrics' + str(N) + '.npy')
    A = np.load("A_fixed.npy")[:N, :]           # extract N rows from the generated matrix

    for per_run in range(permutation_runs):
        print("N: {}, Permutation run: {}".format(N, per_run))
        P = sparsePermutation(N, m, n_Permuted)
        noiseLess_y = P @ A @ x
        z_true = noiseLess_y - A @ x
        max_abs_z = np.max(np.abs(z_true))
        sum_abs_z = np.sum(np.abs(z_true))

        l1_RLWKC_with_constraint, l2_RLWKC_with_constraint = None, None
        l1_RLWKC_without_constraint, l2_RLWKC_without_constraint = None, None
        l1_RL_with_constraint, l2_RL_with_constraint = None, None
        l1_RL_without_constraint, l2_RL_without_constraint = None, None
        l1_one_norm = None
        l1_only_priors = None

        for noise_run in range(noise_runs):

            print("Permutation run: {}, Noisy run: {}".format(per_run, noise_run))
            noisy_y = addNoise(noiseLess_y, noise_fraction)

            # xhat_RLWKC_with_constraint, l1_RLWKC_with_constraint, l2_RLWKC_with_constraint = RLWKC_with_constraint(A, noisy_y, A_cv, y_cv, m, l1_RLWKC_with_constraint, l2_RLWKC_with_constraint)
            # xhat_RLWKC_without_constraint, l1_RLWKC_without_constraint, l2_RLWKC_without_constraint = RLWKC_without_constraint(A, noisy_y, A_cv, y_cv, m, l1_RLWKC_without_constraint, l2_RLWKC_without_constraint)
            # xhat_RL_with_constraint, l1_RL_with_constraint, l2_RL_with_constraint = RL_with_constraint(A, noisy_y, A_cv, y_cv, l1_RL_with_constraint, l2_RL_with_constraint)
            # xhat_RL_without_constraint, l1_RL_without_constraint, l2_RL_without_constraint = RL_without_constraint(A, noisy_y, A_cv, y_cv, l1_RL_without_constraint, l2_RL_without_constraint)
            xhat_HTP, _, l1_htp, _ = HTP(A, noisy_y, A_cv, y_cv)
            # xhat_HTP_with_priors = HTP_with_priors(A, noisy_y, A_cv, y_cv, m)
            # xhat_HTP_with_priors, htp_with_priors_used_k, htp_with_priors_used_mu = HTP_with_priors(A, noisy_y, A_cv, y_cv, m, htp_with_priors_used_k, htp_with_priors_used_mu)
            # xhat_one_norm, l1_one_norm = one_norm(A, noisy_y, A_cv, y_cv, l1_one_norm)
            # xhat_only_priors, l1_only_priors = only_priors(A, noisy_y, A_cv, y_cv, m, l1_only_priors)
  
            # N_Metrics[per_run][noise_run][0] = meanNormalizedError(x, xhat_RLWKC_with_constraint)
            # N_Metrics[per_run][noise_run][1] = meanNormalizedError(x, xhat_RLWKC_without_constraint)
            # N_Metrics[per_run][noise_run][2] = meanNormalizedError(x, xhat_RL_with_constraint)
            # N_Metrics[per_run][noise_run][3] = meanNormalizedError(x, xhat_RL_without_constraint)
            N_Metrics[per_run][noise_run][4] = meanNormalizedError(x, xhat_HTP)
            # N_Metrics[per_run][noise_run][5] = meanNormalizedError(x, xhat_one_norm)
            # N_Metrics[per_run][noise_run][6] = meanNormalizedError(x, xhat_only_priors)
            # N_Metrics[per_run][noise_run][7] = meanNormalizedError(x, xhat_HTP_with_priors)
            # N_Metrics[per_run][noise_run][8] = sum_abs_z

            print(N_Metrics[per_run][noise_run][0], N_Metrics[per_run][noise_run][1], N_Metrics[per_run][noise_run][2], N_Metrics[per_run][noise_run][3], N_Metrics[per_run][noise_run][4], N_Metrics[per_run][noise_run][5], N_Metrics[per_run][noise_run][6], N_Metrics[per_run][noise_run][7], N_Metrics[per_run][noise_run][8])

            np.save('N_Metrics' + str(N) + '.npy', N_Metrics)