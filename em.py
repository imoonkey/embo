import numpy as np
import numpy.matlib as npmat
from hmm import *

def em_restarts(hmm, obs_all, restarts, iters, epsilon):
    iters_log = np.zeros((restarts,))
    ll_log = np.zeros((restarts,))
    for i in range(restarts):
        z_mat_init, t_mat_init = random_hmm(hmm.num_states, hmm.num_obs)
        hmm_est, actual_iters = em(hmm, z_mat_init, t_mat_init, obs_all, iters, epsilon)
        iters_log[i] = actual_iters
        ll_log[i] = hmm_est.loglikelihood(obs_all[0])
    return ll_log, iters_log

def em(hmm, z_mat_init, t_mat_init, obs_all, iters, epsilon):
    prev_z_mat = z_mat_init
    prev_t_mat = t_mat_init
    hmm_est = HMM(prev_z_mat, prev_t_mat, hmm.pi_vec)
    prev_ll = hmm_est.loglikelihood(obs_all[0])
    actual_iters = iters
    for i in range(iters):
        next_z_mat, next_t_mat = em_step(hmm.num_states, hmm.num_obs, hmm.pi_vec, prev_z_mat, prev_t_mat, obs_all)
        hmm_est = HMM(next_z_mat, next_t_mat, hmm.pi_vec)
        next_ll = hmm_est.loglikelihood(obs_all[0])
        if False:
            print('EM estimates')
            print(hmm_est.z_mat)
            print(hmm_est.t_mat)
            print(next_ll)
        if (next_ll - prev_ll) <= epsilon:
            actual_iters = i+1
            break
        prev_z_mat, prev_t_mat, prev_ll = next_z_mat, next_t_mat, next_ll
    return hmm_est, actual_iters

def em_step(num_states, num_obs, pi_vec, z_mat, t_mat, obs_all):
    # given initial parameters, run em from the obs
    # obs are 2d with each row being a trajectory
    # returns estimated observation and transition matrices
    
    num_steps = obs_all.shape[1]
    t_mat = npmat.matrix(t_mat)    
    
    gamma_sum = np.zeros((num_states,))
    xi_sum = np.zeros((num_states,num_states))
    obs_sum = np.zeros((num_obs, num_states))
    
    for t in range(obs_all.shape[0]):
        obs = obs_all[t,:]
        # compute alpha
        alpha = npmat.zeros((num_states, num_steps))
        alpha[:,0] = npmat.matrix((pi_vec * z_mat[obs[0], :])[:,np.newaxis])
        for i in range(1,len(obs)):
            alpha[:,i] = npmat.matrix(((t_mat * alpha[:,i-1]).getA()[:,0] * z_mat[obs[i], :])[:,np.newaxis])
            asum = alpha[:,i].sum()
            alpha[:,i] /= asum
#        print('alpha\n' + str(alpha))
        # compute beta
        beta = npmat.zeros((num_states, num_steps))
        beta[:,num_steps-1] = npmat.ones((num_states, 1))
        for i in range(len(obs)-2,-1,-1):
            beta[:,i] = t_mat.getT() * npmat.matrix((z_mat[obs[i+1], :] * beta[:,i+1].getA()[:,0])[:,np.newaxis])
            bsum = beta[:,i].sum()
            beta[:,i] /= bsum
#        print(beta)
        # compute gamma
        gamma = alpha.getA() * beta.getA()
        gamma = gamma / (gamma.sum(axis=0)[np.newaxis,:])
#        print(gamma)
        # compute xi (transposed from the paper)
        xi = np.zeros((num_states, num_states, num_steps-1))
        for i in range(num_steps-1):
            xi[:,:,i] = np.transpose(alpha[:,i].getA()) * t_mat.getA() * (z_mat[obs[i+1],:] * beta[:,i+1].getA()[0])[:,np.newaxis]
            xsum = xi[:,:,i].sum()
            xi[:,:,i] /= xsum
#            print('xi[:,:,i]' + str(xi[:,:,i]))
#            print('np.transpose(alpha[:,i].getA()): ' + str(np.transpose(alpha[:,i].getA())))
#            print('t_mat: ' + str(t_mat))
#            print('result of *: ' + str(np.transpose(alpha[:,i].getA()) * t_mat.getA()))
        # add to the sums
        gamma_sum += gamma.sum(axis=1)
        xi_sum += xi.sum(axis=2)
        for z in range(num_obs):
            obs_sum[z,:] += gamma[:,obs == z].sum(axis=1)
#        print('gamma_sum\n' + str(gamma_sum))
#        print('xi_sum\n' + str(xi_sum))
#        print('obs_sum\n' + str(obs_sum))
    # finally compute estimates
    est_z_mat = obs_sum / obs_sum.sum(axis=0)[np.newaxis,:]
    est_t_mat = xi_sum / xi_sum.sum(axis=0)[np.newaxis,:]
    
    return est_z_mat, est_t_mat;


def test():
    t_mat = np.array([[0.9, 0.1],[0.1, 0.9]])
    z_mat = np.array([[0.9, 0.1],[0.1, 0.9]])
    pi_vec = np.array([0.5, 0.5])
    
    z_mat_p = np.array([[np.log(9), np.log(1.0/9)]])
    t_mat_p = np.array([[np.log(9), np.log(1.0/9)]])
    
    hmm1 = HMM(z_mat, t_mat, pi_vec)
    hmm2 = make_parameterized_HMM(z_mat_p, t_mat_p, pi_vec)
    hmm3 = HMM(np.array([[0.8, 0.2],[0.1, 0.9]]), np.array([[0.9, 0.1],[0.1, 0.9]]), pi_vec)
    
    np.random.seed(0x6b6c26b2)
    obs1 = hmm1.generate(1000)
#    print(obs1)
#    print(obs2)
    
    ll_log, iters_log = em_restarts(hmm1, obs1[np.newaxis,:], 10, 50, 0.1)
    print('ll_log: ' + str(ll_log))
    print('max: ' + str(np.max(ll_log)))
    print('iters_log: ' + str(iters_log))
    print('real param ll: ' + str(hmm1.loglikelihood(obs1)))
        
    
if __name__ == '__main__':
    test()
    pass