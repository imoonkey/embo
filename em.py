import numpy as np
import numpy.matlib as npmat
from hmm import *

def em(hmm, obs_all):
    # given initial parameters, run em from the obs
    # obs are 2d with each row being a trajectory
    # returns estimated observation and transition matrices
    
    num_steps = obs_all.shape[1]
    t_mat = npmat.matrix(hmm.t_mat)    
    
    gamma_sum = np.zeros((hmm.num_states,))
    xi_sum = np.zeros((hmm.num_states,hmm.num_states))
    obs_sum = np.zeros((hmm.num_obs, hmm.num_states))
    
    for t in range(obs_all.shape[0]):
        obs = obs_all[t,:]
        # compute alpha
        alpha = npmat.zeros((hmm.num_states, num_steps))
        alpha[:,0] = npmat.matrix((hmm.pi_vec * hmm.z_mat[obs[0], :])[:,np.newaxis])
        asum = alpha[:,0].sum()
        alpha[:,0] /= asum
        for i in range(1,len(obs)):
            alpha[:,i] = npmat.matrix(((t_mat * alpha[:,i-1]).getA()[0] * hmm.z_mat[obs[i], :])[:,np.newaxis])
            asum = alpha[:,i].sum()
            alpha[:,i] /= asum
#        print(alpha)
        # compute beta
        beta = npmat.zeros((hmm.num_states, num_steps))
        beta[:,num_steps-1] = npmat.ones((hmm.num_states, 1))
        for i in range(len(obs)-2,-1,-1):
            beta[:,i] = t_mat.getT() * npmat.matrix((hmm.z_mat[obs[i+1], :] * beta[:,i+1].getA()[0])[:,np.newaxis])
            bsum = beta[:,i].sum()
            beta[:,i] /= bsum
#        print(beta)
        # compute gamma
        gamma = alpha.getA() * beta.getA()
        gamma = gamma / (gamma.sum(axis=0)[np.newaxis,:])
#        print(gamma)
        # compute xi (transposed from the paper)
        xi = np.zeros((hmm.num_states, hmm.num_states, num_steps-1))
        for i in range(num_steps-1):
            xi[:,:,i] = np.transpose(alpha[:,i].getA()) * hmm.t_mat * (hmm.z_mat[obs[i+1],:] * beta[:,i+1].getA()[0])[:,np.newaxis]
            xsum = xi[:,:,i].sum()
            xi[:,:,i] /= xsum
#            print(xi[:,:,i])
        # add to the sums
        gamma_sum += gamma.sum(axis=1)
        xi_sum += xi.sum(axis=2)
        for z in range(hmm.num_obs):
            obs_sum[z,:] += gamma[:,obs == z].sum(axis=1)
        print(gamma_sum)
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
    obs1 = hmm1.generate(50)
    obs2 = hmm1.generate(50)
    obs3 = hmm1.generate(50)
#    print(obs1)
#    print(obs2)
    
    
    est_z_mat, est_t_mat = em(hmm1, np.vstack((obs1, obs2, obs3)))
    print(est_z_mat)
    print(est_t_mat)
    
    hmm_est = HMM(est_z_mat, est_t_mat, pi_vec)
    print(hmm_est.loglikelihood(obs3))
    print(hmm1.loglikelihood(obs3))
    
if __name__ == '__main__':
    test()
    pass