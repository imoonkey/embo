import numpy as np
import numpy.matlib as npmat


# Standard way to specify an HMM
# an observation matrix is z(z,s) = p(z | s)
# a transition matrix is t(s',s) = p(s' | s)
# initial state is pi(s0) = p(s0)
# we assume initial state distribution is known so we don't learn them

# Alternative parameterized way for an HMM
# matrices are the same as the standard, but with one less row
# i.e. for two states, only t(0,0) and t(0,1) are given since the other is implicity assumed to be 0
# the way to compute is that p(0 | 0) = exp(t(0,0)) / (exp(t(0,0)) + exp(0))
# and p(1 | 0) = exp(0) / (exp(t(0,0)) + exp(0))
# this is easily extended to more parameters

class HMM:
    def __init__(self, z_mat, t_mat, pi_vec):
        self.z_mat = z_mat
        self.t_mat = t_mat
        self.pi_vec = pi_vec
        
        self.num_states = t_mat.shape[0]
        self.num_obs = z_mat.shape[0]
    
    def generate(self, length):
        # randomly generate a trajectory of length
        rands = np.random.rand(length)
        obs = np.zeros((length,))
        curr_state = sample_discrete(self.pi_vec)
        for i in range(length):
            obs[i] = sample_discrete(self.z_mat[:,curr_state])
            curr_state = sample_discrete(self.t_mat[:,curr_state])
        return obs
    
    def loglikelihood(self, obs):
        # compute the log likelihood given a sequence of obs
        t_mat = npmat.matrix(self.t_mat)
        alpha = npmat.matrix((self.pi_vec * self.z_mat[obs[0], :])[:,np.newaxis])
        asum = alpha.sum()
        alpha /= asum
        log_coef = np.log(asum)
        for i in range(1,len(obs)):
            alpha = npmat.matrix(((t_mat * alpha).getA()[0] * self.z_mat[obs[i], :])[:,np.newaxis])
            asum = alpha.sum()
            alpha /= asum
            log_coef += np.log(asum)
        return log_coef + np.log(alpha.sum())
        

def sample_discrete(pvals):
    # sample from a discrete distribution
    r = np.random.rand()
    acc = 0
    for i,p in enumerate(pvals):
        acc += p
        if acc >= r:
            return i
    else:
        return len(pvals)-1

def make_parameterized_HMM(z_mat_p, t_mat_p, pi_vec):
    num_states = t_mat_p.shape[1]
    z_mat = np.exp(np.vstack((z_mat_p, np.zeros((num_states,)))))
    t_mat = np.exp(np.vstack((t_mat_p, np.zeros((num_states,)))))
    
    z_mat = z_mat / z_mat.sum(axis=0)[np.newaxis,:]
    t_mat = t_mat / t_mat.sum(axis=0)[np.newaxis,:]
    
    return HMM(z_mat, t_mat, pi_vec)

def test():
    t_mat = np.array([[0.9, 0.1],[0.1, 0.9]])
    z_mat = np.array([[0.9, 0.1],[0.1, 0.9]])
    pi_vec = np.array([0.5, 0.5])
    
    z_mat_p = np.array([[np.log(9), np.log(1.0/9)]])
    t_mat_p = np.array([[np.log(9), np.log(1.0/9)]])
    
    hmm1 = HMM(z_mat, t_mat, pi_vec)
    hmm2 = make_parameterized_HMM(z_mat_p, t_mat_p, pi_vec)
    hmm3 = HMM(np.array([[0.8, 0.2],[0.1, 0.9]]), np.array([[0.9, 0.1],[0.1, 0.9]]), pi_vec)
    
    print(hmm1.z_mat)
    print(hmm1.t_mat)
    print(hmm2.z_mat)
    print(hmm2.t_mat)
    
    np.random.seed(0x6b6c26b2)
    obs1 = hmm1.generate(20)
    print(obs1)
    np.random.seed(0x6b6c26b2)
    obs2 = hmm2.generate(20)
    print(obs2)

    print(hmm1.loglikelihood(obs1))
    print(hmm2.loglikelihood(obs1))
    print(hmm3.loglikelihood(obs1))
    
if __name__ == '__main__':
    test()
    pass