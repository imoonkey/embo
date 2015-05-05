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
        # use alpha
        alpha = npmat.matrix((self.pi_vec * self.z_mat[obs[0], :])[:,np.newaxis])
#        print('self.z_mat[obs[0], :]: '+ str(self.z_mat[obs[0], :]))
#        print('self.pi_vec: '+ str(self.pi_vec))
#        print('self.pi_vec * self.z_mat[obs[0], :]: '+ str(self.pi_vec * self.z_mat[obs[0], :]))
#        print('alpha: '+ str(alpha))
        log_coef = 0
        for i in range(1,len(obs)):
            alpha = npmat.matrix(((t_mat * alpha).getA()[:,0] * self.z_mat[obs[i], :])[:,np.newaxis])
            asum = alpha.sum()
            alpha /= asum
            log_coef += np.log(asum)
        alpha_ll = log_coef + np.log(alpha.sum())
        # use beta
        if False:
            beta = npmat.ones((t_mat.shape[0], 1))
            log_coef = 0
            for i in range(len(obs)-2,-1,-1):
    #            print('self.z_mat[obs[i+1], :]: ' + str(self.z_mat[obs[i+1], :]))
    #            print('beta.getA()[0]: ' + str(beta.getA()[:,0]))
    #            print('self.z_mat[obs[i+1], :] * beta.getA()[:,0]: ' + str(self.z_mat[obs[i+1], :] * beta.getA()[:,0]))
    #            print('npmat.matrix(result): ' + str(npmat.matrix((self.z_mat[obs[i+1], :] * beta.getA()[:,0])[:,np.newaxis])))
    #            print('t_mat.getT(): ' + str(t_mat.getT()))
                beta = t_mat.getT() * npmat.matrix((self.z_mat[obs[i+1], :] * beta.getA()[:,0])[:,np.newaxis])
                bsum = beta.sum()
                beta /= bsum
                log_coef += np.log(bsum)
            beta_ll = log_coef + np.log(np.sum(beta.getA()[0] * self.pi_vec * self.z_mat[obs[0],:]))
    #        print('alpha_ll: ' + str(alpha_ll))
    #        print('beta_ll: ' + str(beta_ll))
        return alpha_ll
        

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

def random_hmm(num_states, num_obs):
    # returns a random transition and observation matrix
    z_mat = np.random.rand(num_obs, num_states)
    z_mat += 0.0000001
    z_mat = -np.log(z_mat)
    z_mat /= z_mat.sum(axis=0)[np.newaxis,:]
    
    t_mat = np.random.rand(num_states, num_states)
    t_mat += 0.0000001
    t_mat = -np.log(t_mat)
    t_mat /= t_mat.sum(axis=0)[np.newaxis,:]
    
    return z_mat, t_mat

def make_parameterized_HMM(z_mat_p, t_mat_p, pi_vec):
    num_states = t_mat_p.shape[1]
    # print z_mat_p.shape
    # print num_states
    z_mat = np.exp(np.vstack((z_mat_p, np.zeros((num_states,)))))
    t_mat = np.exp(np.vstack((t_mat_p, np.zeros((num_states,)))))
    
    z_mat = z_mat / z_mat.sum(axis=0)[np.newaxis,:]
    t_mat = t_mat / t_mat.sum(axis=0)[np.newaxis,:]
    
    return HMM(z_mat, t_mat, pi_vec)

def retrieve_parameterized_HMM(hmm):
    # returns a parameterized version of the given hmm
    z_mat = np.log(hmm.z_mat)
    z_mat -= z_mat[-1,:][np.newaxis,:]
    z_mat = z_mat[:-1,:]
    
    t_mat = np.log(hmm.z_mat)
    t_mat -= t_mat[-1,:][np.newaxis,:]
    t_mat = t_mat[:-1,:]
    
    return z_mat, t_mat

def test():
    t_mat = np.array([[0.9, 0.1],[0.1, 0.9]])
    z_mat = np.array([[0.9, 0.1],[0.1, 0.9]])
    pi_vec = np.array([0.5, 0.5])
    
    z_mat_p = np.array([[np.log(9), np.log(1.0/9)]])
    t_mat_p = np.array([[np.log(9), np.log(1.0/9)]])
    
    hmm1 = HMM(z_mat, t_mat, pi_vec)
    hmm2 = make_parameterized_HMM(z_mat_p, t_mat_p, pi_vec)
    hmm3 = HMM(np.array([[0.8, 0.2],[0.1, 0.9]]), np.array([[0.9, 0.1],[0.1, 0.9]]), pi_vec)
    conv_z_mat, conv_t_mat = retrieve_parameterized_HMM(hmm1)
    hmm4 = make_parameterized_HMM(conv_z_mat, conv_t_mat, pi_vec)
    
    print(hmm1.z_mat)
    print(hmm1.t_mat)
    print(hmm2.z_mat)
    print(hmm2.t_mat)
    print(hmm4.z_mat)
    print(hmm4.t_mat)
    
    np.random.seed(0x6b6c26b2)
    obs1 = hmm1.generate(00)
    print(obs1)
#    np.random.seed(0x6b6c26b2)
#    obs2 = hmm2.generate(1000)
#    print(obs2)

    print(hmm1.loglikelihood(obs1))
    print(hmm2.loglikelihood(obs1))
    print(hmm3.loglikelihood(obs1))
    
if __name__ == '__main__':
    test()
    pass