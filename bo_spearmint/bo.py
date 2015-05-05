import os
import sys
import inspect

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)

import hmm
import numpy as np
import json

# Write a function like this called 'main'
def main(job_id, params):
    # print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print 'Params:' + str(params)

    # generate HMM observations
    z_mat_p = np.array([[np.log(9), np.log(1.0 / 9)]])
    t_mat_p = np.array([[np.log(9), np.log(1.0 / 9)]])
    pi_vec = np.array([0.5, 0.5])
    hmm_groundtruth = hmm.make_parameterized_HMM(z_mat_p, t_mat_p, pi_vec)
    np.random.seed(0x6b6c26b2)
    obs = hmm_groundtruth.generate(100)
    # print "gt0:" +str(-hmm_groundtruth.loglikelihood(obs))

    # calculate log likelihood for input HMM parameters
    z_mat_p_input = np.array([[params['z_mat_p_0'][0], params['z_mat_p_1'][0]]])
    t_mat_p_input = np.array([[params['t_mat_p_0'][0], params['t_mat_p_1'][0]]])
    # pi_vec_input = np.array([params['pi_0'], 1 - params['pi_0']])
    hmm_estimate = hmm.make_parameterized_HMM(z_mat_p_input, t_mat_p_input, pi_vec)
    hmm_loglikelihood = hmm_estimate.loglikelihood(obs)
    # print "gt:" +str(-hmm_groundtruth.loglikelihood(obs))
    # print hmm_groundtruth.z_mat
    # print hmm_groundtruth.t_mat
    # print obs
    return -hmm_loglikelihood


if __name__ == '__main__':
    parameters = {
        'z_mat_p_0': [-16],
        'z_mat_p_1': [7],
        't_mat_p_0': [50],
        't_mat_p_1': [50],
    }
    print main(1,parameters)