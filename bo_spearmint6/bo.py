import os
import sys
import inspect

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)

from hmm import HMM, random_hmm, make_parameterized_HMM
import numpy as np
import json


# Write a function like this called 'main'
def main(job_id, params):
    num_runs = 20
    obs_length = 100
    num_states = 2
    num_obs = 2

    # readin hmm indx
    t = 0
    try:
        with open(os.path.join('.', 'hmm_index.txt')) as hmm_index_file:
            t = int(hmm_index_file.read())
        sys.stderr.write("!!!!!!!!!!!!!!!!!!HMM INDEX:  " + str(t) + "   !!!!!!!!!!!!!!!\n")
    except IOError:
        t = 0


    # generate HMM observations
    np.random.seed(0x6b6c26b2)
    seeds = np.random.randint(0x0fffffff, size=num_runs)
    np.random.seed(seeds[t])
    # random hmm
    z_mat, t_mat = random_hmm(num_states, num_obs)
    pi_vec = np.array([1.0 / num_states] * num_states)
    hmm_test = HMM(z_mat, t_mat, pi_vec)
    # random obs trajectory
    obs = hmm_test.generate(obs_length)[np.newaxis,:]

    # calculate log likelihood for input HMM parameters
    z_mat_p_input = np.array([[params['z_mat_p_0'][0]*6, params['z_mat_p_1'][0]*6]])
    t_mat_p_input = np.array([[params['t_mat_p_0'][0]*6, params['t_mat_p_1'][0]*6]])
    # pi_vec_input = np.array([params['pi_0'], 1 - params['pi_0']])
    hmm_estimate = make_parameterized_HMM(z_mat_p_input, t_mat_p_input, pi_vec)
    hmm_loglikelihood = hmm_estimate.loglikelihood(obs[0])

    return -hmm_loglikelihood


if __name__ == '__main__':
    parameters = {
        'z_mat_p_0': [-16],
        'z_mat_p_1': [7],
        't_mat_p_0': [50],
        't_mat_p_1': [50],
    }
    print main(1, parameters)