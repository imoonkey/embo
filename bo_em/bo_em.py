import os
import sys
import inspect

try:
    import simplejson as json
except ImportError:
    import json

import time
import numpy as np
import base64

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        if input object is a ndarray it will be converted into a dict holding dtype, shape and the data base64 encoded
        """
        if isinstance(obj, np.ndarray):
            data_b64 = base64.b64encode(obj.data)
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(self, obj)


def json_numpy_obj_hook(dct):
    """
    Decodes a previously encoded numpy ndarray
    with proper shape and dtype
    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct


# expected = np.arange(100, dtype=np.float)
# dumped = json.dumps(expected, cls=NumpyEncoder)
# result = json.loads(dumped, object_hook=json_numpy_obj_hook)


def load_new_history(work_dir):
    file_path = os.path.join(work_dir, "extra_history.json")
    try:
        with open(file_path) as param_file:
            historical_points = json.load(param_file, object_hook=json_numpy_obj_hook)
        return historical_points['history']
    except Exception:
        return {}


def dump_new_history(work_dir, historical_points):
    file_path = os.path.join(work_dir, "extra_history.json")
    with open(file_path, 'w') as param_file:
        json.dump({'history': historical_points}, param_file, cls=NumpyEncoder)
    print 'New history dumped.'


# TODO::json format, the format of 'params' is the output of task_group.paramify()
# j = {'history': [
# {
# 'params': {
# 't_mat_p_0': {'values': np.array([40.48156738]), 'type': 'float'},
# 't_mat_p_1': {'values': np.array([46.53015137]), 'type': 'float'},
# 'z_mat_p_0': {'values': np.array([4.23278809]), 'type': 'float'},
# 'z_mat_p_1': {'values': np.array([-4.36096191]), 'type': 'float'}
# },
# 'value': 111.1
# }
# ]}

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import hmm
import numpy as np
import em

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

    # calculate log likelihood for input HMM parameters
    z_mat_p_input = np.array([[params['z_mat_p_0'][0], params['z_mat_p_1'][0]]])
    t_mat_p_input = np.array([[params['t_mat_p_0'][0], params['t_mat_p_1'][0]]])
    # pi_vec_input = np.array([params['pi_0'], 1 - params['pi_0']])
    hmm_estimate = hmm.make_parameterized_HMM(z_mat_p_input, t_mat_p_input, pi_vec)
    hmm_loglikelihood = hmm_estimate.loglikelihood(obs)
    
    # use the current suggest point and run EM to get a new point
    hmm_em_est, actual_iters = em.em(hmm_estimate, hmm_estimate.z_mat, hmm_estimate.t_mat, obs[np.newaxis,:], 30, 0.1)
    em_est_z_mat, em_est_t_mat = hmm.retrieve_parameterized_HMM(hmm_em_est)
    em_est_ll = -hmm_em_est.loglikelihood(obs)
    em_est_z_mat.reshape((em_est_z_mat.size,))
    em_est_t_mat.reshape((em_est_t_mat.size,))
    print em_est_t_mat
    print em_est_z_mat
    historical_points = [{'params' : {}}]
    # write z_mat
    for i,v in enumerate(em_est_z_mat[0]):
        historical_points[0]['params']['z_mat_p_' + str(i)] = {'values': np.array([v]), 'type': 'float'}
    # write t_mat
    for i,v in enumerate(em_est_t_mat[0]):
        historical_points[0]['params']['t_mat_p_' + str(i)] = {'values': np.array([v]), 'type': 'float'}
    historical_points[0]['value'] = em_est_ll
    dump_new_history('.', historical_points)
    return -hmm_loglikelihood


if __name__ == '__main__':
    parameters = {
        'z_mat_p_0': [9],
        'z_mat_p_1': [1.0/9],
        't_mat_p_0': [9],
        't_mat_p_1': [1.0/9],
    }
    print main(1,parameters)