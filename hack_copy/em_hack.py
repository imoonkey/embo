__author__ = 'moonkey'

import os

try:
    import simplejson as json
except ImportError:
    import json

from spearmint.utils.database.mongodb import MongoDB
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


def add_historical_points_to_db(db, experiment_name, expt_dir=""):
    jobs = _load_jobs(db, experiment_name)
    historical_points = load_new_history(expt_dir)

    for idx in range(0, len(historical_points)):
        job_id = idx + len(jobs) + 1
        job = {
            'id': job_id,
            'params': historical_points[idx]['params'],
            'expt_zdir': expt_dir,
            'tasks': ['main'],
            'resource': "Main",
            'main-file': "extra_history.json",
            'language': 'PYTHON',
            'status': 'complete',
            'submit time': time.time(),
            'start time': time.time(),
            'end time': time.time(),
            "values": {"main": historical_points[idx]['value']},
        }
        _save_job(job, db, experiment_name)


def _load_jobs(db, experiment_name):
    """load the jobs from the database

    Returns
    -------
    jobs : list
        a list of jobs or an empty list
    """
    jobs = db.load(experiment_name, 'jobs')

    if jobs is None:
        jobs = []
    if isinstance(jobs, dict):
        jobs = [jobs]

    return jobs


def _save_job(job, db, experiment_name):
    """save a job to the database"""
    db.save(job, experiment_name, 'jobs', {'id': job['id']})


def _test_json():
    j = [
        {
            'params': {
                't_mat_p_0': {'values': np.array([-50.]), 'type': 'float'},
                't_mat_p_1': {'values': np.array([-50.]), 'type': 'float'},
                'z_mat_p_0': {'values': np.array([-50.]), 'type': 'float'},
                'z_mat_p_1': {'values': np.array([-50.]), 'type': 'float'}
            },
            'value': 1786.8302035693616
        },
        {
            'params': {
                't_mat_p_0': {'values': np.array([0.]), 'type': 'float'},
                't_mat_p_1': {'values': np.array([0.]), 'type': 'float'},
                'z_mat_p_0': {'values': np.array([0.]), 'type': 'float'},
                'z_mat_p_1': {'values': np.array([0.]), 'type': 'float'}
            },
            'value': 13.862943611198906
        }
    ]
    dump_new_history('', j)
    k = load_new_history('')
    print j == k


def _test_db():
    db = MongoDB(database_address='localhost')
    add_historical_points_to_db(db, 'simple-bo-hmm', '')


if __name__ == "__main__":
    # _test_json()
    _test_db()