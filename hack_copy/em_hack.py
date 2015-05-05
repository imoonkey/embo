__author__ = 'moonkey'

import os

try:
    import simplejson as json
except ImportError:
    import json

from spearmint.utils.database.mongodb import MongoDB
import time
import numpy as np


def load_new_history(work_dir):
    file_path = os.path.join(work_dir, "historical_params.json")
    try:
        with open(file_path) as param_file:
            historical_points = json.load(param_file)
        return historical_points['history']
    except Exception:
        return {}


def dump_new_history(work_dir, historical_points):
    file_path = os.path.join(work_dir, "historical_params.json")
    with open(file_path) as param_file:
        param_file.write(json.dump({'history': historical_points}))
    print 'New history dumped.'


# TODO::json format, the format of 'params' is the output of task_group.paramify()
# j = {'history': [
#     {
#         'params': {
#             't_mat_p_0': {'values': np.array([40.48156738]), 'type': 'float'},
#             't_mat_p_1': {'values': np.array([46.53015137]), 'type': 'float'},
#             'z_mat_p_0': {'values': np.array([4.23278809]), 'type': 'float'},
#             'z_mat_p_1': {'values': np.array([-4.36096191]), 'type': 'float'}
#         },
#         'value': {111.1}
#     }
# ]}


def add_historical_points_to_db(db, experiment_name, expt_dir=""):
    jobs = _load_jobs(db, experiment_name)
    historical_points = load_new_history(expt_dir)

    for idx in range(0, len(historical_points)):
        job_id = idx + len(jobs) + 1
        job = {
            'id': job_id,
            'params': _paramify(historical_points[idx]['params']),
            'expt_dir': expt_dir,
            'tasks': ['main'],
            'resource': "Main",
            'main-file': "historical_params.json",
            'language': 'PYTHON',
            'status': 'complete',
            'submit time': time.time(),
            'start time': None,
            'end time': None,
            "values": {"main": historical_points[idx]['value']},
        }
        _save_job(job, db, experiment_name)


def _paramify(a):
    pass


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
