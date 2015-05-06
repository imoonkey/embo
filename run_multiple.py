__author__ = 'moonkey'
from pymongo import Connection
import os
import subprocess
import json

# ###################################
##########   Setting    ############
####################################

exp_config_boem = {
    'hmm_num': 2,
    'base_experiment_name': "bo-em-hmm",
    'exp_folder': "bo_em",
}
exp_config_simple_bo = {
    'hmm_num': 2,
    'base_experiment_name': "simple-bo-hmm",
    'exp_folder': "bo_spearmint",
}


def run_exp(exp_config):
    mongo = Connection()

    for hmm_idx in range(0, exp_config['hmm_num']):
        experiment_name = exp_config['base_experiment_name'] + "-" + str(hmm_idx)

        # clean the database, uncomment this if the program was broken accidentally somewhere
        mongo['spearmint'].drop_collection(experiment_name + '.jobs')
        mongo['spearmint'].drop_collection(experiment_name + '.hypers')

        spearmint_config = open(os.path.join(os.path.abspath(exp_config['exp_folder']), 'config.json'))
        spearmint_config_json = json.load(spearmint_config)
        spearmint_config_json['experiment-name'] = experiment_name
        spearmint_config = open(os.path.join(os.path.abspath(exp_config['exp_folder']), 'config.json'), 'w')
        json.dump(spearmint_config_json, spearmint_config)
        spearmint_config.close()

        extra_history = open(os.path.join(exp_config['exp_folder'], 'extra_history.json'), 'w')
        extra_history.write('{"history": []}')
        extra_history.close()

        hmm_idx_file = open(os.path.join(exp_config['exp_folder'], 'hmm_index.txt'), 'w')
        hmm_idx_file.write(str(hmm_idx))
        hmm_idx_file.close()

        # start the experiment for this HMM

        subprocess.call('./run_' + exp_config['base_experiment_name']+ '.sh', shell=True)


if __name__ == "__main__":
    # run_exp(exp_config_simple_bo)
    run_exp(exp_config_boem)