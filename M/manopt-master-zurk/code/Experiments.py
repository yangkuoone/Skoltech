import os
import sys
import numpy as np
import pandas as pd
from log_progress import log_progress
from algorithms.benchmarks import LancichinettiBenchmark, LB_set_seed
from algorithms.utils import time
from algorithms.NMI import NMI


def run_methods_model_data(methods, data_params, data_name, M=20, mixing_range=None, seed=1231231, ):
    filepath = '../data/model_data_{}.csv'
    if os.path.exists(filepath.format(data_name)):
        experiments_result_pd = pd.DataFrame.from_csv(filepath.format(data_name)).T
        experiments_result = experiments_result_pd.to_dict()
    else:
        experiments_result = {}
    mixing_range = np.linspace(0, 0.5, 6) if mixing_range is None else mixing_range
    # mixing_range = np.linspace(0, 0.5, 3)

    LB_set_seed(seed)
    for bb in log_progress(range(M), every=1):
        G, real_comms = LancichinettiBenchmark(**data_params)
        real_comms = real_comms.values()
        nodes = list(G.nodes())
        inv_nodes = {v: i for i, v in enumerate(nodes)}
        real_comms_res = [[inv_nodes[i] for i in com if i in inv_nodes] for com in real_comms]
        for mix in mixing_range:
            name = '{}_mix-{}_{}'.format(data_name, mix, bb)
            if name not in experiments_result:
                print '{} name: {}'.format(time(), name)
                data_params['on'] = np.floor(data_params['N'] * mix)

                experiments_result[name] = {}
                experiments_result[name]['mix'] = mix
                experiments_result[name]['N'] = data_params['N']
                experiments_result[name]['mut'] = data_params['mut']

                for method_name in methods:
                    print method_name,
                    sys.stdout.flush()
                    try:
                        method_comms = methods[method_name](G, len(real_comms))
                    except Exception as e:
                        print('Some error occurred during {}: {} set quality to -1'.format(method_name, e))
                        experiments_result[name][method_name] = -1
                    try:
                        experiments_result[name][method_name] = NMI(real_comms_res, method_comms)
                    except Exception as e:
                        print('Some error occurred during NMI: {} set quality to -1'.format(e))
                        experiments_result[name][method_name] = -1
                print '\r', ' ' * 120, '\r',
                experiments_result_pd = pd.DataFrame.from_dict(experiments_result).T
                experiments_result_pd.to_csv(filepath.format(data_name))
            else:
                print('{} already exist. skip'.format(name))
    return experiments_result_pd
