from __future__ import division
import networkx as nx
import subprocess
import os
import platform
joinpath = os.path.join


data_dir = '../data/'
facebook_data_dir = joinpath(data_dir, 'facebook/')
twitter_data_dir = joinpath(data_dir, 'twitter/')
gplus_data_dir = joinpath(data_dir, 'gplus/')

facebook_graphs = [joinpath(facebook_data_dir, filename.replace(".edges", "")) for filename in os.listdir(facebook_data_dir) if filename.endswith(".edges")]
twitter_graphs = [joinpath(twitter_data_dir, filename.replace(".edges", "")) for filename in os.listdir(twitter_data_dir) if filename.endswith(".edges")]
gplus_graphs = [joinpath(gplus_data_dir, filename.replace(".edges", "")) for filename in os.listdir(gplus_data_dir) if filename.endswith(".edges")]
polblogs_graph = joinpath(data_dir, 'polblogs', 'polblogs.gml')



def get_realdatapaths():
    res = {}
    data_dict = {'facebook': facebook_graphs,
                 'twitter': twitter_graphs,
                 'gplus': gplus_graphs,
                 # 'polblogs': polblogs_graph
                 }
    for key in data_dict:
        if isinstance(data_dict[key], list):
            for indx, data_path in enumerate(data_dict[key]):
                res[key + str(indx)] = data_path
        else:
            res[key] = data_dict[key]

    return res


def load_real_data(datapath):
    if datapath.endswith(".gml"):
        G = nx.read_gml(datapath)
    else:
        edges_list = get_edges_list(datapath)
        comms = get_circles_list(datapath)
        G = nx.Graph(edges_list)

    return G, comms

def LB_set_seed(seed):
    with file(r'..\external\Lancichinetti_benchmark\benchmark\time_seed.dat', 'w') as f:
        f.write(str(seed))

def LancichinettiBenchmark(**kwargs):
    """
    Benchmark graphs for testing community detection algorithms from
    Andrea Lancichinetti, Santo Fortunato and Filippo Radicchi1
    http://arxiv.org/pdf/0805.4770.pdf

    This function is wrapper for exe file.
    In future it should be rewritten by Python-C API

    [FLAG]		    [P]
    :param N		number of nodes
    :param k		average degree
    :param maxk		maximum degree
    :param mut		mixing parameter for the topology
    :param muw		mixing parameter for the weights
    :param beta		exponent for the weight distribution
    :param t1		minus exponent for the degree sequence
    :param t2		minus exponent for the community size distribution
    :param minc		minimum for the community sizes
    :param maxc		maximum for the community sizes
    :param on		number of overlapping nodes
    :param om		number of memberships of the overlapping nodes
    :param C        [average clustering coefficient]
    """

    default = { 'N': 1000,
                 'mut': 0.1,
                 'maxk': 50,
                 'k': 30,
                 'om': 2,
                 'muw': 0.1,
                 'beta': 2,
                 't1': 2,
                 't2': 2,
                 'on': 100
                 }

    default.update(kwargs)

    cwd = os.getcwd()
    os.chdir('../external/Lancichinetti_benchmark/benchmark')

    with open("parameters.dat", 'w') as f:
        f.write('\n'.join('-{} {}'.format(key, default[key]) for key in default))
    with open('outputlog', 'wb') as outputlog:
        if platform.system() == "Windows":
            p = subprocess.Popen('benchmark.exe -f parameters.dat', stdout=outputlog, stderr=outputlog)
        else:
            p = subprocess.Popen('./benchmark -f parameters.dat', shell = True, stdout=outputlog, stderr=outputlog)
        p.wait()

    with open("network.dat") as nw:
        G = nx.read_weighted_edgelist(nw)
    with open("community.dat") as nw:
         comm_inv = {line.split()[0]: line.split()[1:] for line in nw}
    comm = {}
    for key in comm_inv:
        for x in comm_inv[key]:
            t = int(x)
            if t in comm:
                comm[t].append(key)
            else:
                comm[t] = [key]

    os.chdir(cwd)

    return G, comm

def fill_filename(filename, ext):
    return filename if filename.endswith(ext) else filename + ext

def get_edges_list(filename):
    filename = fill_filename(filename, '.edges')
    with open(filename) as f:
        return [list(map(int, line.split())) for line in f]

def get_circles_list(filename):
    filename = fill_filename(filename, '.circles')
    with open(filename) as f:
        return [list(map(int, line.split()[1:])) for line in f]
