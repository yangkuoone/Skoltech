import os
import subprocess
import networkx as nx

def copra(A, K=None):
    #TODO: may be something wrong and we should fix it. (comm + or - 1)
    data_dir = '../external/COPRA/'
    data_file_name = 'test'
    COPRA_data_file_path = os.path.join(data_dir, data_file_name) + '.COPRA'
    java_path = '../external/COPRA/copra.jar'

    toCorpraFormat(A, COPRA_data_file_path)

    args = '-w -v 2 -prop 500000'

    with open('../external/COPRA/COPRA_output.log', 'w') as output_f:
        subprocess.call('java -cp \"{}\" COPRA \"{}\" {}'.format(java_path, COPRA_data_file_path, args),
                        stdout=output_f, stderr=output_f)

    with open('clusters-test.COPRA', 'r') as f:
        res = [[int(x) for x in line.split()] for indx, line in enumerate(f)]
    return res

def toCorpraFormat(A, file_name):
    G = nx.Graph(A)
    with file(file_name, 'w') as f:
        [f.write('{} {} {}\n'.format(e[0]+1, e[1]+1, e[2]['weight'])) for e in G.edges(data=True)]