import os
import subprocess
import platform


def sorted_t(l):
    return sorted(list(l), key=lambda x: int(x))


def NMI(comms, Comm_True):
    #min_indx = min(c for comm in comms for c in comm) - 1

    with open('../external/Lancichinetti_benchmark/nmi/clu1', 'w') as f:
            for indx, comm in enumerate(comms):
                for c in sorted_t(comm):
                    f.write("1 {} {}\n".format(c, indx))

    with open('../external/Lancichinetti_benchmark/nmi/clu2', 'w') as f:
        for indx, comm in enumerate(Comm_True):
                for c in sorted_t(comm):
                    f.write("1 {} {}\n".format(c, indx))

    with open('../external/Lancichinetti_benchmark/nmi/outputlog-nmi', 'wb') as outputlog:
        if platform.system() == "Windows":
            p = subprocess.Popen("../external/Lancichinetti_benchmark/nmi/nmi.exe \
            ../external/Lancichinetti_benchmark/nmi/clu1 \
            ../external/Lancichinetti benchmark/nmi/clu2", stdout=outputlog, stderr=outputlog)
        else:
            p = subprocess.Popen("../external/Lancichinetti_benchmark/nmi/./nmi \
            ../external/Lancichinetti_benchmark/nmi/clu1 \
            ../external/Lancichinetti_benchmark/nmi/clu2", shell=True, stdout=outputlog, stderr=outputlog)
        p.wait()

    with open('../external/Lancichinetti_benchmark/nmi/outputlog-nmi', 'r') as f:
        for line in f:
            res = line.split()
            if res[0] == 'Multiplex':
                return float(res[-1])
