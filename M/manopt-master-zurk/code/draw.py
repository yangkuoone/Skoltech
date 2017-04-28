import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

col = ['b', 'r', 'g', 'm', 'c', 'y', '#56A0D3', '#ED9121', '#00563F', '#062A78', '#703642', '#C95A49',
       '#92A1CF', '#ACE1AF', '#007BA7', '#2F847C', '#B2FFFF', '#4997D0',
       '#DE3163', '#EC3B83', '#007BA7', '#2A52BE', '#6D9BC3', '#007AA5',
       '#E03C31', '#AAAAAA']

def draw_dolan_more(experiments_result_pd, disp, title=''):
    experiments_result_pd = experiments_result_pd[disp]
    min_rate = experiments_result_pd.min(axis=1)
    experiments_result_pd_norm = experiments_result_pd.div(min_rate, axis=0)
    plt.figure(figsize=(15,9))
    for i, col_name in enumerate(disp):
        s = experiments_result_pd_norm[col_name].unique()
        s.sort()
        y = [(experiments_result_pd_norm[col_name] <= x).mean() for x in s]
        plt.plot(s, y, c=col[i])
        #print(y)
    plt.legend(disp, loc=4)
    plt.xlim([min(s),max(s)])
    plt.ylim([0,1.05])
    plt.grid()
    plt.title(title)

def draw_matrix(photo_l, title="", hide_ticks=True):
    if photo_l.shape[0] / photo_l.shape[1] > 5 or photo_l.shape[0] / photo_l.shape[1] < 0.2:
        k = np.floor(np.max(photo_l.shape) / np.min(photo_l.shape) / 2)
        if photo_l.shape[1] < photo_l.shape[0]:
            photo_l = np.reshape(np.tile(photo_l.copy(), (k, 1)).T, (photo_l.shape[1] * k, photo_l.shape[0])).T
        else:
            photo_l = np.reshape(np.tile(photo_l.copy(), (1, k)), (photo_l.shape[0] * k, photo_l.shape[1]))
    ax = plt.gca()
    im = plt.imshow(photo_l, interpolation='none',cmap='jet')
    if hide_ticks:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.title(title, y=1.02, x=0.6)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad=0.07)
    plt.colorbar(im, cax=cax, ticks=np.linspace(np.min(photo_l), np.max(photo_l), 5))


def draw_graph(G, comms, save_path=None):

    pos=nx.spring_layout(G)
    print('Pos!')

    size = 24
    ax = plt.figure(figsize=(size, size))
    #nx.draw_networkx(G_test, pos=pos, node_size=25, alpha=0.3, linewidths=0, width=0.5, with_labels=False)
    #max_w = max(e[2]['weight'] for e in G.edges(data=True))

    node_size = 2 * size
    drawn = []
    print('1')
    for i, comm in enumerate(comms):
        col_i = i
        print('.')
        G_part = nx.subgraph(G, comm)
        #width = [0.2 + 1.5 * e[2]['weight'] / max_w for e in G_part.edges(data=True)]
        width = 0.5
        edgelist = list(G_part.edges())
        nx.draw_networkx_edges(G, pos, edgelist=edgelist, width=width, alpha=0.2, edge_color=col[col_i if col_i < len(col) else -1])
        drawn.extend(edgelist)
    print('2')
    temp = list(set(G.edges())- set(drawn))
    print('2.5')
    nx.draw_networkx_edges(G, pos, edgelist=temp , width=width, alpha=0.2)
    print('3')
    nx.draw_networkx_nodes(G, pos, node_color='#CCCCCC', node_size=node_size, alpha=1, linewidths=0)
    if len(comms) > len(col):
        print ('WARNING: too low colors count')
    print('4')
    for j in G:
        node_cols = [(col_i if col_i < len(col) else col_i%len(col)) for col_i, comm in enumerate(comms) if j in comm]
        for k, col_i in enumerate(node_cols):
            nx.draw_networkx_nodes(G, pos, nodelist=[j], node_color=col[col_i], node_size=(len(node_cols)-1 * k) * node_size,
                                   alpha=1, linewidths=0, width=0.3, with_labels=False)

    bord = np.array((np.percentile(np.array(list(pos.values())), 0, axis=0), np.percentile(np.array(list(pos.values())), 100, axis=0)))
    plt.xlim(bord[:,0])
    plt.ylim(bord[:,1])
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path, dpi=400)



def draw_groups(A, F, ids=None, names=None, figname = 'NoName', png=True, pdf=False, display=False, svg=False, dpi=2300):
    """
    Old but cool visualisation tool,
    """
    # TODO: need to be remaked to new comms (list of lists, not F)
    N, K = F.shape

    C = F > np.sum(A) / (A.shape[0] * (A.shape[0] - 1))
    indx = np.argmax(F, axis=1)
    for i in xrange(N):
        C[i, indx[i]] = True
    print F
    print C

    comm = [[] for i in xrange(N)]
    for x, y in zip(*np.where(C)):
        comm[x].append(y)
    u_comm = np.unique(comm)

    comm2id = []
    for u in u_comm:
        comm2id.append([i for i, c in enumerate(comm) if c == u])

    G = nx.Graph(A)
    plt.figure(num=None, figsize=(10, 10))

    pos = []
    centers = [np.array([0, 1])]
    angle = np.pi / K
    turn = np.array([[np.cos(2*angle), np.sin(2*angle)], [-np.sin(2*angle), np.cos(2*angle)]])
    radius = np.sin(angle)
    new_pos = {i: [] for i in xrange(N)}

    U, s, V = np.linalg.svd(F.T.dot(F))
    posSVD =[x[0] for x in sorted([x for x in enumerate(U[0])], key= lambda x: x[1])]

    for i in xrange(K):
        if i + 1 != K:
            centers.append(turn.dot(centers[-1]))

    for i in xrange(K):
        for key, value in nx.spring_layout(G.subgraph(np.where(C[:, posSVD[i]])[0])).iteritems():# positions for all nodes
            new_pos[key].append(value * radius + 0.8 * centers[posSVD[i]])

    for key in new_pos:
        new_pos[key] = np.sum(np.array(new_pos[key]), axis=0) / (1.5 * len(new_pos[key])) ** 1.2

    for val in comm2id:
        if len(comm[val[0]]) < 2:
            continue
        m = np.mean(np.array([new_pos[x] for x in val]), axis=0)
        for x in val:
            new_pos[x] = 0.8 * len(comm[val[0]]) * (new_pos[x] - m) + m

    nx.draw_networkx_edges(G, new_pos, width=0.25, alpha=0.07)
    nx.draw_networkx_nodes(G, new_pos, node_color='#BBBBBB', node_size=15, alpha=1, linewidths=0)
    for j in xrange(C.shape[0]):
        k = 0
        for i in xrange(C.shape[1]):
            if(C[j][i]):
                nx.draw_networkx_nodes(G, new_pos, nodelist=[j], node_color=col[i], node_size=10-1*k,
                                       alpha=0.6, linewidths=0)
                k += 1
    if ids is not None and names is not None:
        from transliterate import translit
        labels = {i: u' '.join([str(n) for n in np.where(c)[0]]) + u'\n> {} <'.format(translit(names[ids[i]].replace(u'\u0456', u'~'), u'ru', reversed=True)) for i, c in enumerate(C)}
        nx.draw_networkx_labels(G, new_pos, labels, font_size=0.1)
    plt.axis('off')

    if pdf:
        plt.savefig("../plots/{}.pdf".format(figname))
    if png:
        plt.savefig("../plots/{}.png".format(figname), dpi=dpi)
    if svg:
        plt.savefig("../plots/{}.svg".format(figname))
    if display:
        plt.show() # display