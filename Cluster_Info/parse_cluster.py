CMAP = 'Accent'
def parse_ib(file='iblinkinfo.txt'):
    fh = open(file)

    nodes = dict()
    switches = dict()
    line = fh.readline()

    while line:
        if line.startswith('CA'):
            name = line.split()[1]
            if name == 'localhost':
                line = fh.readline()
                line = fh.readline()
                continue
            if '-' in name:
                name = name.split('-')[-1]
            nodes[name] = fh.readline().split("\"")[-2].split('/')[-2]
            line = fh.readline()

        elif line.startswith('Switch'):
            name = line.split()[-1][:-1].split('/')[-2]
            switches[name] = []
            line = fh.readline()
            while not line.startswith("Switch") and not line.startswith("CA"):
                # print(line)
                link = line.strip().split('[  ]')[-1].strip().split("\"")[1]
                if link != '':
                    link = link.split()[0]
                    if '-' in link:
                        link = link.split('-')[-1]
                    if '/' in link:
                        link = link.split('/')[-2]
                    if link not in switches[name] and link != 'localhost':
                        switches[name].append(link)
                line = fh.readline()
            switches[name].sort()
    # print('-'*8+'Node'+'-'*8)
    # [print(key, node) for key,node in nodes.items()]
    # print('-'*8+'Switch'+'-'*8)
    # [print(key, switch) for key,switch in switches.items()]
    return nodes, switches


def parse_partitions(file ='sinfo.txt'):
    lines = open(file).readlines()[1:]
    partitions = dict()
    for line in lines:
        line = line.split()
        name = line[0]
        nds = line[-1]
        if name not in partitions:
            partitions[name] = []
        if '[' in nds:
            tmp = nds.split('[')[-1][:-1].split(',')
            for n in tmp:
                if '-' not in n:
                    partitions[name].append(n)
                else:
                    a, b = (eval(v) for v in n.split('-'))
                    for i in range(a,b+1):
                        partitions[name].append(f'{i}')
    for k, v in partitions.items():
        v.sort(key=lambda x:eval(x))
    # print(k, v)
    return partitions

# """
# Create cluster info file
# """

import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
#import colormap
from matplotlib import cm
import matplotlib


def get_rgb(x, vmin=0, vmax=10):
    # normalize item number values to colormap
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # colormap possible values = viridis, jet, spectral
    return matplotlib.colors.rgb2hex(cm.get_cmap(CMAP)(norm(x))[:3])

def draw_topo(nodes, switches, partition=None):
    B = nx.Graph()
    # Add nodes with the node attribute "bipartite"
    nd = nodes.keys()
    ssw = [f'S0{i}' for i in range(1, 7)]
    sw = list(set(switches.keys()) - set(ssw))
    B.add_nodes_from(nd)
    B.add_nodes_from(sw)
    B.add_nodes_from(ssw)

    # Add edges only between nodes of opposite node sets
    B.add_edges_from([(key, val) for key, val in nodes.items()],)
    for key, val in switches.items():
        B.add_edges_from([(key, v) for v in val],)

    for a, b in B.edges:
        if a.startswith("S") or b.startswith("S"):
            B.edges[a, b]['weight'] = 50
            if a.startswith("S") and b.startswith("S"):
                B.edges[a, b]['weight'] = 1
        else:
            B.edges[a, b]['weight'] = 100
    #
    node_par = dict()
    par_id = dict()
    cnt = 0
    if partitions is not None:
        for k, v in partitions.items():
            par_id[k] = cnt
            cnt+=1
            for n in v:
                node_par[n] = k


    colors = []
    for nd in B.nodes:
        if nd.startswith('L'):
            colors.append('g')
        elif nd.startswith('S'):
            colors.append('r')
        else:
            if nd in node_par:
                colors.append(get_rgb(par_id[node_par[nd]],0,len(par_id)))
            else:
                colors.append('b')

    pos = nx.spring_layout(B)
    nx.draw(B, pos, node_color=colors, with_labels=True, alpha=0.8, size=300)
    plt.draw()
    sm = plt.cm.ScalarMappable(cmap=cm.get_cmap(CMAP), norm=plt.Normalize(vmin=0, vmax=len(par_id)))
    cbar = plt.colorbar(sm, shrink=0.6)
    cbar.ax.set_yticklabels(partitions.keys())
    plt.show()


def generate_csv(nodes, switches):
    fh = open('cluster_info.csv','w')
    fh.write('node_id,node_gpu,node_cpu,node_mem,gpu_type,switch_id\n')
    for k, v in nodes.items():
        if k.isdigit():
            fh.write(f"{k},8,128,256,V100,{v}\n")


if __name__ == '__main__':
    nodes, switches = parse_ib()
    partitions = parse_partitions()
    # draw_topo(nodes, switches, partitions)
    generate_csv(nodes, switches)