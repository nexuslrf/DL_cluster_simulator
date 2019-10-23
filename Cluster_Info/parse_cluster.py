import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
#import colormap
from matplotlib import cm
import matplotlib
import csv
import numpy as np
from matplotlib import ticker

CMAP = 'tab20'


def parse_ib(file='iblinkinfo.txt'):
    fh = open(file)

    nodes = dict()
    switches = dict()
    line = fh.readline()

    while line:
        if line.startswith('CA'):
            name = line.split()[1]
            if '-' in name:
                name = name.split('-')[-1]
                nodes[name] = fh.readline().split("\"")[-2].split('/')[-2]
            else:
                line = fh.readline()
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
                    flag = False
                    if '-' in link:
                        link = link.split('-')[-1]
                        flag = True
                    if '/' in link:
                        link = link.split('/')[-2]
                        flag = True
                    if flag and link not in switches[name]:
                        switches[name].append(link)
                line = fh.readline()
            switches[name].sort()
    # print('-'*8+'Node'+'-'*8)
    # [print(key, node) for key,node in nodes.items()]
    # print('-'*8+'Switch'+'-'*8)
    # [print(key, switch) for key,switch in switches.items()]
    return nodes, switches


"""
cmd to get sinfo.csv:
```bash
    sinfo -lN -o "%P,%N,%c,%G,%m" >sinfo.csv
    sed -i '1d' sinfo.csv
``` 
"""


def parse_sinfo(file='sinfo.csv'):
    fh = open(file)
    reader = csv.DictReader(fh)
    partitions = dict()
    nodes_info = dict()
    for line in reader:
        name = line['PARTITION']
        node_name = line['NODELIST'].split('-')[-1]
        if name not in partitions:
            partitions[name] = list()
        partitions[name].append(node_name)
        for k, v in line.items():
            node = dict()
            node['partition'] = name
            node['gpu'] = eval(line['GRES'].split(':')[-1])
            node['cpu'] = eval(line['CPUS'])
            node['mem'] = eval(line['MEMORY'])//1000
            nodes_info[node_name] = node
    for k, v in partitions.items():
        v.sort(key=lambda x: eval(x))
    # print(k, v)
    return nodes_info, partitions

# """
# Create cluster info file
# """


def get_rgb(x, vmin=0, vmax=10):
    # normalize item number values to colormap
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # colormap possible values = viridis, jet, spectral
    return matplotlib.colors.rgb2hex(cm.get_cmap(CMAP, vmax-vmin+1)(norm(x))[:3])


def draw_topo(nodes, switches, partitions=None):
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
    par_tick = ['L1 SW', 'L2 SW']
    cnt = 0
    if partitions is not None:
        for k, v in partitions.items():
            par_id[k] = cnt
            cnt+=1
            par_tick.append(k)
            for n in v:
                node_par[n] = k
    par_tick.append('Others')

    colors = []
    for nd in B.nodes:
        if nd.startswith('L'):
            colors.append(1)
        elif nd.startswith('S'):
            colors.append(0)
        else:
            if nd in node_par:
                colors.append(2+par_id[node_par[nd]])
            else:
                colors.append(cnt)
    pos = dict()
    pos.update(nx.circular_layout(ssw, 1, [0, 0]))
    pos.update(nx.circular_layout(sw, 5, [0, 0]))
    for k, v in switches.items():
        if k not in ssw:
            tmp = []
            for i in v:
                if i not in ssw and i not in sw:
                    tmp.append(i)
            pos.update(nx.spring_layout(tmp, 1, center=pos[k]))

    cmap = plt.cm.get_cmap(CMAP)
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, cnt+3, 1), cmap.N)
    nx.draw(B, pos, node_color=colors, with_labels=True, alpha=0.8, size=300, cmap=cmap, norm=norm)
    # pc = matplotlib.collections.PatchCollection(edges, cmap=cmap)
    # pc.set_array(colors)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),  shrink=0.8)
    tick_locator = ticker.MaxNLocator(nbins=len(par_tick))
    cb.locator = tick_locator
    cb.update_ticks()
    cb.ax.set_yticklabels(par_tick)
    plt.draw()
    plt.show()


def generate_csv(nodes, nodes_info=None):
    fh = open('cluster_info.csv', 'w')
    fh.write('node_id,node_gpu,node_cpu,node_mem,gpu_type,switch_id\n')
    for k, v in nodes.items():
        if k.isdigit():
            if nodes_info is not None and k in nodes_info:
                fh.write("{},{},{},{},V100,{}\n".format(k, nodes_info[k]['gpu'], nodes_info[k]['cpu'], nodes_info[k]['mem'], v))
            else:
                fh.write(f"{k},8,56,128,V100,{v}\n")


if __name__ == '__main__':
    nds, sws = parse_ib()
    ninfo, pars = parse_sinfo()
    draw_topo(nds, sws, pars)
    generate_csv(nds, ninfo)