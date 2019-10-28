import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import copy
from cluster import Node, Switch, Cluster, Partition
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib import animation
import numpy as np
import bisect

r"""
To show vis of the topology at a given time, we have to keep a json of cluster state of every event step. 
Json format:
    {event_time, pending_jobs, running_jobs, used_nodes {sw_id, node_id, used_gpu, used_cpu} }  
"""


def event_log(logger, event_time, jobs):
    r"""
    :param event_time:
    :param logger: list of dict
    :param cluster: Cluster class
    :param jobs: JobEvent class
    :return:
    """
    event = dict()
    event['event_time'] = event_time
    event['pending_jobs'] = copy.deepcopy(jobs.pending_jobs)
    event['running_jobs'] = copy.deepcopy(jobs.running_jobs)
    used_nodes = dict()
    for jid in jobs.running_jobs:
        job = jobs.submit_jobs[jid]
        for place in job['placements']:
            for node in place['nodes']:
                name = f"{place['switch']}_{node['id']}"
                if name not in used_nodes:
                    used_nodes[name] = {'switch_id': place['switch'], 'node_id': node['id'],
                                        'used_gpu': node['num_gpu'], 'used_cpu': node['num_cpu'],
                                        'jobs': [jid, ]}
                else:
                    used_nodes[name]['used_gpu'] += node['num_gpu']
                    used_nodes[name]['used_cpu'] += node['num_cpu']
                    used_nodes[name]['jobs'].append(jid)

    event['used_nodes'] = [val for k, val in used_nodes.items()]
    logger.append(event)


def cluster_visualization(cluster, logger, d=2, fig_w=8, node_h=15, node_w=16):
    r = d / 2
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_w / node_w * node_h))

    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.20, )
    ax.set_xlim(0, node_w * d)
    ax.set_ylim(0, node_h * d)
    ax.set_xticks(np.arange(r, d * node_w, d))
    ax.set_xticklabels(np.arange(1, node_w + 1))
    if type(logger) == str:
        logger = json.loads(open(logger).read())

    timeline = [v['event_time'] for v in logger]

    row = 0
    par_n = 0
    node_to_cir = dict()
    ticks = []
    norm = mpl.colors.Normalize(vmin=0, vmax=9)
    cmap = cm.get_cmap('gist_heat')
    maps = cm.ScalarMappable(norm=norm, cmap=cmap)
    call_time = True
    anim_pause = True

    for par, sw_dict in cluster.partitions.items():
        col = 0
        ticks.append(row * d + r)
        for sw, node_list in sw_dict.items():
            for nid in node_list:
                if col == node_w:
                    col = 0
                    row += 1
                    ticks[-1] += r
                name = f"{sw}_{nid}"
                node_to_cir[name] = Circle(xy=(col * d + r, row * d + r), alpha=0.8,
                                           radius=r * 0.9, color=maps.to_rgba(0))

                ax.text(col * d + r, row * d + r, cluster.switch_list[sw].node_list[nid].name,
                        horizontalalignment='center', verticalalignment='center')
                ax.add_patch(node_to_cir[name])
                col += 1
        row += 1
        plt.plot([0, node_w * d], [row * d, row * d], 'k--')
        par_n += 1
    ax.set_yticks(ticks)
    ax.set_yticklabels(cluster.partitions.keys())
    fig.canvas.draw_idle()

    axcolor = 'lightgoldenrodyellow'
    axtime = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
    axevent = plt.axes([0.15, 0.03, 0.65, 0.03], facecolor=axcolor)

    time_slider = Slider(axtime, 'Event Time', 0, timeline[-1], valinit=0, valstep=1)
    event_idx_slider = Slider(axevent, 'Event Index', -1, len(timeline), valinit=0, valstep=1)

    anim = None

    def init():
        for k, cir in node_to_cir.items():
            cir.set_color(maps.to_rgba(0))

    def update_time(time):
        nonlocal call_time
        # time = time_slider.val
        init()
        if time >= timeline[0]:
            idx = bisect.bisect_right(timeline, time) - 1
            call_time = False
            event_idx_slider.set_val(idx)
            for node in logger[idx]['used_nodes']:
                node_to_cir['{}_{}'.format(node['switch_id'], node['node_id'])] \
                    .set_color(maps.to_rgba(node['used_gpu']))
        fig.canvas.draw_idle()

    def update_event(idx):
        nonlocal call_time
        if call_time:
            if idx < 0:
                time = 0
            else:
                time = timeline[int(idx)]
            time_slider.set_val(time)
        else:
            call_time = True

    def animate(i):
        init()
        i = (int(event_idx_slider.val) + 1) % len(timeline)
        event_idx_slider.set_val(i)

    def animate_button(event):
        nonlocal anim_pause, anim
        if not anim_pause:
            anim.event_source.start()
            anim_pause = True
        else:
            if anim is None:
                anim = animation.FuncAnimation(fig=fig,
                                               func=animate,
                                               frames=len(logger),
                                               init_func=init,
                                               interval=5,
                                               repeat=True,
                                               )
            anim.event_source.stop()
            anim_pause = False

    time_slider.on_changed(update_time)
    event_idx_slider.on_changed(update_event)

    # animation button
    ax_run = fig.add_axes([0.85, 0.07, 0.12, 0.03], facecolor=axcolor)
    run_button = Button(ax_run, 'Run Sim!')
    run_button.on_clicked(animate_button)

    # cbar_ax2 = fig.add_axes([0.85, 0.2, 0.05, 0.75])
    cbar_ax2 = fig.add_axes([0.9, 0.20, 0.05, 0.60])
    # cbar = fig.colorbar(sca, cax=cbar_ax2, ticks=[0, 4, 9, 14, 19])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax2)
    plt.show()


if __name__ == '__main__':
    cluster = Cluster()
    cluster.init_from_csv('Cluster_Info/cluster_info.csv')
    partition = Partition(cluster, 'Cluster_Info/sinfo.csv')
    cluster_visualization(cluster, 'cluster_log.json')
