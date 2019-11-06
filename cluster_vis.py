import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import colorcet as cc
import json
import copy
from cluster import Node, Switch, Cluster, Partition
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib import animation
import numpy as np
import bisect
import argparse
from opt import opt

args = opt

r"""
To show vis of the topology at a given time, we have to keep a json of cluster state of every event step. 
Json format:
    {event_time, pending_jobs, running_jobs, used_nodes {sw_id, node_id, used_gpu, used_cpu} }  
"""


def event_log(logger, event_time, jobs, track_nodes=False):
    r"""
    :param event_time:
    :param logger: list of dict
    :param cluster: Cluster class
    :param jobs: JobEvent class
    :return:
    """
    event = dict()
    event['time'] = event_time
    event['p_jobs'] = copy.deepcopy(jobs.pending_jobs)
    event['r_jobs'] = copy.deepcopy(jobs.running_jobs)

    if track_nodes:
        used_nodes = dict()
        for jid in jobs.running_jobs:
            job = jobs.submit_jobs[jid]
            for place in job['placements']:
                for node in place['nodes']:
                    name = f"{place['switch']}_{node['id']}"
                    if name not in used_nodes:
                        used_nodes[name] = {'sid': place['switch'], 'nid': node['id'],
                                            'used_gpu': node['num_gpu'], 'used_cpu': node['num_cpu'], }
                    else:
                        used_nodes[name]['used_gpu'] += node['num_gpu']
                        used_nodes[name]['used_cpu'] += node['num_cpu']

        event['used_nodes'] = [val for k, val in used_nodes.items()]

    logger.append(event)


def get_alloced_nodes(job_placements, time):
    for res, s_t, e_t in job_placements:
        if s_t <= time <= e_t:
            return res
    return []


def cluster_visualization(cluster, logger, trace, d=4, fig_w=12, node_h=15, node_w=16, schedule='fifo', save='',
                          frames=None, draw_mitigate=True):
    r = d / 2
    l = d / 4
    border = d / 20
    h = d / 4 * 3
    lh = h / 3
    # fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_w / node_w / d * node_h * h), dpi=80)
    fig = plt.figure(figsize=(fig_w * 1.5, fig_w / node_w / d * node_h * h * 1.2), dpi=80, constrained_layout=True)
    fig.suptitle("Deep Learning Cluster Simulation -- {} Schedule".format(schedule))
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[:, :-1])
    ax.set_title('Monitor')
    ax_pending = fig.add_subplot(gs[0, -1])
    ax_pending.set_title('#Pending Jobs')
    ax_running = fig.add_subplot(gs[1, -1])
    ax_running.set_title('#Running Jobs')

    plt.subplots_adjust(left=0.15, bottom=0.20, )
    ax.set_xlim(0, node_w * d)
    ax.set_ylim(0, node_h * h)
    ax.set_xticks(np.arange(r, d * node_w, d))
    ax.set_xticklabels(np.arange(1, node_w + 1))
    if type(logger) == str:
        logger = json.loads(open(logger).read())
    if type(trace) == str:
        raw_trace = json.loads(open(trace).read())
        trace = dict()
        for item in raw_trace['traceEvents']:
            if 'placements' not in item['args']:
                continue
            placements = json.loads(item['args']['placements'].replace('\'', '"'))
            tmp = []
            for sw in placements:
                for nd in sw['nodes']:
                    for i in nd['gpu_assign']:
                        tmp.append('{}_{}_{}'.format(sw['switch'], nd['id'], i))
            if item['tid'] not in trace:
                trace[item['tid']] = []
            trace[item['tid']].append((tmp, item['args']['s_t'], item['args']['e_t']))
        del raw_trace

    timeline = [v['time'] for v in logger]
    num_pending = [len(v['p_jobs']) for v in logger]
    num_running = [len(v['r_jobs']) for v in logger]

    ax_pending.plot(num_pending[:])
    ax_running.plot(num_running[:])

    row = 0
    par_n = 0
    gpu_to_rect = dict()
    node_pos = dict()
    ticks = []
    frames = len(timeline) if frames is None else frames
    # norm = mpl.colors.Normalize(vmin=0, vmax=9)
    cmap = cc.cm.glasbey  # cm.get_cmap('gist_heat')
    call_time = True
    anim_pause = True

    for par, sw_dict in cluster.partitions.items():
        col = 0
        ticks.append(row * h + h / 2)
        for sw, node_list in sw_dict.items():
            for nid in node_list:
                if col == node_w:
                    col = 0
                    row += 1
                    ticks[-1] += h / 2

                num_gpu = cluster.switch_list[sw].node_list[nid].num_gpu
                offset_x = col * d
                offset_y = row * h
                node_rect = Rectangle(xy=(offset_x, offset_y), width=d, height=h, alpha=0.6,
                                      facecolor='#262626', edgecolor='k')
                ax.add_patch(node_rect)

                coli = 0
                rowi = 0
                for i in range(num_gpu):
                    if coli == 4:
                        coli = 0
                        rowi += 1
                    gpu = Rectangle(xy=(coli * l + border + offset_x, rowi * lh + border + offset_y),
                                    width=l - 2 * border,
                                    height=lh - 2 * border,
                                    color='w'
                                    )
                    ax.add_patch(gpu)
                    gpu_to_rect[f"{sw}_{nid}_{i}"] = gpu
                    coli += 1

                ax.text(col * d + r, row * h + 2 * lh, cluster.switch_list[sw].node_list[nid].name, color='w',
                        horizontalalignment='center', )  # verticalalignment='center')

                node_pos[f'{sw}_{nid}'] = (col * d + r, row * h + 2 * lh)

                col += 1
        row += 1
        ax.plot([0, node_w * d], [row * h, row * h], 'b--')
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
    running_gpus = []
    gpu_j_map = dict()
    j_node_map = dict()
    links = []

    def init():
        for k, cir in gpu_to_rect.items():
            cir.set_color('w')

    def update_time(time):
        nonlocal call_time, j_node_map
        # time = time_slider.val
        # init()
        if time >= timeline[0]:
            idx = bisect.bisect_right(timeline, time) - 1
            call_time = False
            event_idx_slider.set_val(idx)
            new_gpus = []
            new_j_node_map = dict()
            if draw_mitigate:
                while len(links) > 0:
                    arc = links.pop()
                    arc.remove()

            for j in logger[idx]['r_jobs']:
                j_gpu = get_alloced_nodes(trace[j], timeline[idx] + 0.1)
                new_gpus += j_gpu
                new_j_node_map[j] = set()
                for g in j_gpu:
                    gpu_j_map[g] = j
                    new_j_node_map[j].add(g[:-2])
                if draw_mitigate:
                    if j in j_node_map:
                        new_nodes = list(new_j_node_map[j] - j_node_map[j])
                        old_nodes = list(j_node_map[j] - new_j_node_map[j])
                        for nn, on in zip(new_nodes, old_nodes):
                            arc = ax.annotate("", xy=node_pos[nn], xytext=node_pos[on],
                                              arrowprops=dict(arrowstyle="->", color=cmap(j % cmap.N), alpha=0.8,
                                                              shrinkA=5, shrinkB=5,
                                                              patchA=None, patchB=None,
                                                              connectionstyle="arc3,rad=0.3",
                                                              ),
                                              )
                            links.append(arc)

            j_node_map = new_j_node_map

            for g in running_gpus:
                if g not in new_gpus:
                    gpu_to_rect[g].set_color('w')
            running_gpus[:] = new_gpus[:]
            for g in running_gpus:
                gpu_to_rect[g].set_color(cmap(gpu_j_map[g] % cmap.N))

            ax_pending.cla()
            ax_pending.plot(num_pending[:idx + 1])
            ax_running.cla()
            ax_running.plot(num_running[:idx + 1])
            ax_pending.set_title('#Pending Jobs')
            ax_running.set_title('#Running Jobs')
            ax_running.set_xlabel('Event_Idx')

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
        # init()
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
                                               frames=frames,
                                               init_func=init,
                                               interval=1,
                                               repeat=True,
                                               )
                if save != '':
                    anim.save(save, writer='imagemagick')
            anim.event_source.stop()
            anim_pause = False

    def text_submit(val):
        val = eval(val)
        time_slider.set_val(val)

    def key_respond(event):
        if event.key == 'a':
            val = event_idx_slider.val
            event_idx_slider.set_val(max(0, val - 1))
        elif event.key == 'd':
            val = event_idx_slider.val
            event_idx_slider.set_val(min(len(timeline), val + 1))

    time_slider.on_changed(update_time)
    event_idx_slider.on_changed(update_event)

    # animation button
    ax_run = fig.add_axes([0.85, 0.03, 0.12, 0.03], facecolor=axcolor)
    run_button = Button(ax_run, 'Run Sim!')
    run_button.on_clicked(animate_button)

    ax_txt = fig.add_axes([0.90, 0.1, 0.07, 0.03], facecolor=axcolor)
    textbox = TextBox(ax_txt, 'Event Time')
    textbox.on_submit(text_submit)

    # # cbar_ax2 = fig.add_axes([0.85, 0.2, 0.05, 0.75])
    # cbar_ax2 = fig.add_axes([0.9, 0.20, 0.05, 0.60])
    # # cbar = fig.colorbar(sca, cax=cbar_ax2, ticks=[0, 4, 9, 14, 19])
    # cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), cax=cbar_ax2)

    fig.canvas.mpl_connect('key_press_event', key_respond)

    plt.show()


if __name__ == '__main__':
    cluster = Cluster()
    cluster.init_from_csv('Cluster_Info/cluster_info.csv')
    partition = Partition(cluster, 'Cluster_Info/sinfo.csv')
    cluster_visualization(cluster, args.logger_file, args.chrome_trace_file)
