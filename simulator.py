'''
main script to run the code.
'''
import csv
import sys
import os
import time
import math
from opt import opt
from cluster import Node, Switch, Cluster, Partition
from job import JobEvents
import scheduling
from tracing_viewer import generate_trace_json
import json
import numpy as np

args = opt


def avg_pending_time(jobs):
    pending_time = [j['pending_time'] for jid, j in jobs.items()]
    return np.average(pending_time)


def main():
    # Parse cluster info
    cluster = Cluster()
    cluster.init_from_csv(args.cluster_info)
    partition = Partition(cluster, args.partition)
    logger = None
    if args.logger_file != '':
        logger = list()
    # Parse job trace
    jobs = JobEvents()
    jobs.init_jobs_from_csv(args.job_trace)
    jobs.init_events_from_jobs()
    # Start sim
    if args.schedule == 'fifo':
        scheduling.fifo_sim(cluster, jobs, logger)
    if args.schedule == 'sjf':
        scheduling.sjf_sim(cluster, jobs, logger)
    if args.schedule == 'lsf':
        scheduling.lsf_sim(cluster, jobs, logger)

    print('{} Average Waiting Time: {}'.format(args.schedule, avg_pending_time(jobs.submit_jobs)))

    if args.chrome_trace_file != '':
        generate_trace_json(jobs, args.chrome_trace_file)
    if logger is not None:
        with open(args.logger_file, 'w') as json_file:
            json.dump(logger, json_file)


if __name__ == '__main__':
    main()

# fifo Average Waiting Time: 8883.591556728232
# lsf Average Waiting Time: 8770.715567282323
# sjf Average Waiting Time: 4017.566226912929