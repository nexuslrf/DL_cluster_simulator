'''
main script to run the code.
'''
import csv
import sys
import os
import time
import math
from opt import opt
from cluster import Node, Switch, Cluster
from job import JobEvents
import scheduling

args = opt


def main():
    # Parse cluster info
    cluster = Cluster()
    cluster.init_from_csv(args.cluster_info)
    # Parse job trace
    jobs = JobEvents()
    jobs.init_jobs_from_csv(args.job_trace)
    jobs.init_events_from_jobs()
    # Start sim
    if args.schedule == 'fifo':
        scheduling.fifo_sim(cluster, jobs)


if __name__ == '__main__':
    main()
