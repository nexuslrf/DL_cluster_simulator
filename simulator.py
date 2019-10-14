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
    # if args.schedule == 'fifo':
    #     schedulors.fifo_sim(cluster, job_events)
    finish = False
    while not finish:
        if jobs.PC:
            # @TODO cluster out of usage
            pass

        event = jobs.events[jobs.PC]
        event_time = event['time']
        for job in event['end_jobs']:
            # @TODO
            cluster.release_job_res(job)

        for job in event['start_jobs']:
            jobs.pend_jobs(job)

        if cluster.free_gpus > 0:
            # We can start jobs!
            jobs.issuing_jobs = []
            for job in jobs.pending_jobs:
                if cluster.try_alloc_res(job):
                    jobs.issuing_jobs.append(job)

            for job in jobs.issuing_jobs:
                jobs.issue_jobs(job, event_time)
                jobs.add_end_events(job)
                print('job[{}] start from pending'.format(job['jid']))

        jobs.PC += 1
