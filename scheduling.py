from cluster_vis import event_log


def fifo_sim(cluster, jobs, logger=None, policy='first-fit', fit_first=True, **kwargs):
    r"""
    FIFO scheduling baseline
    :param policy:
    :param logger:
    :param cluster:
    :param jobs:
    :return:
    """
    while jobs.PC < len(jobs.events):
        if jobs.check_overload():
            # cluster out of usage
            print("This cluster is not large enough to run the job")
            break

        event = jobs.events[jobs.PC]
        event_time = event['time']
        if 'end_jobs' in event:
            for jid in event['end_jobs']:
                job = jobs.submit_jobs[jid]
                if cluster.release_job_res(job):
                    jobs.finish_jobs('COMPLETED', job)
                else:
                    jobs.finish_jobs('ERROR', job)

        if 'start_jobs' in event:
            for jid in event['start_jobs']:
                job = jobs.submit_jobs[jid]
                jobs.pend_jobs(job)

        # We can start jobs!
        issuing_jobs = []
        for par, queue in jobs.pending_queue.items():
            for jid in queue:
                job = jobs.submit_jobs[jid]
                if cluster.try_alloc_res(job, policy=policy):
                    issuing_jobs.append(jid)
                elif not fit_first:
                    break

        for jid in issuing_jobs:
            job = jobs.submit_jobs[jid]
            jobs.issue_jobs(job, event_time)
            jobs.add_event(job, job['end_time'], 'end_jobs')

        jobs.PC += 1
        print(f"time[{event_time}] ", end='')
        cluster.report()
        print(f"time[{event_time}] ", end='')
        jobs.report()
        if logger is not None:
            event_log(logger, event_time, jobs)


def sjf_sim(cluster, jobs, logger=None, policy='first-fit', fit_first=True, **kwargs):
    r"""
    FIFO scheduling baseline
    :param logger:
    :param cluster:
    :param jobs:
    :return:
    """
    while jobs.PC < len(jobs.events):
        if jobs.check_overload():
            # cluster out of usage
            print("This cluster is not large enough to run the job")
            break

        event = jobs.events[jobs.PC]
        event_time = event['time']
        if 'end_jobs' in event:
            for jid in event['end_jobs']:
                job = jobs.submit_jobs[jid]
                if cluster.release_job_res(job):
                    jobs.finish_jobs('COMPLETED', job)
                else:
                    jobs.finish_jobs('ERROR', job)

        if 'start_jobs' in event:
            for jid in event['start_jobs']:
                job = jobs.submit_jobs[jid]
                jobs.pend_jobs(job)

        # We can start jobs!
        issuing_jobs = []
        for par, queue in jobs.pending_queue.items():
            queue.sort(key=lambda jid: jobs.submit_jobs[jid]['running_time'])
            for jid in queue:
                job = jobs.submit_jobs[jid]
                if cluster.try_alloc_res(job, policy=policy):
                    issuing_jobs.append(jid)
                elif not fit_first:
                    break

        for jid in issuing_jobs:
            job = jobs.submit_jobs[jid]
            jobs.issue_jobs(job, event_time)
            jobs.add_event(job, job['end_time'], 'end_jobs')

        jobs.PC += 1
        print(f"time[{event_time}] ", end='')
        cluster.report()
        print(f"time[{event_time}] ", end='')
        jobs.report()
        if logger is not None:
            event_log(logger, event_time, jobs)


def lsf_sim(cluster, jobs, logger=None, policy='first-fit', fit_first=True, **kwargs):
    r"""
    Least service/resource (num_gpu) first scheduling baseline
    :param fit_first:
    :param policy:
    :param logger:
    :param cluster:
    :param jobs:
    :return:
    """
    while jobs.PC < len(jobs.events):
        if jobs.check_overload():
            # cluster out of usage
            print("This cluster is not large enough to run the job")
            break

        event = jobs.events[jobs.PC]
        event_time = event['time']
        if 'end_jobs' in event:
            for jid in event['end_jobs']:
                job = jobs.submit_jobs[jid]
                if cluster.release_job_res(job):
                    jobs.finish_jobs('COMPLETED', job)
                else:
                    jobs.finish_jobs('ERROR', job)

        if 'start_jobs' in event:
            for jid in event['start_jobs']:
                job = jobs.submit_jobs[jid]
                jobs.pend_jobs(job)

        # Sort pending queue
        # We can start jobs!
        issuing_jobs = []
        for par, queue in jobs.pending_queue.items():
            queue.sort(key=lambda jid: jobs.submit_jobs[jid]['num_gpu'])
            for jid in queue:
                job = jobs.submit_jobs[jid]
                if cluster.try_alloc_res(job, policy=policy):
                    issuing_jobs.append(jid)
                elif not fit_first:
                    break

        for jid in issuing_jobs:
            job = jobs.submit_jobs[jid]
            jobs.issue_jobs(job, event_time)
            jobs.add_event(job, job['end_time'], 'end_jobs')

        jobs.PC += 1
        print(f"time[{event_time}] ", end='')
        cluster.report()
        print(f"time[{event_time}] ", end='')
        jobs.report()
        if logger is not None:
            event_log(logger, event_time, jobs)


# def lpf_sim(cluster, jobs, logger=None, policy='first-fit', fit_first=True):
#     r"""
#     Longest pending first scheduling baseline
#     :param fit_first:
#     :param policy:
#     :param logger:
#     :param cluster:
#     :param jobs:
#     :return:
#     """
#     while jobs.PC < len(jobs.events):
#         if jobs.check_overload():
#             # cluster out of usage
#             print("This cluster is not large enough to run the job")
#             break
#
#         event = jobs.events[jobs.PC]
#         event_time = event['time']
#         if 'end_jobs' in event:
#             for jid in event['end_jobs']:
#                 job = jobs.submit_jobs[jid]
#                 if cluster.release_job_res(job):
#                     jobs.finish_jobs('COMPLETED', job)
#                 else:
#                     jobs.finish_jobs('ERROR', job)
#
#         if 'start_jobs' in event:
#             for jid in event['start_jobs']:
#                 job = jobs.submit_jobs[jid]
#                 jobs.pend_jobs(job)
#
#         # Sort pending queue
#         # We can start jobs!
#         issuing_jobs = []
#         for par, queue in jobs.pending_queue.items():
#             queue.sort(key=lambda jid: jobs.submit_jobs[jid]['submit_time'])
#             for jid in queue:
#                 job = jobs.submit_jobs[jid]
#                 if cluster.try_alloc_res(job, policy=policy):
#                     issuing_jobs.append(jid)
#                 elif not fit_first:
#                     break
#
#         for jid in issuing_jobs:
#             job = jobs.submit_jobs[jid]
#             jobs.issue_jobs(job, event_time)
#             jobs.add_event(job, job['end_time'], 'end_jobs')
#
#         jobs.PC += 1
#         print(f"time[{event_time}] ", end='')
#         cluster.report()
#         print(f"time[{event_time}] ", end='')
#         jobs.report()
#         if logger is not None:
#             event_log(logger, event_time, jobs)


def dlas_sim(cluster, jobs, logger=None, policy='first-fit', fit_first=True,
             gputime=True, promotion_knob=None, **kwargs):
    r"""
    Discretized Two-Dimensional Least Attained Service;
    :param cluster:
    :param jobs:
    :param logger:
    :param policy:
    :return:
    """
    num_q = 3
    queue_lim = {i: [3250, 7200, 18000] for i in jobs.pending_queue.keys()}
    # reformat the pending_queue into multi-level version
    jobs.init_multilevel_queue(num_q)

    while jobs.PC < len(jobs.events):
        if jobs.check_overload():
            # cluster out of usage
            print("This cluster is not large enough to run the job")
            break

        event = jobs.events[jobs.PC]
        event_time = event['time']
        jobs.PC += 1

        if 'preempt_jobs' in event:
            for jid in event['preempt_jobs']:
                job = jobs.submit_jobs[jid]
                qid = job['qid']
                if qid + 1 != num_q:  # demotion
                    job['qid'] = qid + 1
                cluster.release_job_res(job)
                jobs.pause_jobs(job, event_time)
                jobs.pend_jobs(job)
                # penalty for preemption: 8sec * #node
                job['running_time'] += 8  # * job['num_node']
                if promotion_knob is not None:
                    promotion_time = int(promotion_knob * job['executed_time']) + event_time
                    jobs.add_event(job, promotion_time, 'promotion_jobs')
                    job['need_promote'] = True

        if 'end_jobs' in event:
            for jid in event['end_jobs']:
                job = jobs.submit_jobs[jid]
                if cluster.release_job_res(job):
                    jobs.finish_jobs('COMPLETED', job)
                else:
                    jobs.finish_jobs('ERROR', job)

        if 'start_jobs' in event:
            for jid in event['start_jobs']:
                job = jobs.submit_jobs[jid]
                job['qid'] = 0
                jobs.pend_jobs(job)

        if 'promotion_jobs' in event:
            for jid in event['promotion_jobs']:
                job = jobs.submit_jobs[jid]
                if job['need_promote']:
                    if 'partition' in job:
                        jobs.pending_queue[job['partition']][job['qid']].remove(jid)
                    else:
                        jobs.pending_queue['all'][job['qid']].remove(jid)
                    job['qid'] = 0
                    jobs.pend_jobs(job)
                    job['need_promote'] = False

        # We can start jobs!
        issuing_jobs = []
        for par, queues in jobs.pending_queue.items():
            for queue in queues:
                for jid in queue:
                    job = jobs.submit_jobs[jid]
                    if cluster.try_alloc_res(job, policy=policy):
                        issuing_jobs.append(jid)
                        job['need_promote'] = False
                    elif not fit_first:
                        break

        for jid in issuing_jobs:
            job = jobs.submit_jobs[jid]
            jobs.issue_jobs(job, event_time)
            par = 'all' if 'partition' not in job else job['partition']
            if gputime:
                preempt_time = queue_lim[par][job['qid']]//job['num_gpu'] + event_time
            else:
                preempt_time = queue_lim[par][job['qid']] + event_time
            if preempt_time < job['end_time']:
                jobs.add_event(job, preempt_time, 'preempt_jobs')
            else:
                jobs.add_event(job, job['end_time'], 'end_jobs')

        print(f"time[{event_time}] ", end='')
        cluster.report()
        print(f"time[{event_time}] ", end='')
        jobs.report()
        if logger is not None:
            event_log(logger, event_time, jobs)

        # deformat the pending_queue
    jobs.release_multilevel_queue()
