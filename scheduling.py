from cluster_vis import event_log


def fifo_sim(cluster, jobs, logger=None, policy='first-fit'):
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
        for jid in event['end_jobs']:
            job = jobs.submit_jobs[jid]
            if cluster.release_job_res(job):
                jobs.finish_jobs('COMPLETED', job)
            else:
                jobs.finish_jobs('ERROR', job)

        for jid in event['start_jobs']:
            job = jobs.submit_jobs[jid]
            jobs.pend_jobs(job)

        # We can start jobs!
        issuing_jobs = []
        for jid in jobs.pending_jobs:
            job = jobs.submit_jobs[jid]
            if cluster.try_alloc_res(job, policy=policy):
                issuing_jobs.append(jid)

        for jid in issuing_jobs:
            job = jobs.submit_jobs[jid]
            jobs.issue_jobs(job, event_time)
            jobs.add_end_events(job)

        jobs.PC += 1
        print(f"time[{event_time}] ", end='')
        cluster.report()
        print(f"time[{event_time}] ", end='')
        jobs.report()
        if logger is not None:
            event_log(logger, event_time, jobs)


def sjf_sim(cluster, jobs, logger=None, policy='first-fit'):
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
        for jid in event['end_jobs']:
            job = jobs.submit_jobs[jid]
            if cluster.release_job_res(job):
                jobs.finish_jobs('COMPLETED', job)
            else:
                jobs.finish_jobs('ERROR', job)

        for jid in event['start_jobs']:
            job = jobs.submit_jobs[jid]
            jobs.pend_jobs(job)

        # Sort pending queue
        jobs.pending_jobs.sort(key=lambda jid: jobs.submit_jobs[jid]['running_time'])
        # We can start jobs!
        issuing_jobs = []
        for jid in jobs.pending_jobs:
            job = jobs.submit_jobs[jid]
            if cluster.try_alloc_res(job, policy=policy):
                issuing_jobs.append(jid)

        for jid in issuing_jobs:
            job = jobs.submit_jobs[jid]
            jobs.issue_jobs(job, event_time)
            jobs.add_end_events(job)

        jobs.PC += 1
        print(f"time[{event_time}] ", end='')
        cluster.report()
        print(f"time[{event_time}] ", end='')
        jobs.report()
        if logger is not None:
            event_log(logger, event_time, jobs)


def lsf_sim(cluster, jobs, logger=None, policy='first-fit'):
    r"""
    Least service/resource first scheduling baseline
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
        for jid in event['end_jobs']:
            job = jobs.submit_jobs[jid]
            if cluster.release_job_res(job):
                jobs.finish_jobs('COMPLETED', job)
            else:
                jobs.finish_jobs('ERROR', job)

        for jid in event['start_jobs']:
            job = jobs.submit_jobs[jid]
            jobs.pend_jobs(job)

        # Sort pending queue
        jobs.pending_jobs.sort(key=lambda jid: jobs.submit_jobs[jid]['num_gpu'])
        # We can start jobs!
        issuing_jobs = []
        for jid in jobs.pending_jobs:
            job = jobs.submit_jobs[jid]
            if cluster.try_alloc_res(job, policy=policy):
                issuing_jobs.append(jid)

        for jid in issuing_jobs:
            job = jobs.submit_jobs[jid]
            jobs.issue_jobs(job, event_time)
            jobs.add_end_events(job)

        jobs.PC += 1
        print(f"time[{event_time}] ", end='')
        cluster.report()
        print(f"time[{event_time}] ", end='')
        jobs.report()
        if logger is not None:
            event_log(logger, event_time, jobs)
