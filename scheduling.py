
def fifo_sim(cluster, jobs):
    r"""
    FIFO scheduling baseline
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
            if cluster.try_alloc_res(job):
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

