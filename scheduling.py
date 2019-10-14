
def fifo_sim(cluster, jobs):
    r"""
    FIFO scheduling baseline
    :param cluster:
    :param jobs:
    :return:
    """
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

        jobs.pend_jobs()
        if cluster.free_gpus() > 0:
            # We can start jobs!
            new_start_list = []
            for job in jobs.pending_jobs:

