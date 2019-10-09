"""
job class: to keep the necessary info about a submitted DL job.
"""


class Job:
    def __init__(self, jid, submit_time=0, running_time=0, num_gpu=0, model='Unknown', **kwargs):
        self.jid = jid
        self.submit_time = submit_time
        self.running_time = running_time
        self.num_gpu = num_gpu
        self.model = model


def init_jobs_from_csv(file_path):
    r"""
    :param file_path: trace of completed jobs [CURRENT VERSION]
        file format: time unit: sec
            jid,num_gpu,submit_time,running_time,model
            0,4,100,20000,CNN
    :return: list of sorted jobs
    """
    fh = open(file_path)

    return list

