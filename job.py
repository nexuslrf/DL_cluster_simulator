import csv

"""
job class: to keep the necessary info about a submitted DL job.
"""


class JobEvents:
    def __init__(self):
        self.submit_jobs = []
        self.pending_jobs = []
        self.issuing_jobs = []
        self.end_jobs = []
        self.events = []
        self.PC = 0

    def init_jobs_from_csv(self, file_path):
        r"""
        :param file_path: trace of completed jobs [CURRENT VERSION]
            file format: time unit: sec
                jid,num_gpu,submit_time,running_time,model
                0,4,100,20000,CNN
        :return: list of sorted jobs
        """
        fh = open(file_path)
        reader = csv.DictReader(fh)
        for job in reader:
            job = {key: eval(val) if key != 'model' else val for (key, val) in job.items()}
            job['state'] = "UNISSUED"
            self.submit_jobs.append(job)
        self.submit_jobs.sort(key=lambda t: t['submit_time'])

    def init_events_from_jobs(self):
        r"""
        event: time_stamp, start_jobs, end_jobs.
        :return:
        """
        for i, job in enumerate(self.submit_jobs):
            tmp_event = dict()
            if len(self.events) == 0 or self.events[-1]['time']!=job['submit_time']:
                tmp_event['time'] = job['submit_time']
                tmp_event['start_jobs'] = [i, ]
                tmp_event['end_jobs'] = []
                self.events.append(tmp_event)
            else:
                self.events[-1]['start_time'].append(i)

    def pend_jobs(self, job):
        job['state'] = 'PENDING'
        self.pending_jobs.append(job)

    def issue_jobs(self, job, issue_time):
        r"""
        transit jobs with PENDING state to RUNNING state
        :param job:
        :param issue_time:
        :return:
        """
        job['state'] = 'RUNNING'
        job['start_time'] = issue_time
        job['end_time'] = issue_time + job['running_time']
        job['pending_time'] = job['start_time'] - job['submit_time']
        self.pending_jobs.remove(job)

    def add_end_events(self, job):
        end_time = job['end_time']
        insert = True
        index = len(self.events)
        for i, event in enumerate(self.events[self.PC:]):
            if event['time'] == end_time:
                event['end_jobs'].append(job)
                insert = False
                break
            elif event['time'] > end_time:
                index = i + self.PC
                break
        if insert:
            tmp_event = dict()
            tmp_event['time'] = end_time
            tmp_event['start_jobs'] = []
            tmp_event['end_jobs'] = [job, ]
            self.events.insert(index, tmp_event)


if __name__ == '__main__':
    job_events = JobEvents()
    job_events.init_jobs_from_csv('Trace_Collector/jobs_Pose.csv')
    job_events.init_events_from_jobs()
    for j in job_events.events:
        print(j)
