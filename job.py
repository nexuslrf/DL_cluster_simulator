import csv

"""
job class: to keep the necessary info about a submitted DL job.
"""


class JobEvents:
    def __init__(self, verbose=False):
        self.submit_jobs = []
        self.pending_jobs = []
        self.pending_queue = dict()
        self.running_jobs = []
        self.finished_jobs = []  # including both completed ones and failed ones
        self.end_jobs = []
        self.events = []
        self.PC = 0
        self.verbose = verbose

    def print_verbose(self, out_str):
        if self.verbose: print(out_str)

    def init_jobs_from_csv(self, file_path):
        r"""
        :param file_path: trace of completed jobs [CURRENT VERSION]
            file format: time unit: sec
                jid,num_gpu,submit_time,running_time,model,[partition(optional)]
                0,4,100,20000,CNN, [Pose]
        :return: list of sorted jobs
        """
        fh = open(file_path)
        reader = csv.DictReader(fh)
        for job in reader:
            job = {key: eval(val) if val.isdigit() else val for (key, val) in job.items()}
            job['state'] = "UNISSUED"
            job['num_gpu_p_node'] = (job['num_gpu'] - 1) // job['num_node'] + 1
            job['num_gpu'] = job['num_gpu_p_node'] * job['num_node']
            if 'partition' in job and job['partition'] not in self.pending_queue:
                self.pending_queue[job['partition']] = []
            self.submit_jobs.append(job)
        # self.submit_jobs.sort(key=lambda t: t['submit_time'])
        if len(self.pending_queue) == 0:
            self.pending_queue['all'] = []
        self.submit_jobs = {job['jid']: job for job in self.submit_jobs}

    def init_events_from_jobs(self):
        r"""
        event: time_stamp, start_jobs, end_jobs.
        :return:
        """
        event_dict = dict()
        for jid, job in self.submit_jobs.items():
            tmp_event = dict()
            if job['submit_time'] not in event_dict:
                tmp_event['time'] = job['submit_time']
                tmp_event['start_jobs'] = [jid, ]
                event_dict[job['submit_time']] = tmp_event
            else:
                event_dict[job['submit_time']]['start_jobs'].append(jid)

        self.events = [item for _, item in event_dict.items()]
        self.events.sort(key=lambda t: t['time'])

    def init_multilevel_queue(self, num_q):
        for k in self.pending_queue.keys():
            self.pending_queue[k] = [[] for i in range(num_q)]

    def release_multilevel_queue(self):
        for k in self.pending_queue.keys():
            self.pending_queue[k] = []

    def pend_jobs(self, job):
        if type(job) != dict:
            job = self.submit_jobs[job]
        job['state'] = 'PENDING'
        self.pending_jobs.append(job['jid'])
        if 'partition' in job:
            if 'qid' in job:
                self.pending_queue[job['partition']][job['qid']].append(job['jid'])
            else:
                self.pending_queue[job['partition']].append(job['jid'])
        else:
            if 'qid' in job:
                self.pending_queue['all'][job['qid']].append(job['jid'])
            else:
                self.pending_queue['all'].append(job['jid'])
        self.print_verbose('time[{}]\tjob[{}] PENDING'.format(job['submit_time'], job['jid']))

    def issue_jobs(self, job, issue_time):
        r"""
        transit jobs with PENDING state to RUNNING state
        :param job:
        :param issue_time:
        :return:
        """
        if type(job) != dict:
            job = self.submit_jobs[job]
        job['state'] = 'RUNNING'
        self.running_jobs.append(job['jid'])
        job['start_time'] = issue_time
        job['end_time'] = issue_time + job['running_time']
        job['pending_time'] = job['start_time'] - job['submit_time']
        self.pending_jobs.remove(job['jid'])
        if 'partition' in job:
            if 'qid' in job:
                self.pending_queue[job['partition']][job['qid']].remove(job['jid'])
            else:
                self.pending_queue[job['partition']].remove(job['jid'])
        else:
            if 'qid' in job:
                self.pending_queue['all'][job['qid']].remove(job['jid'])
            else:
                self.pending_queue['all'].remove(job['jid'])

        self.print_verbose('time[{}]\tjob[{}] RUNNING'.format(issue_time, job['jid']))

        # For preemption/MFQ
        if 'qid' in job:
            if 'start_time_list' not in job:
                job['start_time_list'] = []
                job['executed_time'] = 0
                job['total_pending_time'] = job['pending_time']
            else:
                job['executed_time'] += (job['preempt_time'][-1] - job['start_time_list'][-1])
                job['total_pending_time'] += (issue_time - job['preempt_time'][-1])
            job['start_time_list'].append(issue_time)
            job['end_time'] -= job['executed_time']
            job['pending_time'] = job['total_pending_time']

    def finish_jobs(self, state, job):
        if type(job) != dict:
            job = self.submit_jobs[job]
        job['state'] = state
        self.running_jobs.remove(job['jid'])
        self.finished_jobs.append(job['jid'])
        self.print_verbose("time[{}]\tjob[{}] {}".format(job['end_time'], job['jid'], state))

    def pause_jobs(self, job, event_time):
        if type(job) != dict:
            job = self.submit_jobs[job]
        self.running_jobs.remove(job['jid'])
        if 'preempt_time' not in job:
            job['preempt_time'] = []
            job['placements_history'] = []
        job['preempt_time'].append(event_time)
        job['placements_history'].append(job.pop('placements'))
        self.print_verbose("time[{}]\tjob[{}] Pause".format(event_time, job['jid']))

    def add_event(self, job, event_time, event_type='end'):
        insert = True
        index = len(self.events)
        for i, event in enumerate(self.events[self.PC:]):
            if event['time'] == event_time:
                if event_type not in event:
                    event[event_type] = []
                event[event_type].append(job['jid'])
                insert = False
                break
            elif event['time'] > event_time:
                index = i + self.PC
                break
        if insert:
            tmp_event = dict()
            tmp_event['time'] = event_time
            tmp_event[event_type] = [job['jid'], ]
            self.events.insert(index, tmp_event)

    def check_overload(self):
        if len(self.pending_jobs) > 0 and len(self.running_jobs) == 0:
            return True
        else:
            return False

    def report(self):
        print(f"PENDING: {len(self.pending_jobs)}\tRUNNING: {len(self.running_jobs)}")


if __name__ == '__main__':
    job_events = JobEvents()
    job_events.init_jobs_from_csv('Trace_Collector/jobs_Pose.csv')
    job_events.init_events_from_jobs()
    for j in job_events.events:
        print(j)
