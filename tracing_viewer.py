import json
"""
Trace Event Format:
https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
Goal Enable Chrome Tracing functionality of this simulator for the ease of profiling and debugging
"""


def generate_trace_json(jobs, file='tracing.json', max_jobs=None):
    job_trace = dict()
    job_trace['traceEvents'] = []
    cnt = 0
    interval = 1
    for jid, job in jobs.submit_jobs.items():
        if 'preempt_time' not in job:
            issue_entry = dict()
            issue_entry['pid'] = 1
            issue_entry['tid'] = jid
            issue_entry['ph'] = "X"
            issue_entry['ts'] = (job['submit_time']) * 1000
            issue_entry['dur'] = (job['pending_time'] + interval) * 1000
            issue_entry['name'] = "Waiting"
            issue_entry['args'] = {
                'num_node': job['num_node'],
                'num_gpu': job['num_gpu'],
                'job_name': job['model'],
            }
            run_entry = dict()
            run_entry['pid'] = 1
            run_entry['tid'] = jid
            run_entry['ph'] = "X"
            run_entry['ts'] = (job['start_time']+interval) * 1000
            run_entry['dur'] = (job['running_time']-interval) * 1000
            run_entry['name'] = "Running"
            run_entry['args'] = {
                'num_node': job['num_node'],
                'num_gpu': job['num_gpu'],
                'job_name': job['model'],
                'placements': str(job['placements']),
                's_t': job['start_time'],
                'e_t': job['end_time'],
            }
            job_trace['traceEvents'].append(issue_entry)
            job_trace['traceEvents'].append(run_entry)
        else:
            # start
            s_t = job['start_time_list'][0]
            entry = dict()
            entry['pid'] = 1
            entry['tid'] = jid
            entry['ph'] = "X"
            entry['ts'] = (job['submit_time']) * 1000
            entry['dur'] = (s_t - job['submit_time'] + interval) * 1000
            entry['name'] = "Waiting"
            entry['args'] = {
                'num_node': job['num_node'],
                'num_gpu': job['num_gpu'],
                'job_name': job['model'],
            }
            job_trace['traceEvents'].append(entry)
            for i in range(len(job['preempt_time'])):
                p_t = job['preempt_time'][i]
                entry = dict()
                entry['pid'] = 1
                entry['tid'] = jid
                entry['ph'] = "X"
                entry['ts'] = (s_t + interval) * 1000
                entry['dur'] = (p_t-s_t - interval) * 1000
                entry['name'] = "Running"
                entry['args'] = {
                    'num_node': job['num_node'],
                    'num_gpu': job['num_gpu'],
                    'job_name': job['model'],
                    'placements': str(job['placements_history'][i]),
                    's_t': s_t,
                    'e_t': p_t,
                }
                job_trace['traceEvents'].append(entry)
                s_t = job['start_time_list'][i+1]
                pending_time = s_t - p_t
                entry = dict()
                entry['pid'] = 1
                entry['tid'] = jid
                entry['ph'] = "X"
                entry['ts'] = p_t * 1000
                entry['dur'] = (pending_time + interval) * 1000
                entry['name'] = "Waiting"
                entry['args'] = {
                    'num_node': job['num_node'],
                    'num_gpu': job['num_gpu'],
                    'job_name': job['model'],
                }
                job_trace['traceEvents'].append(entry)

            entry = dict()
            entry['pid'] = 1
            entry['tid'] = jid
            entry['ph'] = "X"
            entry['ts'] = (s_t + interval) * 1000
            entry['dur'] = (job['end_time']-s_t) * 1000
            entry['name'] = "Running"
            entry['args'] = {
                'num_node': job['num_node'],
                'num_gpu': job['num_gpu'],
                'job_name': job['model'],
                'placements': str(job['placements']),
                's_t': s_t,
                'e_t': job['end_time']
            }
            job_trace['traceEvents'].append(entry)

        cnt += 1
        if max_jobs is not None and cnt == max_jobs:
            break

    job_trace['displayTimeUnit'] = "ms"
    with open(file, 'w') as json_file:
        json.dump(job_trace, json_file)
