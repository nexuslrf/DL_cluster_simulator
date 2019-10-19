import json
"""
Trace Event Format:
https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
Goal Enable Chrome Tracing functionality of this simulator for the ease of profiling and debugging
"""


def generate_trace_json(jobs, file='tracing.json'):
    job_trace = dict()
    job_trace['traceEvents'] = []
    for jid, job in jobs.submit_jobs.items():
        issue_entry = dict()
        issue_entry['pid'] = 1
        issue_entry['tid'] = jid
        issue_entry['ph'] = "X"
        issue_entry['ts'] = (job['submit_time'] - 0.1) * 1000
        issue_entry['dur'] = (job['pending_time'] + 0.1) * 1000
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
        run_entry['ts'] = (job['start_time']) * 1000
        run_entry['dur'] = (job['running_time']) * 1000
        run_entry['name'] = "Running"
        run_entry['args'] = {
            'num_node': job['num_node'],
            'num_gpu': job['num_gpu'],
            'job_name': job['model'],
            'switch': str(job['placements']),
        }
        job_trace['traceEvents'].append(issue_entry)
        job_trace['traceEvents'].append(run_entry)
    job_trace['displayTimeUnit'] = "ms"
    with open(file, 'w') as json_file:
        json.dump(job_trace, json_file)
