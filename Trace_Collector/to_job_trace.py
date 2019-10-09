import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tqdm
import argparse

parser = argparse.ArgumentParser(description='filtering task trace')
parser.add_argument('--task_file', default='tasks.csv', type=str)
parser.add_argument('--out_file', default='jobs.csv', type=str)
parser.add_argument('--partition', default=5, type=int)
parser.add_argument('--time_L_bnd', default='2019-09-01 00:00', type=str)
parser.add_argument('--time_R_bnd', default='2019-09-10 00:00', type=str)
args = parser.parse_args()


def omit_minutes(series):
    try:
        return pd.to_datetime(series.dt.strftime("%Y-%m-%d %H:00"), format="%Y-%m-%d %H:00")
    except:
        tmp = series.dt.strftime('%Y-%m-%d %H:00')
        return series.map(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:00'))


taskinfo_col = [
    'job_id', 'job_name', 'partition_name', 'node_num', 'res_gpu',
    'user', 'priority', 'state', 'submit', 'start', 'end', 'gpu_num',
    'pending_time', 'running_time'
]

task_file = args.task_file
taskinfo = pd.read_csv(task_file, names=taskinfo_col, index_col=0,
                       parse_dates=['submit', 'start', 'end'])

"""
-------------------------Set filtering requirements--------------------------------
"""
# drop outlandish tasks
taskinfo = taskinfo.dropna()
# rm job without using GPUs
taskinfo = taskinfo[taskinfo['gpu_num'] > 0]
# only choose job with COMPLETED state
taskinfo = taskinfo[taskinfo['state'] != 'COMPLETED']
# time range to select
taskinfo = taskinfo[taskinfo['submit'] >= pd.to_datetime(args.time_L_bnd)]
taskinfo = taskinfo[taskinfo['submit'] <= pd.to_datetime(args.time_R_bnd)]
"""
-----------------------------------------------------------------------------------
"""
taskinfo.index = taskinfo.index.astype(int)
taskinfo['running_time'] = taskinfo['running_time'].astype(int)
taskinfo['gpu_num'] = taskinfo['gpu_num'].astype(int)
# select partition
parti_name = taskinfo['partition_name'].astype('category').cat.categories.tolist()
print(parti_name)
for par in parti_name:
    jobs = taskinfo[taskinfo['partition_name'] == par]
    print(f"There are {len(jobs)} "
          f"valid submitted jobs, partition name: {par}")
    jobs['r_submit'] = (jobs['submit'] - pd.to_datetime(args.time_L_bnd)).dt.seconds
    jobs.to_csv(f'jobs_{par}.csv',
                columns=['gpu_num', 'r_submit', 'running_time', 'job_name'],
                index_label='jid',
                header=['num_gpu','submit_time', 'running_time', 'model'])




