import argparse

parser = argparse.ArgumentParser(description='Deep Learning Cluster Monitor')
schedule = 'dlas'
policy = 'free-gpu'
penalty = '0.2'
parser.add_argument('--mode', default='', type=str)
parser.add_argument('--cluster_info', default='Cluster_Info/cluster_info.csv', type=str)
parser.add_argument('--job_trace', default='Trace_Collector/jobs.csv', type=str)
parser.add_argument('--schedule', default=schedule, type=str)
parser.add_argument('--partition', default='Cluster_Info/sinfo.csv', type=str)
parser.add_argument('--logger_file', default=f'cluster_log_{schedule}_{policy}{penalty}.json', type=str)
parser.add_argument('--chrome_trace_file', default=f'tracing_{schedule}_{policy}{penalty}.json', type=str)
parser.add_argument('--placement_policy', default=policy, type=str)
parser.add_argument('--fifo_queue', action='store_false')
parser.add_argument('--cputime', action='store_false')
parser.add_argument('--migration', action='store_true')
args = parser.parse_args()
# parser.add_argument()

opt = parser.parse_args()
