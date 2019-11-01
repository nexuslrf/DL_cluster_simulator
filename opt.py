import argparse

parser = argparse.ArgumentParser(description='Deep Learning Cluster Monitor')

parser.add_argument('--mode', default='nmsl', type=str)
parser.add_argument('--cluster_info', default='Cluster_Info/cluster_info.csv', type=str)
parser.add_argument('--job_trace', default='Trace_Collector/jobs.csv', type=str)
parser.add_argument('--schedule', default='fifo', type=str)
parser.add_argument('--partition', default='Cluster_Info/sinfo.csv', type=str)
parser.add_argument('--logger_file', default='cluster_log_fifo_best-fit.json', type=str)
parser.add_argument('--chrome_trace_file', default='tracing_fifo_best-fit.json', type=str)
parser.add_argument('--placement_policy', default='best-fit', type=str)
args = parser.parse_args()
# parser.add_argument()

opt = parser.parse_args()
