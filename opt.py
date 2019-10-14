import argparse

parser = argparse.ArgumentParser(description='Deep Learning Cluster Monitor')

parser.add_argument('--mode', default='nmsl', type=str)
parser.add_argument('--cluster_info', default='cluster_info.csv', type=str)
parser.add_argument('--job_trace', default='Trace_Collector/jobs_Pose.csv', type=str)
parser.add_argument('--schedule', default='fifo', type=str)
# parser.add_argument()

opt = parser.parse_args()
