import argparse

parser = argparse.ArgumentParser(description='Deep Learning Cluster Monitor')

parser.add_argument('--mode', default='nmsl', type=str)

opt = parser.parse_args()
