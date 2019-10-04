"""
cluster class: to represent the cluster info
"""


"""
Suppose each node contains GPUs of the same type.
"""


class Node:
    def __init__(self, num_gpu=0, num_cpu=0, mem=0, gpu_type='1080Ti'):
        self.num_gpu = num_gpu
        self.num_cpu = num_cpu
        self.mem = mem
        self.gpu_type = gpu_type


class Switch:
    def __init__(self, node_list=[]):
        self.node_list = node_list
        self.num_node = len(node_list)
        self.num_gpu = sum(node.num_gpu for node in node_list)
        self.num_cpu = sum(node.num_cpu for node in node_list)


"""
mata data of cluster: Cluster -> Switch -> Node -> GPU
"""


class Cluster:
    def __init__(self, switch_list):
        self.switch_list = switch_list
        self.num_switch = len(switch_list)
        self.num_gpu = sum(switch.num_gpu for switch in switch_list)
        self.num_cpu = sum(switch.num_cpu for switch in switch_list)
