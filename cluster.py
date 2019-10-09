import csv
"""
cluster class: to represent the cluster info
"""

"""
Suppose each node contains GPUs of the same type.
"""


class Node:
    def __init__(self, node_gpu=4, node_cpu=20, node_mem=128, gpu_type='1080Ti', **kwargs):
        self.num_gpu = node_gpu
        self.num_cpu = node_cpu
        self.mem = node_mem
        self.gpu_type = gpu_type


class Switch:
    def __init__(self, node_list=[], **kwargs):
        self.node_list = node_list
        self.num_node = len(node_list)
        self.num_gpu = sum(node.num_gpu for node in node_list)
        self.num_cpu = sum(node.num_cpu for node in node_list)
        if self.num_node == 0:
            if 'switch_nodes' in kwargs:
                self.num_node = kwargs['switch_nodes']
            else:
                return
            self.node_list = [Node(**kwargs) for i in range(self.num_node)]
            self.num_gpu = sum(node.num_gpu for node in self.node_list)
            self.num_cpu = sum(node.num_cpu for node in self.node_list)

    def add(self, **kwargs):
        newNode = Node(**kwargs)
        self.num_gpu += newNode.num_gpu
        self.num_cpu += newNode.num_cpu
        self.node_list.append(newNode)



"""
mata data of cluster: Cluster -> Switch -> Node -> GPU
"""


class Cluster:
    def __init__(self, switch_list=[]):
        self.switch_list = switch_list
        self.num_switch = len(switch_list)
        self.num_gpu = sum(switch.num_gpu for switch in switch_list)
        self.num_cpu = sum(switch.num_cpu for switch in switch_list)

    def init_from_csv(self, file_path):
        r"""
        :param file_path: file format issue: can be homogeneous and heterogeneous
            For homogeneous format: only 2 lines
                num_switch, switch_nodes, node_gpu, node_cpu, node_mem
                4,32,8,128,256
                --------------
                switch_nodes: #nodes of switch; node_gpu/cpu/mem: #gpu/cpu/mem per node
            For heterogeneous format: rows of nodes
                node_id, node_gpu, node_cpu, node_mem, gpu_type, switch_id
                0, 6, 4, 20, 128, 1080Ti, 0
        :return:
        """
        fh = open(file_path)
        reader = csv.DictReader(fh)
        keys = reader.fieldnames
        self.switch_list = []
        if len(keys) == 5:  # homogeneous case
            for row in reader:
                row = {key: eval(val) if key != 'gpu_type' else val for (key, val) in row.items()}
                self.num_switch = row['num_switch']
                self.switch_list = [Switch(**row) for i in range(self.num_switch)]
        elif len(keys) == 6:  # heterogeneous case
            switch_dict = dict()
            for row in reader:
                row = {key: eval(val) if key != 'gpu_type' else val for (key, val) in row.items()}
                if row['switch_id'] not in switch_dict:
                    switch_dict[row['switch_id']] = Switch()
                switch_dict[row['switch_id']].add(**row)
            for key, val in switch_dict.items():
                self.switch_list.append(val)
            self.num_switch = len(switch_dict)
        else:
            # oops!
            print("Invalid file format!")
            exit()
        self.num_gpu = sum(switch.num_gpu for switch in self.switch_list)
        self.num_cpu = sum(switch.num_cpu for switch in self.switch_list)

    def add(self, **kwargs):
        pass

    def generate_cluster_file(self):
        pass

    def __str__(self):
        out_str = 'Cluster Overview:\n'+8*'-'+'\n'
        out_str += f'Cluster:\t#GPU: {self.num_gpu}\t#CPU: {self.num_cpu}\n'
        out_str += "/" + 10 * '-' + '\n'
        for s in range(self.num_switch):
            switch = self.switch_list[s]
            out_str += f"|- Switch{s}:\t#GPU: {switch.num_gpu}\t#CPU: {switch.num_cpu}\n"
            out_str += "|  /" + 7*'-' + '\n'
            for n in range(self.switch_list[s].num_node):
                node = self.switch_list[s].node_list[n]
                out_str += f"|  |- Node{n}:\t#GPU: {node.num_gpu}\t#CPU: {node.num_cpu}\tGPU Type: {node.gpu_type}\n"
            out_str += "|  \\" + 7 * '-' + '\n'
        out_str += "\\" + 10 * '-'
        return out_str


def main():
    cluster = Cluster()
    cluster.init_from_csv('cluster_info.csv')
    print(cluster)


if __name__ == '__main__':
    main()
