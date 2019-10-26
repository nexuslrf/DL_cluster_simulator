import csv
num_cpu_p_node = 4
"""
cluster class: to represent the cluster info
"""

"""
Suppose each node contains GPUs of the same type.
"""


def create_node_placement(job, switch_id, node_ids, num_gpus, num_cpus):
    if type(node_ids) != list:
        node_ids = [node_ids, ]
        num_cpus = [num_cpus, ]
        num_gpus = [num_gpus, ]
    sw_place = dict()
    sw_place['switch'] = switch_id
    sw_place['nodes'] = []
    for nid, num_gpu, num_cpu in zip(node_ids, num_gpus, num_cpus):
        node_place = dict()
        node_place['id'] = nid
        node_place['num_gpu'] = num_gpu
        node_place['num_cpu'] = num_cpu
        # node_place['mem']
        # node_place['tasks'] = list()
        # node_place['network']
        sw_place['nodes'].append(node_place)

    if 'placements' not in job:
        job['placements'] = []
    job['placements'].append(sw_place)


class Node:
    def __init__(self, idx=0, node_gpu=4, node_cpu=20, node_mem=128, gpu_type='1080Ti', name=None, **kwargs):
        self.id = idx
        self.num_gpu = node_gpu
        self.num_cpu = node_cpu
        self.mem = node_mem
        self.gpu_type = gpu_type
        self.free_cpus = self.num_cpu
        self.free_gpus = self.num_gpu
        self.free_mem = self.mem
        self.jobs = []
        self.partition = ''
        if name is None:
            self.name = f'{idx}'
        else:
            self.name = name

    def release_job_res(self, jid, node):
        if jid in self.jobs:
            self.jobs.remove(jid)
        # [X] self.release_network_load(node['network']) # if necessary
        if self.free_cpus + node['num_cpu'] > self.num_cpu:
            free_cpus = self.num_cpu
            cpu = False
        else:
            free_cpus = self.free_cpus + node['num_cpu']
            cpu = True

        if self.free_gpus + node['num_gpu'] > self.num_gpu:
            free_gpus = self.num_gpu
            gpu = False
        else:
            free_gpus = self.free_gpus + node['num_gpu']
            gpu = True
        # [X] self.free_mem = self.free_mem + node_dict['mem']
        ret_cpu = free_cpus - self.free_cpus
        ret_gpu = free_gpus - self.free_gpus
        self.free_cpus = free_cpus
        self.free_gpus = free_gpus
        return gpu and cpu, ret_cpu, ret_gpu


class Switch:
    def __init__(self, idx=0, node_list=None, name=None, **kwargs):
        if node_list is None:
            node_list = []
        self.id = idx
        if name is None:
            self.name = f'Switch{idx}'
        else:
            self.name = name
        self.node_list = node_list
        self.num_node = len(node_list)
        self.num_gpu = sum(node.num_gpu for node in node_list)
        self.num_cpu = sum(node.num_cpu for node in node_list)
        if self.num_node == 0:
            if 'switch_nodes' in kwargs:
                self.num_node = kwargs['switch_nodes']
                self.node_list = [Node(idx=i, **kwargs) for i in range(self.num_node)]
                self.num_gpu = sum(node.num_gpu for node in self.node_list)
                self.num_cpu = sum(node.num_cpu for node in self.node_list)
        self.free_cpus = self.num_cpu
        self.free_gpus = self.num_gpu

    def add(self, **kwargs):
        newNode = Node(idx=self.num_node, **kwargs)  # idx=kwargs['node_id']
        self.num_node += 1
        self.num_gpu += newNode.num_gpu
        self.num_cpu += newNode.num_cpu
        self.node_list.append(newNode)
        self.free_gpus += newNode.num_gpu
        self.free_cpus += newNode.num_cpu

    def release_job_res(self, jid, nodes):
        ret_cpu = 0
        ret_gpu = 0
        for node in nodes:
            if ('id' not in node) or ('num_gpu' not in node) or ('num_cpu' not in node):
                return False, ret_cpu, ret_gpu
            done, cpus, gpus = self.node_list[node['id']].release_job_res(jid, node)
            ret_gpu += gpus
            ret_cpu += cpus
            self.free_cpus += cpus
            self.free_gpus += gpus
            if not done:
                return False, ret_cpu, ret_gpu
        return True, ret_cpu, ret_gpu

    def slurm_get_res(self, job, nlist=None):
        r"""
        alloc res from a single switch
        :param nlist:
        :param job:
        :return:
        """
        # @TODO consider ps-worker network load!
        need_node = job['num_node']
        if need_node == 1:  # non-distributed
            return self.try_single_node_alloc(job, nlist)
        elif need_node > 1:
            return self.try_cross_node_alloc(job, nlist)
        else:
            print("Invalid node number for job[{}]".format(job['jid']))
            return False, list()  # @TODO

    def try_cross_node_alloc(self, job, nlist=None):
        need_gpu = job['num_gpu']
        need_node = job['num_node']
        need_gpu_p_node = job['num_gpu_p_node']
        # @NOTE What if need_gpu_p_node > max_gpu_num: -> current solution: just pend it...

        possible_nodes = []
        node_cnt = 0
        # @TODO finer grained placement policy?
        for node in self.node_list:
            if nlist is not None and node.id not in nlist:
                continue
            if node.free_gpus >= need_gpu_p_node and node.free_cpus >= num_cpu_p_node:
                possible_nodes.append(node.id)
                node_cnt += 1
                if node_cnt == need_node:
                    break

        if node_cnt == need_node:
            return True, possible_nodes
        else:
            return False, possible_nodes

    def try_single_node_alloc(self, job, nlist=None):
        need_gpu = job['num_gpu']
        # @NOTE each job is assigned 4 cores per node
        # @TODO may perform like Tiresias for num of CPU
        need_cpu = 4
        if self.free_gpus < need_gpu or self.free_cpus < need_cpu:
            return False, list()
        # @TODO finer grained placement policy?
        for node in self.node_list:
            if nlist is not None and node.id not in nlist:
                continue
            if node.free_gpus >= need_gpu and node.free_cpus >= need_cpu:
                return True, [node.id]
        return False, list()

    def slurm_alloc_res(self, job, nodes):
        num_gpu_p_node = job['num_gpu_p_node']
        num_node = job['num_node']
        for nid in nodes:
            node = self.node_list[nid]
            node.free_gpus -= num_gpu_p_node
            node.free_cpus -= num_cpu_p_node
            node.jobs.append(job['jid'])
            self.free_gpus -= num_gpu_p_node
            self.free_cpus -= num_cpu_p_node

        create_node_placement(job, self.id, nodes,
                              [num_gpu_p_node]*num_node,
                              [num_cpu_p_node]*num_node)

"""
mata data of cluster: Cluster -> Switch -> Node -> GPU
"""


class Cluster:
    def __init__(self, switch_list=None):
        if switch_list is None:
            switch_list = []
        self.switch_list = switch_list
        self.num_switch = len(switch_list)
        self.num_gpu = sum(switch.num_gpu for switch in switch_list)
        self.num_cpu = sum(switch.num_cpu for switch in switch_list)
        self.free_gpus = self.num_gpu
        self.free_cpus = self.num_cpu
        self.use_partition = False
        self.partitions = None

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
            To use partition function, use Partition class specified input file
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
                self.switch_list = [Switch(idx=i, **row) for i in range(self.num_switch)]
        elif len(keys) == 6:  # heterogeneous case
            switch_dict = dict()
            for row in reader:
                row = {key: eval(val) if val.isdigit() else val for (key, val) in row.items()}
                if row['switch_id'] not in switch_dict:
                    switch_dict[row['switch_id']] = Switch(name=row['switch_id'])
                switch_dict[row['switch_id']].add(name='{}'.format(row['node_id']), **row)
            cnt = 0
            for key, val in switch_dict.items():
                val.id = cnt
                self.switch_list.append(val)
                cnt += 1
            self.num_switch = len(switch_dict)
        else:
            # oops!
            print("Invalid file format!")
            exit()
        self.num_gpu = sum(switch.num_gpu for switch in self.switch_list)
        self.num_cpu = sum(switch.num_cpu for switch in self.switch_list)
        self.free_gpus = self.num_gpu
        self.free_cpus = self.num_cpu

    # @TODO partition based alloc!
    def try_alloc_res(self, job, policy='slurm'):
        r"""
        placements:
        list of switches: -> list of nodes
        [{'switch':xx, 'node': [{'id':xx, 'num_gpu':xx, 'num_cpu':xx, 'network':xxx, 'tasks': [w0, w1, ps1]}]}]
        :param policy:
        :param job:
        :return:
        """
        if policy == 'slurm':
            ret = self.slurm_placement(job)
        else:
            ret = self.slurm_placement(job)
        return ret

    def slurm_placement(self, job):
        r"""
        slurm: 1. try to alloc res from one switch, intra-switch
               2. see the possibilities of inter-switch alloc
        :param job:
        :return:
        """
        sw_nodes = []
        possible_sw = []
        nodes_cnt = 0
        nlist = None
        need_node = job['num_node']

        for switch in self.switch_list:
            if self.use_partition:
                if switch.id not in self.partitions[job['partition']]:
                    continue
                else:
                    nlist = self.partitions[job['partition']][switch.id]
            done, nodes = switch.slurm_get_res(job, nlist)
            if done:
                self.free_gpus -= job['num_gpu']
                self.free_cpus -= job['num_node'] * num_cpu_p_node
                # real alloc res
                switch.slurm_alloc_res(job, nodes)
                return True
            elif len(nodes) > 0:
                sw_nodes.append(nodes)
                possible_sw.append(switch.id)
                nodes_cnt += len(nodes)
                if nodes_cnt >= need_node:
                    for sw, nlist in zip(possible_sw, sw_nodes):
                        need_node -= len(nlist)
                        nlist = nlist[:need_node] if need_node < 0 else nlist
                        self.switch_list[sw].slurm_alloc_res(job, nlist)
                    self.free_gpus -= job['num_gpu']
                    self.free_cpus -= job['num_node'] * num_cpu_p_node
                    return True
        return False

    def release_job_res(self, job):
        for placement in job['placements']:
            if ('switch' not in placement) or ('nodes' not in placement):
                return False

            done, ret_cpu, ret_gpu = self.switch_list[placement['switch']].release_job_res(job['jid'],
                                                                                           placement['nodes'])
            # maintain metadata for cluster
            self.free_gpus += ret_gpu  # #gpus that are really idle!
            self.free_cpus += ret_cpu
            if not done:
                return False
        return True

    def add(self, **kwargs):
        pass

    def generate_cluster_file(self):
        pass

    def __str__(self):
        out_str = 'Cluster Overview:\n' + 8 * '-' + '\n'
        out_str += f'Cluster:\t#GPU: {self.free_gpus}/{self.num_gpu}\t#CPU: {self.free_cpus}/{self.num_cpu}\n'
        out_str += "/" + 10 * '-' + '\n'
        for s in range(self.num_switch):
            switch = self.switch_list[s]
            out_str += f"|- Switch[{switch.id}]:\t{switch.name}\t#GPU: {switch.free_gpus}/{switch.num_gpu}\t" \
                       f"#CPU: {switch.free_cpus}{switch.num_cpu}\n"
            out_str += "|  /" + 7 * '-' + '\n'
            for n in range(self.switch_list[s].num_node):
                node = self.switch_list[s].node_list[n]
                out_str += f"|  |- Node[{node.id}]:\t{node.name}\t" \
                           f"#GPU: {node.free_gpus}/{node.num_gpu}\t" \
                           f"#CPU: {node.free_cpus}/{node.num_cpu}\tGPU Type: {node.gpu_type}\n"
            out_str += "|  \\" + 7 * '-' + '\n'
        out_str += "\\" + 10 * '-'
        return out_str

    def report(self):
        out_str = 'Cluster Report:'
        out_str += f'Cluster:\t#GPU: {self.free_gpus}/{self.num_gpu}\t#CPU: {self.free_cpus}/{self.num_cpu}'
        print(out_str)


class Partition:
    r"""
    sub-graph of cluster
    """
    def __init__(self, cluster, file='sinfo.csv'):
        r"""
        partitions -> switches dict -> list of node id
        """
        self.partition_name = []
        self.partitions = dict()
        # node_name_id -> node_name: (switch_id, node_id)
        self.node_name_id = dict()
        self.cluster = cluster
        cluster.use_partition = True
        self.get_id_name_map(cluster)
        # init_from_csv
        fh = open(file)
        reader = csv.DictReader(fh)
        partitions = dict()
        for line in reader:
            name = line['PARTITION']
            node_name = line['NODELIST'].split('-')[-1]
            if name not in partitions:
                partitions[name] = list()
            partitions[name].append(node_name)
        for k, v in partitions.items():
            v.sort(key=lambda x: eval(x))
        self.partition_name = partitions.keys()

        for par, nlist in partitions.items():
            self.partitions[par] = dict()
            for n_name in nlist:
                if n_name in self.node_name_id:
                    sw_id, n_id = self.node_name_id[n_name]
                    if sw_id not in self.partitions[par]:
                        self.partitions[par][sw_id] = []
                    self.partitions[par][sw_id].append(n_id)
                    cluster.switch_list[sw_id].node_list[n_id].partition = par
        cluster.partitions = self.partitions    # my ugly programming! :-(

    def get_id_name_map(self, default_par='Others'):
        for switch in self.cluster.switch_list:
            for node in switch.node_list:
                self.node_name_id[node.name] = (switch.id, node.id)
                node.partition = default_par

    def __str__(self):
        out_str = 'Partition Overview:\n' + 8 * '-' + '\n'
        for par, sw_dict in self.partitions.items():
            out_str += f"Partition: {par}\n" + '/' + 9 * '-' + '\n'
            for sw_id, nlist in sw_dict.items():
                switch = self.cluster.switch_list[sw_id]
                out_str += f"|- Switch[{switch.id}]:\t{switch.name}\t#GPU: {switch.free_gpus}/{switch.num_gpu}\t" \
                        f"#CPU: {switch.free_cpus}{switch.num_cpu}\n"
                # out_str += "|  /" + 7 * '-' + '\n'
                for n_id in nlist:
                    node = switch.node_list[n_id]
                    out_str += f"|  |- Node[{node.id}]:\t{node.name}\t" \
                               f"#GPU: {node.free_gpus}/{node.num_gpu}\t" \
                               f"#CPU: {node.free_cpus}/{node.num_cpu}\tGPU Type: {node.gpu_type}\n"
                out_str += "|  \\" + 6 * '-' + '\n'
            out_str += "\\" + 9 * '-' + '\n'
        return out_str


def main():
    cluster = Cluster()
    cluster.init_from_csv('Cluster_Info/cluster_info.csv')
    print(cluster)
    partition = Partition(cluster, 'Cluster_Info/sinfo.csv')
    print(partition)

if __name__ == '__main__':
    main()
