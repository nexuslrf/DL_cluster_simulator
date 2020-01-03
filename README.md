# GPU Cluster Simulation For DL Tasks

![gif](./demo_fifo.gif)

### Environment:

* Python3
* packages: `matplotlib`, `colorcet`, `networkx`

### Codes in the Repo:

* simulator.py: running the simulation; output: two *.json file containing scheduling information.
* cluster_vis.py: running the visualization of the simulation based on generated *.json files.
* opt.py: set arguments for the whole simulation.

### Dependant Data:

* Cluster_Info/cluster_info.csv: 
  * containing the information of each node in the cluster
* Trace_Collector/jobs.csv: 
  * containing the information of submitted jobs. 
  * Data are filtered by codes in ./Trace_Collector

### Run the Simulation:

After setting the opt.py (or directly input args in command line), you can

* run simulation:

  ``` 
  python simulator.py
  ```

* run visualization:

  ```
  python cluster_vis.py
  ```

  

