# vTopology: Topology prober for vSched
![vSched](https://img.shields.io/badge/vSched-vTopology-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

**This component is part of the vSched project. The main repository for vSched is located at: https://github.com/vSched/vsched_main**

⚠️ **You must be running the vSched custom kernel with the vSched kernel module activated for this component to function**


## Overview


vTopology is a program used to determine the topology of vCPUs in a virtual machine. It measures cache line transfer latencies between vCPU pairs to infer their relationships, including SMT siblings, vCPUs on the same socket, and stacked vCPUs. 

