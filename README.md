# vCapacity: Capacity prober for vSched
![vSched](https://img.shields.io/badge/vSched-vCapacity-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

**This component is part of the vSched project. The main repository for vSched is located at: https://github.com/vSched/vsched_main**

⚠️ **You must be running the vSched custom kernel with the vSched kernel module activated for this component to function**


## Overview

vCapacity uses a cooperative and multi-phase sampling approach to measure the dynamic capacity of vCPUs in a virtual machine. It provides accurate capacity information without requiring hypervisor modifications, enabling better scheduling decisions in vSched.




## Parameters and Configuration

It is reccomended that you use vCap's default options.

### Auto-Configuration

vCapacity can automatically increase and decrease profiling time based on vCPU activity, 
enabling greater performance. Essentially, each vCPU needs to be some minimum level of active, and
there needs to be at least one maximally active vCPU, and if these conditions are met, an accurate
view of capacity is likely met.

This can be disabled and reenabled with the -o flag


#### Details

| Flag  | Description | Default Value
| ------------- | ------------- | ------------- |
| -v  | Verbose  | False  |
| -p  | Profiling time (ms)  | 100  |
| -d  | Decay Length (Number of runs before EMA discounts by 50%)  | 2 |
| -c  | Context Window (how many runs have data stored) | 5 |
| -s  | Sleep time (ms)  | 1000  |
| -i  |  Heavy Profile Interval | 5 |
| -o  |  Enable Optimizations | True |



