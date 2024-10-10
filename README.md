# vCapacity: Capacity prober for vSched
![vSched](https://img.shields.io/badge/vSched-vCapacity-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

**This component is part of the vSched project. The main repository for vSched is located at: https://github.com/vSched/vsched_main**

⚠️ **You must be running the vSched custom kernel with the vSched kernel module activated for this component to function**


## Overview

vCapacity uses a cooperative and multi-phase sampling approach to measure the dynamic capacity of vCPUs in a virtual machine. It provides accurate capacity information without requiring hypervisor modifications, enabling better scheduling decisions in vSched.
