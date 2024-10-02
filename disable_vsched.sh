#!/bin/bash

# Get the number of CPUs
num_cpus=$(nproc)

# Initialize an empty string
cpu_string=""

# Loop to create the string
for ((i=0; i<num_cpus; i++))
do
    cpu_string="${cpu_string}0;"
done

# Remove the trailing semicolon
cpu_string=${cpu_string%;}

generate_section() {
    local section=""
    for ((block=0; block<num_cpus; block++))
    do
        for ((cpu=0; cpu<num_cpus; cpu++))
        do
            if [ $block -eq $cpu ]; then
                section="${section}1"
            else
                section="${section}0"
            fi
        done
        # Add semicolon after every block, including the last one
        section="${section};"
    done
    echo "$section"
}


generate_last_section() {
    local section=""
    for ((block=0; block<num_cpus; block++))
    do
        for ((cpu=0; cpu<num_cpus; cpu++))
        do
        	section="${section}1"
        done
        # Add semicolon after every block, including the last one
        section="${section};"
    done
    echo "$section"
}



# Generate the full string with three sections
default_topology_string="$(generate_section):$(generate_section):$(generate_last_section):"

default_capacity_string=""
for ((i=0; i<num_cpus; i++))
do
    default_capacity_string = "${default_capacity_string}0;"
done

echo "$default_topology_string" > /proc/vtopology_write

if [ $? -eq 0 ]; then
    echo "Successfully wrote to /proc/vtopology_write"
else
    echo "Failed to write to /proc/vtopology_write"
fi

echo "$default_capacity_string" > /proc/vcapacity_write

if [ $? -eq 0 ]; then
    echo "Successfully wrote to /proc/vcapacity_write"
else
    echo "Failed to write to /proc/vcapacity_write"
fi

sudo rmmod preempt_proc.ko
sudo killall vcap
sudo killall vtop





