sudo echo "+cpu+cpuset" | sudo tee /sys/fs/cgroup/cgroup.subtree_control;
#sudo echo "+cpu" | sudo tee /sys/fs/cgroup/cgroup.subtree_control;


if [ ! -d /sys/fs/cgroup/hi_prgroup ]; then
    sudo mkdir /sys/fs/cgroup/hi_prgroup
fi


if [ ! -d /sys/fs/cgroup/lw_prgroup ]; then
    sudo mkdir /sys/fs/cgroup/lw_prgroup
fi


sudo echo "threaded" > /sys/fs/cgroup/lw_prgroup/cgroup.type;
sudo echo "threaded" > /sys/fs/cgroup/hi_prgroup/cgroup.type;
sudo echo 1 | sudo tee /sys/fs/cgroup/lw_prgroup/cpu.idle
sudo echo -20 | sudo tee /sys/fs/cgroup/hi_prgroup/cpu.weight.nice
