#include <iostream>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <sys/resource.h>
#include <vector>
#include <sys/types.h>
#include <sys/syscall.h>
#include <cstdlib>
#include <sys/vfs.h>
#include <cmath>
#include <deque>
#include <numeric>
using namespace std::chrono;
typedef uint64_t u64;

//initialize global variables
int num_threads = sysconf(_SC_NPROCESSORS_ONLN);
int sleep_length = 1000;
int profile_time = 100;
int decay_length = 2;
//this is for saving how many profiling periods have gone by, so far. Offset by 1 because we want capacity measurement to start on next cycle
int profiler_iter = -1;
//this decides how many regular profile intervals go by before a "heavy" profile happens, where we try to get the actual capacity of the core
int heavy_profile_interval = 30;
int context_window = 5;
double milliseconds_totick_factor = static_cast<double>(sysconf(_SC_CLK_TCK))/1000.0;
bool verbose = false;
bool awake_workers_flag = false;
int initialized = 0;
std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::_V2::system_clock::duration> endtime;
void* run_computation(void * arg);
pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
pthread_cond_t cv1 = PTHREAD_COND_INITIALIZER;
pthread_mutex_t ready_check = PTHREAD_MUTEX_INITIALIZER;
std::vector<int> vtop_banned;
int ready_counter = 0;
int banned_amount =0;
//Arguments for each thread
struct thread_args {
  int id;
  int tid = -1;
  pthread_mutex_t mutex;
  u64 *addition_calc;
  double user_time;
};


int stick_this_thread_to_core(int core_id) {
   int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
   if (core_id < 0 || core_id >= num_cores)
      return EINVAL;

   cpu_set_t cpuset;
   CPU_ZERO(&cpuset);
   CPU_SET(core_id, &cpuset);

   pthread_t current_thread = pthread_self();    
   return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

struct raw_data {
  u64 steal_time;
  u64 preempts;
  u64 raw_compute;
  u64 use_time;
  u64 max_latency;
};



struct profiled_data{
  double capacity_perc_stddev;
  double capacity_adj_stddev;
  double latency_stddev;
  double preempts_stddev;

  double capacity_perc_ema;
  double latency_ema;
  double preempts_ema;

  double capacity_perc_ema_a;
  double capacity_adj_ema_a;
  double latency_ema_a;
  double preempts_ema_a;

  std::deque<double> capacity_perc_hist;
  std::deque<double> capacity_adj_hist;
  std::deque<double> latency_hist;
  std::deque<double> preempts_hist;

  double preempts;
  double capacity_perc;
  double capacity_adj;
  double latency;
  double max_latency;
};



double calculateStdDev(const std::deque<double>& v) {
    if (v.size() == 0) {
        return 0.0;
    }

    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double stdDev = std::sqrt(sq_sum / v.size() - mean * mean);

    return stdDev;
}


std::string_view get_option(
    const std::vector<std::string_view>& args, 
    const std::string_view& option_name) {
    for (auto it = args.begin(), end = args.end(); it != end; ++it) {
        if (*it == option_name)
            if (it + 1 != end)
                return *(it + 1);
    }
    
    return "";
};


bool has_option(
    const std::vector<std::string_view>& args, 
    const std::string_view& option_name) {
    for (auto it = args.begin(), end = args.end(); it != end; ++it) {
        if (*it == option_name)
            return true;
    }
    
    return false;
};



void moveThreadtoLowPrio(pid_t tid) {

    std::string path = "/sys/fs/cgroup/lw_prgroup/cgroup.threads";
    std::ofstream ofs(path, std::ios_base::app);
    if (!ofs) {
        std::cerr << "Could not open the file\n";
        return;
    }
    ofs << tid << "\n";
    ofs.close();
    struct sched_param params;
    params.sched_priority = sched_get_priority_min(SCHED_IDLE);
    sched_setscheduler(tid,SCHED_IDLE,&params);
}

void moveThreadtoHighPrio(pid_t tid) {

    std::string path = "/sys/fs/cgroup/hi_prgroup/cgroup.threads";
    std::ofstream ofs(path, std::ios_base::app);
    if (!ofs) {
        std::cerr << "Could not open the file\n";
        return;
    }
    ofs << tid << "\n";
    ofs.close();
}


void moveCurrentThread() {
    pid_t tid;
    tid = syscall(SYS_gettid);
    std::string path = "/sys/fs/cgroup/cgroup.procs";
    std::ofstream ofs(path, std::ios_base::app);
    if (!ofs) {
        std::cerr << "Could not open the file\n";
        return;
    }
    ofs << tid << "\n";
    ofs.close();
    struct sched_param params;
    params.sched_priority = sched_get_priority_max(SCHED_RR);
    sched_setscheduler(tid,SCHED_RR,&params);
}


void get_cpu_information(int cpunum,std::vector<raw_data>& data_arr,std::vector<thread_args*> thread_arg){
  std::ifstream f("/proc/preempts");
  std::string s;
  u64 preempts;
  u64 steals;
  u64 max_latency;
  for (int i = 0; i < cpunum; i++) {
    std::getline(f,s);
    std::getline(f,s);
    data_arr[i].preempts = std::stoull(s);
    std::getline(f,s);
    data_arr[i].steal_time = std::stoull(s);

  }
  std::ifstream laten("/proc/max_latency");
  for (int i = 0; i < cpunum; i++) {
    std::getline(laten,s);
    std::getline(laten,s);
    data_arr[i].max_latency = std::stoull(s);
  }
  std::ifstream stat_file("/proc/stat");
  std::string line;
  
  // Skip the first line (total CPU stats)
  std::getline(stat_file, line);
	for (int i = 0; i < cpunum && std::getline(stat_file, line); i++) {
    std::istringstream iss(line);
    std::string cpu;
    u64 user_time, nice_time, system_time;
    if (!(iss >> cpu >> user_time >> nice_time >> system_time)) {
        std::cerr << "Error reading CPU data for CPU " << i << std::endl;
        continue;
    }
    if (cpu == "cpu" + std::to_string(i)) {
        data_arr[i].use_time = user_time + nice_time + system_time;
    } else {
        std::cerr << "Unexpected CPU identifier: " << cpu << " for index " << i << std::endl;
    }
}
// Check if we've read data for all CPUs
if (stat_file.eof() && cpunum > 0) {
    std::cerr << "Warning: Reached end of file before reading all CPU data" << std::endl;
}
  

}

void reset_max_latency(){
  std::fstream write_file;
  write_file.open("/proc/max_latency", std::ios::out);
  write_file<<"0";
  write_file.close();
}

double calculate_ema(int decay_len, double& ema_help, double prev_ema,double new_value) {
  double decay_factor = pow(0.5,(1/(double)decay_len));
  double newA = (1+decay_factor*ema_help);
  double result = (new_value + ((prev_ema)*ema_help*decay_factor))/newA;
  ema_help = newA;
  return result;
}


//helper function to set context window to be short
void addToHistory(std::deque<double>& history_list,double item){
  if(history_list.size() > context_window) {
    history_list.pop_front();
  }
  history_list.push_back(item);
}

void setArguments(const std::vector<std::string_view>& arguments) {
    verbose = has_option(arguments, "-v");
    auto set_option_value = [&](const std::string_view& option, int& target) {
        if (auto value = get_option(arguments, option); !value.empty()) {
            try {
                target = std::stoi(std::string(value));
            } catch(const std::invalid_argument&) {
                throw std::invalid_argument(std::string("Invalid argument for option ") + std::string(option));
            } catch(const std::out_of_range&) {
                throw std::out_of_range(std::string("Out of range argument for option ") + std::string(option));
            }
        }
    };
    set_option_value("-s", sleep_length);
    set_option_value("-p", profile_time);
    set_option_value("-d", decay_length);
    set_option_value("-c", context_window);
    set_option_value("-i", heavy_profile_interval);
    num_threads = sysconf( _SC_NPROCESSORS_ONLN );
}

void process_values(std::vector<profiled_data>& data) {
    // Step 1: Normalize the capacity_adj values to have an average of 1.
    double sum = std::accumulate(data.begin(), data.end(), 0.0,
                                 [](double total, const profiled_data& pd) {
                                     return total + pd.capacity_adj;
                                 });
    double mean = sum / data.size();

    for (profiled_data& pd : data) {
        pd.capacity_adj /= mean;
    }

    // Step 2: Round capacity_adj to nearest multiple of 0.2
    for (profiled_data& pd : data) {
        pd.capacity_adj = std::round(pd.capacity_adj * 5) / 5;
    }
}


void getFinalizedData(int numthreads,double profile_time,std::vector<raw_data>& data_begin,std::vector<raw_data>& data_end,std::vector<profiled_data>& result_arr,std::vector<thread_args*> thread_arg){
  double largest_capacity_adj = 0;
  for (int i = 0; i < numthreads; i++) {
      u64 stolen_pass = data_end[i].steal_time - data_begin[i].steal_time;
      u64 preempts = data_end[i].preempts - data_begin[i].preempts;
      double used_time = (data_end[i].use_time - data_begin[i].use_time)*10000000;
      if(!(profile_time)==0){
	double capacity_perc_1 =  ((double)(used_time)/(used_time+stolen_pass));
	//double capacity_perc_2 = ((double)(profile_time-stolen_pass)/profile_time);
	//if(capacity_perc_2-capacity_perc_1>0.20){
	result_arr[i].capacity_perc = capacity_perc_1;
	//}else{
	//	result_arr[i].capacity_perc = capacity_perc_2;
	//}
	if(stolen_pass < 10000){
	result_arr[i].capacity_perc=1.0;
	}
      }else{
	result_arr[i].capacity_perc = 0;
	}
	if(vtop_banned[i]){
	result_arr[i].capacity_perc = 0.5;
	}

      result_arr[i].preempts = preempts;
      if(result_arr[i].capacity_perc < 0.001){
        result_arr[i].capacity_perc = 0.001;
      }
	if (profiler_iter % heavy_profile_interval == 0){
        double perf_use = thread_arg[i]->user_time;
        result_arr[i].capacity_adj = (1/perf_use) * data_end[i].raw_compute;
        if(result_arr[i].capacity_adj > largest_capacity_adj){
          largest_capacity_adj = result_arr[i].capacity_adj;
        }
     	
      }
      if(preempts == 0){
        result_arr[i].latency = 0;
      } else {
        result_arr[i].latency = stolen_pass/preempts; 
      }
      result_arr[i].max_latency = data_end[i].max_latency;
      addToHistory(result_arr[i].capacity_perc_hist,result_arr[i].capacity_perc);
      addToHistory(result_arr[i].capacity_adj_hist,result_arr[i].capacity_adj);
      addToHistory(result_arr[i].latency_hist,result_arr[i].latency);
      addToHistory(result_arr[i].preempts_hist,result_arr[i].preempts);
      result_arr[i].latency_ema = calculate_ema(decay_length,result_arr[i].latency_ema_a,result_arr[i].latency_ema,result_arr[i].latency);
      result_arr[i].capacity_perc_ema = calculate_ema(decay_length,result_arr[i].capacity_perc_ema_a,result_arr[i].capacity_perc_ema,result_arr[i].capacity_perc);
      
      result_arr[i].latency_ema = calculateStdDev(result_arr[i].latency_hist);
      result_arr[i].capacity_perc_stddev = calculateStdDev(result_arr[i].capacity_perc_hist);
    };
    for (int i = 0; i < numthreads; i++) {
      if(largest_capacity_adj != 0){
      result_arr[i].capacity_adj = result_arr[i].capacity_adj/largest_capacity_adj * 1024;
      }
    };
    
    if (profiler_iter % heavy_profile_interval == 0){
        //process_values(result_arr);
      }
}



void printResult(int cpunum,std::vector<profiled_data>& result,std::vector<thread_args*> thread_arg){
  for (int i = 0; i < cpunum; i++){
        std::cout <<"CPU:"<<i<<" TID:"<<thread_arg[i]->tid<<std::endl;
        std::cout<<"Capacity Perc:"<<result[i].capacity_perc<<":Latency:"<<result[i].latency<<":Preempts:"<<result[i].preempts<<":Capacity Raw:"<<result[i].capacity_adj<<std::endl;
        std::cout<<":Cperc stddev:"<<result[i].capacity_perc_stddev<<":Max latency:"<<result[i].max_latency;
        std::cout <<":Cperc ema: "<<result[i].capacity_perc_ema <<std::endl<<std::endl;
        
  }
  auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    auto timer = std::chrono::system_clock::to_time_t(now);

    std::tm bt = *std::localtime(&timer);
  
    std::cout << "hardtimestamp: " << (bt.tm_min*60000 + bt.tm_sec*1000 + ms.count()) << std::endl;
  std::cout<<"--------------"<<std::endl;
}


void give_to_kernel(int cpunum,std::vector<profiled_data>& result_arr){
  std::fstream write_file;
  std::string capacity_res;
  write_file.open("/proc/edit_capacity", std::ios::out);
  for (int i = 0; i < cpunum; i++){
	capacity_res = capacity_res + std::__cxx11::to_string((int)round(result_arr[i].capacity_perc_ema * 1024)) + ";";
//	capacity_res = capacity_res + std::__cxx11::to_string(result_arr[i].latency) + ";";
  }
  write_file << capacity_res;
  write_file.close();

  std::string latency_res;
  write_file.open("/proc/edit_latency", std::ios::out);
  for (int i = 0; i < cpunum; i++){
	latency_res = latency_res + std::__cxx11::to_string((int)round(result_arr[i].latency)) + ";";
//      capacity_res = capacity_res + std::__cxx11::to_string(result_arr[i].latency) + ";";
  }
  write_file <<latency_res;
  write_file.close();
}




void waitforWorkers(){
  pthread_mutex_lock(&ready_check);
  while(ready_counter != (num_threads-banned_amount)){
    pthread_cond_wait(&cv1, &ready_check);
  }
  pthread_mutex_unlock(&ready_check);
  ready_counter = 0;
}

void banVcpus(std::vector<profiled_data>& data_arr){
	std::ifstream file("/sys/fs/cgroup/user.slice/cpuset.cpus");
	if (!file) {
        	std::cerr << "Failed to access file" << std::endl;
        	return;
    	}
	std::string bans="";
	for(int i = 0; i<num_threads; i++){
		if(vtop_banned[i] != 1 &&  (data_arr[i].capacity_perc_ema > 0.1)){
		 	bans+=std::to_string(i)+",";
		}
	}
     if (!bans.empty()) {
        bans.pop_back();
    }

    // Write the banned vCPUs to the file
    std::ofstream outfile("/sys/fs/cgroup/user.slice/cpuset.cpus");
    if (!outfile) {
        std::cerr << "Failed to open file for writing" << std::endl;
        return;
    }

    outfile << bans;
    outfile.close();
    //std::ofstream ooffile("/sys/fs/cgroup/system.slice/cpuset.cpus");
    //ooffile <<bans;
    file.close();
}

void updateVectorFromBanlist(std::string fileLocation) {
    std::ifstream file(fileLocation);

    if (!file) {
        std::cerr << "Failed to access file" << std::endl;
        return;
    }

    // Set all elements of vtop_banned to 0
    std::fill(vtop_banned.begin(), vtop_banned.end(), 0);
    banned_amount = 0;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string item;
        while (std::getline(iss, item, ',')) {
            // Remove leading/trailing whitespace from the item
            item.erase(0, item.find_first_not_of(" \t"));
            item.erase(item.find_last_not_of(" \t") + 1);

            // Convert the item to an integer and update the vector
            try {
                int index = std::stoi(item);
                if (index >= 0 && index < vtop_banned.size()) {
                    	std::cout<<"vtop_banend"<<index<<std::endl;
			vtop_banned[index] = 1;
                }
            } catch (const std::invalid_argument& e) {
                // Skip invalid integers
                continue;
            }
        }
    }
    file.close();
}
void disableStragglerCpus(std::vector<profiled_data>& result_arr){
    std::string banlist = "";

    for (int z = 0; z < result_arr.size(); z++) {
        if (result_arr[z].capacity_perc_ema < 0.10) {
            banlist += std::to_string(z) + ",";
        }
    }

    if (!banlist.empty()) {
        banlist.pop_back();  // Remove the trailing comma
    }

    std::ofstream banlistFile("/home/ubuntu/banlist/vcap_strag.txt");
    if (banlistFile.is_open()) {
        banlistFile << banlist;
        banlistFile.close();
    } else {
        std::cout << "Unable to open file /home/ubuntu/banlist/vcap_strag.txt" << std::endl;
    }

}

void do_profile(std::vector<raw_data>& data_end,std::vector<thread_args*> thread_arg){

    std::vector<raw_data> data_begin(num_threads);
    std::vector<profiled_data> result_arr(num_threads);

    while(true){
      //If the last interval was heavy, move the threads to low priority. If interval is less then 2, obviously special workload. 
      if ((!heavy_profile_interval < 2) && ((profiler_iter-1) % heavy_profile_interval == 0)){
        for (int i = 0; i < num_threads; i++) {
          moveThreadtoLowPrio(thread_arg[i]->tid);
        }
      }
      updateVectorFromBanlist("/home/ubuntu/banlist/vtop.txt");
      //sleep during sleep
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_length));

      //We want to set the endtime and get data immediately after the threads have woken up in order to minimize innacuracy, this is to keep threads waiting
      endtime = high_resolution_clock::now() + std::chrono::milliseconds(1000000000);

      //this is for the heavy profile period
      awake_workers_flag=false;
      
      //wake up threads and broadcast 
      initialized = 1;
      pthread_cond_broadcast(&cv);

      //if it's a heavy profile period wait for the workers to wake up
      if((profiler_iter) % heavy_profile_interval == 0){
        waitforWorkers();
        awake_workers_flag=true;
      }

      //set the endtime and get data
      endtime = high_resolution_clock::now() + std::chrono::milliseconds(profile_time);
      get_cpu_information(num_threads,data_begin,thread_arg);
      //TODO-sleep every x ms and wake up to see if it's now(potentially)try nano sleep? (do some testing)
      //Wait for processors to finish profiling

      //sleep during profiling
      std::this_thread::sleep_for(std::chrono::milliseconds(profile_time));

      //wait for everybody to finish reporting data
      if ((profiler_iter) % heavy_profile_interval == 0){
        waitforWorkers();
      }
      get_cpu_information(num_threads,data_end,thread_arg);
      initialized = 0;
      //get actual profiling period
     double test = (profile_time * 1000000
        + static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count())
        - static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(endtime.time_since_epoch()).count()));
    //  test = (profile_time * 1000000);
getFinalizedData(num_threads,test,data_begin,data_end,result_arr,thread_arg);
      give_to_kernel(num_threads,result_arr);
       //If the next interval is heavy, move the threads to high priority.
      if ((profiler_iter+1) % heavy_profile_interval == 0){
        for (int i = 0; i < num_threads; i++) {
          moveThreadtoHighPrio(thread_arg[i]->tid);
        }
      }
      banVcpus(result_arr);
	disableStragglerCpus(result_arr);
      profiler_iter++;
      if(verbose){
        printResult(num_threads,result_arr,thread_arg);
      }
    }
}


std::vector<thread_args*> setup_threads(std::vector<pthread_t>& thread_array,std::vector<raw_data>& data_end){
  cpu_set_t cpuset;
  std::vector<thread_args*> threads_arg(num_threads);
  //create all the threads and initilize mutex
  for (int i = 0; i < num_threads; i++) {
    struct thread_args *args = new struct thread_args;
    //init mutex
    //TODO:use pthread_mutex_init
    //decide which cores to bind cpus to
    CPU_ZERO(&cpuset);
    CPU_SET(i, &cpuset);
    //give an id and assign mutex to all threads
    args->id = i;
    args->mutex = PTHREAD_MUTEX_INITIALIZER;
    args->addition_calc = &(data_end[i].raw_compute);
    //set prio of thread to MIN
    //TODO-error handling for thread creation mistakes
    pthread_create(&thread_array[i], NULL, run_computation, (void *) args);
    pthread_setaffinity_np(thread_array[i], sizeof(cpu_set_t), &cpuset);
    threads_arg[i] = args;
  }
  //we need to make sure that all the threads have fetched the thread ID before we go into whatever computation
  while(true){
    bool allset = true;
    for (int i = 0; i < num_threads; i++) {
      if(threads_arg[i]->tid == -1 ){
        bool allset = false;
      }
    }
    if(allset){
      break;
    }
   }
  
  return threads_arg;
}



int main(int argc, char *argv[]) {
  int num_cpus = sysconf(_SC_NPROCESSORS_ONLN); // Get the number of online processors
   vtop_banned.resize(num_cpus, 0); 
  //the threads need to be moved to root level cgroup before they can be distributed to high/low cgroup
  moveCurrentThread();
  //Setting up arguments
  const std::vector<std::string_view> args(argv, argv + argc);
  setArguments(args);

  std::vector<pthread_t> thread_array(num_threads);
  //note that this needs to be here because the computations and the main thread need to communicate with each other
  std::vector<raw_data> data_end(num_threads);

  std::vector<thread_args*> threads_arg = setup_threads(thread_array,data_end);
   moveThreadtoHighPrio(syscall(SYS_gettid));



  do_profile(data_end,threads_arg);

  //TODO-Close or start on command;
  //join the threads
  for (int i = 0; i < num_threads; i++) {
    pthread_join(thread_array[i], NULL);
  }
  printf("Process Finished");
  return 0;
}



int get_profile_time(int cpunum) {
  std::ifstream f("/proc/stat");
  std::string s;
  for (int i = 0; i <= cpunum; i++) {
    std::getline(f, s);
  }
  unsigned n;
  std::string l;
  if(std::istringstream(s)>> l >> n >> n >> n ) {
    return(n);
  }
  return 0;
}

u64 timespec_diff_to_ns(struct timespec *start, struct timespec *end) {
    u64 start_ns = start->tv_sec * 1000000000LL + start->tv_nsec;
    u64 end_ns = end->tv_sec * 1000000000LL + end->tv_nsec;
    return end_ns - start_ns;
}



void alertMainThread(){
  pthread_mutex_lock(&ready_check);
  ready_counter += 1;
  pthread_mutex_unlock(&ready_check);
  pthread_cond_signal(&cv1);
}





void* run_computation(void * arg)
{
    //TODO-Learn how to use kernel shark to visualize whole process
    struct thread_args *args = (struct thread_args *)arg;
    moveThreadtoLowPrio(syscall(SYS_gettid));
    args->tid = syscall(SYS_gettid);
    while(true) {
      stick_this_thread_to_core(args->id);
      struct timespec start,end,lstart,lend;
      //here to avoid a race condition
      bool heavy_interval = false;
      pthread_mutex_lock(&args->mutex);
      while (! initialized) {
        pthread_cond_wait(&cv, &args->mutex);
      }
      pthread_mutex_unlock(&args->mutex);
      int addition_calculator = 0;
      if (profiler_iter % heavy_profile_interval == 0){
        alertMainThread();
        while(!awake_workers_flag){
        }
        clock_gettime(CLOCK_MONOTONIC_RAW, &lstart);
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
        heavy_interval = true;
      }
     if(vtop_banned[args->id]==0){
      while(std::chrono::high_resolution_clock::now() < endtime) {
        addition_calculator += 1;
      };
      }else{
	std::this_thread::sleep_for(std::chrono::milliseconds(profile_time));
	}
      *args->addition_calc = addition_calculator;
      if(heavy_interval){
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
        clock_gettime(CLOCK_MONOTONIC_RAW, &lend);

        double test = static_cast<double>(timespec_diff_to_ns(&start, &end)) /static_cast<double>(timespec_diff_to_ns(&lstart, &lend));

        args->user_time = test;
        alertMainThread();

        }
      initialized = 0;
      }
      return NULL;
} 
