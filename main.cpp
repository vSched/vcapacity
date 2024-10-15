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
#include <filesystem>
#include <algorithm> 
namespace fs = std::filesystem;
using namespace std;
typedef uint64_t u64;

//initialize global variables
int num_threads = sysconf(_SC_NPROCESSORS_ONLN);
int sleep_length = 1000;
int profile_time = 100;
int decay_length = 2;
int heavy_profile_interval = 5;
int base_heavy_profiling = 5;
int context_window = 5;
bool verbose = false;
double milliseconds_totick_factor = static_cast<double>(sysconf(_SC_CLK_TCK))/1000.0;

//Heavy Profiling Utilities
bool awake_workers_flag = false;
//Profiling utilties
int initialized = 0;
int profiler_iter = -1;
chrono::time_point<chrono::_V2::system_clock, chrono::_V2::system_clock::duration> endtime;
chrono::time_point<chrono::_V2::system_clock, chrono::_V2::system_clock::duration> hard_endtime;

pthread_cond_t prof_cond = PTHREAD_COND_INITIALIZER;
pthread_cond_t heav_prof_cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t heav_prof_mut = PTHREAD_MUTEX_INITIALIZER;
int heav_ready = 0;

int average_capacity = 500;

float straggler_cutoff = 0.20;
void* run_computation(void * arg);
vector<int> vtop_banlist;
int banned_amount = 0;

std::filesystem::path banlistPath = "../banlist";



//Arguments for each thread
struct thread_args {
  int id;
  int tid = -1;
  pthread_mutex_t mutex;
  u64 *addition_calc;
  double user_time;
};

//structs for raw and processed data
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

  deque<double> capacity_perc_hist;
  deque<double> capacity_adj_hist;
  deque<double> latency_hist;
  deque<double> preempts_hist;

  double preempts;
  double capacity_perc;
  double capacity_adj;
  double latency;
  double max_latency;
};

//Math utilities
double calculate_std_dev(const deque<double>& v) {
  if (v.size() == 0) {
    return 0.0;
  }
  double sum = accumulate(v.begin(), v.end(), 0.0);
  double mean = sum / v.size();

  double sq_sum = inner_product(v.begin(), v.end(), v.begin(), 0.0);
  double std_dev = sqrt(sq_sum / v.size() - mean * mean);
  return std_dev;
}

double calculate_ema(int decay_len, double& ema_help, double prev_ema,double new_value) {
  double decay_factor = pow(0.5,(1/(double)decay_len));
  double newA = (1+decay_factor*ema_help);
  double result = (new_value + ((prev_ema)*ema_help*decay_factor))/newA;
  ema_help = newA;
  return result;
}


//parameter utilities
string_view get_option(const vector<string_view>& args, const string_view& option_name) {
  for (auto it = args.begin(), end = args.end(); it != end; ++it) {
    if ((*it == option_name) && (it + 1 != end))
      return *(it + 1);
  }  
  return "";
}


bool has_option(
    const vector<string_view>& args, 
    const string_view& option_name) {
    for (auto it = args.begin(), end = args.end(); it != end; ++it) {
        if (*it == option_name)
            return true;
    }
    
    return false;
};

//Core/Prio utilities
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


void move_thread_to_low_prio(pid_t tid) {
  string path = "/sys/fs/cgroup/lw_prgroup/cgroup.threads";
  ofstream ofs(path, ios_base::app);
  if (!ofs) {
    cerr << "Low priority cGroup cannot be found - Try running the setup script\n";
    return;
  }
  ofs << tid << "\n";
  ofs.close();
  struct sched_param params;
  params.sched_priority = sched_get_priority_min(SCHED_IDLE);
  sched_setscheduler(tid,SCHED_IDLE,&params);
}

void move_thread_to_high_prio(pid_t tid) {
  string path = "/sys/fs/cgroup/hi_prgroup/cgroup.threads";
  ofstream ofs(path, ios_base::app);
  if (!ofs) {
    cerr << "High priority cGroup cannot be found - Try running the setup script\n";
    return;
  }
  ofs << tid << "\n";
  ofs.close();
}

//Used as pre-step to moving high/low prio
void move_current_thread() {
  pid_t tid;
  tid = syscall(SYS_gettid);
  string path = "/sys/fs/cgroup/cgroup.procs";
  ofstream ofs(path, ios_base::app);
  if (!ofs) {
    cerr << "Something is wrong with cgroup process- check linux version is right\n";
    return;
  }
  ofs << tid << "\n";
  ofs.close();
  struct sched_param params;
  params.sched_priority = sched_get_priority_max(SCHED_RR);
  sched_setscheduler(tid,SCHED_RR,&params);
}


void get_cpu_information(int cpunum,vector<raw_data>& data_arr,vector<thread_args*> thread_arg){
  ifstream f("/proc/vcap_info");
  if (!f.is_open()) {
    cerr << "Check to see if kernel modules have been installed - vsched is necessary. "<< endl;
    return;
  }
  string s;

  for (int i = 0; i < cpunum; i++) {
    getline(f,s);
    getline(f,s);
    data_arr[i].preempts = stoull(s);
    getline(f,s);
    data_arr[i].steal_time = stoull(s);
    getline(f,s);
    data_arr[i].max_latency = stoull(s);
  }

  ifstream stat_file("/proc/stat");
  string line;
  
  // Skip the first line (total CPU stats)
  getline(stat_file, line);
	for (int i = 0; i < cpunum && getline(stat_file, line); i++) {
    istringstream iss(line);
    string cpu;
    u64 user_time, nice_time, system_time;
    if (!(iss >> cpu >> user_time >> nice_time >> system_time)) {
      cerr << "Error reading CPU data for CPU " << i << endl;
      continue;
    }

    if (cpu == "cpu" + to_string(i)) {
      data_arr[i].use_time = user_time + nice_time + system_time;
    } else {
      cerr << "Unexpected CPU identifier: " << cpu << " for index " << i << endl;
    }
  }

  // Check if we've read data for all CPUs
  if (stat_file.eof() && cpunum > 0) {
    cerr << "Warning: Reached end of file before reading all CPU data" << endl;
  }
}


//helper function to set context window to be short
void add_to_history(deque<double>& history_list,double item){
  if(history_list.size() > (size_t)context_window) {
    history_list.pop_front();
  }
  history_list.push_back(item);
}


void set_arguments(const vector<string_view>& arguments) {
  verbose = has_option(arguments, "-v");

  auto set_option_value = [&](const string_view& option, int& target) {
    if (auto value = get_option(arguments, option); !value.empty()) {
      try {
        target = stoi(string(value));
      }catch (const invalid_argument&) {
        throw invalid_argument(string("Invalid argument for option ") + string(option));
      }catch (const out_of_range&) {
        throw out_of_range(string("Out of range argument for option ") + string(option));
      }
    }
  };

  set_option_value("-s", sleep_length);
  set_option_value("-p", profile_time);
  set_option_value("-d", decay_length);
  set_option_value("-c", context_window);
  set_option_value("-i", base_heavy_profiling);
  heavy_profile_interval = base_heavy_profiling;
  num_threads = sysconf( _SC_NPROCESSORS_ONLN );
}


void process_raw_capacity(vector<profiled_data>& data) {
  double sum = accumulate(data.begin(), data.end(), 0.0,
                          [](double total, const profiled_data& pd) {
                                     return total + pd.capacity_adj;
                          });
  double mean = sum / data.size();
  for (profiled_data& pd : data) {
    pd.capacity_adj /= mean;
  }
  //raw capacity is meant to capture ARM-big-little architetcture not necessarily frequency fluctuations,
  //so our rounding is quite aggressive
  for (profiled_data& pd : data) {
	if(pd.capacity_adj<0.5){
   	 pd.capacity_adj  = 0;
	}else{
		pd.capacity_adj=1;
	}
  }
  //if the last 5 raw capacities are stable, we can decrease heavy profiling frequency
  //for (profiled_data& pd : data) {
  //  pd.capacity_adj = round(pd.capacity_adj * 2) / 2;
  //}
}


bool checkAndUpdateVtopStatus() {
    std::ifstream inFile(banlistPath / "vtop_stat.txt");
    if (!inFile.is_open()) {
        throw std::runtime_error("Unable to open file for reading: ");
    }

    std::string content;
    std::getline(inFile, content);
    inFile.close();

    // Remove any whitespace from the content
    content.erase(std::remove_if(content.begin(), content.end(), ::isspace), content.end());

    if (content == "0") {
        return false;
    } else if (content == "1") {
        // Change the content to "0"
        std::ofstream outFile(banlistPath / "vtop_stat.txt", std::ios::trunc);
        if (!outFile.is_open()) {
            throw std::runtime_error("Unable to open file for writing: ");
        }
        outFile << "0";
        outFile.close();
        return true;
    } else {
        throw std::runtime_error("Invalid content in file");
    }
}


void get_finalized_data(int numthreads,double profile_time,vector<raw_data>& data_begin,vector<raw_data>& data_end,vector<profiled_data>& result_arr,vector<thread_args*> thread_arg){
  double largest_capacity_adj = 0;
  int decay_heavy = 1;
  float total_capacity = 0;
  int total_countable = 0;
   int count = 0;
    double mean = 0.0;
    double M2 = 0.0;

  for (int i = 0; i < numthreads; i++) {
    u64 stolen_pass = data_end[i].steal_time - data_begin[i].steal_time;
    u64 preempts = data_end[i].preempts - data_begin[i].preempts;
    double used_time = (data_end[i].use_time - data_begin[i].use_time)*10000000;
    if(!(profile_time)==0){
	    double capacity_perc_1 = ((double)(used_time)/(used_time+stolen_pass));
  	    result_arr[i].capacity_perc = capacity_perc_1;
      if(stolen_pass < 10000){
	      result_arr[i].capacity_perc=1.0;
	    }
    }else{
	    result_arr[i].capacity_perc = 0;
	  }
    if(vtop_banlist[i]){
    result_arr[i].capacity_perc = 0.5;
    }
    result_arr[i].preempts = preempts;
    if(result_arr[i].capacity_perc < 0.001){
      result_arr[i].capacity_perc = 0.001;
    }
       ++count;
        double delta = result_arr[i].capacity_perc_ema - mean;
        mean += delta / count;
        double delta2 = result_arr[i].capacity_perc_ema - mean;
        M2 += delta * delta2;

    if(!vtop_banlist[i] && (result_arr[i].capacity_perc > straggler_cutoff)){
	  total_capacity += result_arr[i].capacity_perc_ema * 1024;
	  total_countable += 1;

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
      add_to_history(result_arr[i].capacity_perc_hist,result_arr[i].capacity_perc);
      add_to_history(result_arr[i].latency_hist,result_arr[i].latency);
      add_to_history(result_arr[i].preempts_hist,result_arr[i].preempts);
      result_arr[i].latency_ema = calculate_ema(decay_length,result_arr[i].latency_ema_a,result_arr[i].latency_ema,result_arr[i].latency);
      result_arr[i].capacity_perc_ema = calculate_ema(decay_length,result_arr[i].capacity_perc_ema_a,result_arr[i].capacity_perc_ema,result_arr[i].capacity_perc);
      result_arr[i].capacity_perc_stddev = calculate_std_dev(result_arr[i].capacity_perc_hist);
    };
    

    if (profiler_iter % heavy_profile_interval == 0){
        process_raw_capacity(result_arr);
        for (int i = 0; i < numthreads; i++) {
          add_to_history(result_arr[i].capacity_adj_hist,result_arr[i].capacity_adj);
          result_arr[i].capacity_adj_stddev = calculate_std_dev(result_arr[i].capacity_adj_hist);
            if(result_arr[i].capacity_adj_stddev > 0.1){
              decay_heavy = 0;
            }
        }
	bool getvtopChanged = checkAndUpdateVtopStatus();
        if(decay_heavy && !getvtopChanged){
          heavy_profile_interval = round(heavy_profile_interval * 1.6);
        } else {
          heavy_profile_interval = base_heavy_profiling;
      }

    }
    double variance = M2 / (count - 1);
    if(variance>0.1){
    //std::cout<<"here's variance"<<variance<<std::endl;
    average_capacity = (int)(total_capacity/total_countable);
    }else{
    	average_capacity = 0;
    }
}



void print_result(int cpunum,vector<profiled_data>& result,vector<thread_args*> thread_arg){
  for (int i = 0; i < cpunum; i++){
        cout <<"CPU:"<<i<<" TID:"<<thread_arg[i]->tid<<endl;
        cout<<"Capacity Perc:"<<result[i].capacity_perc<<":Latency:"<<result[i].latency<<":Preempts:"<<result[i].preempts<<":Capacity Raw:"<<result[i].capacity_adj<<endl;
        cout<<":Cperc stddev:"<<result[i].capacity_perc_stddev<<":Max latency:"<<result[i].max_latency;
        cout <<":Cperc ema: "<<result[i].capacity_perc_ema<<":Latency EMA:"<<result[i].latency_ema<<endl<<endl;
  }
  auto now = chrono::system_clock::now();
    auto ms = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()) % 1000;
    auto timer = chrono::system_clock::to_time_t(now);

    tm bt = *localtime(&timer);
  
    cout << "hardtimestamp: " << (bt.tm_min*60000 + bt.tm_sec*1000 + ms.count()) << endl;
  cout<<"--------------"<<endl;
}


void give_to_kernel(int cpunum,vector<profiled_data>& result_arr){
  fstream write_file;
  string capacity_res;
  write_file.open("/proc/vcapacity_write", ios::out);
  for (int i = 0; i < cpunum; i++){
	  capacity_res = capacity_res + __cxx11::to_string((int)round(result_arr[i].capacity_perc_ema * 1024)) + ";";
  }
  write_file << capacity_res;
  write_file.close();

  write_file.open("/proc/vav_capacity_write", ios::out);

  write_file << average_capacity;
  write_file.close();
  string latency_res;
  write_file.open("/proc/vlatency_write", ios::out);
  for (int i = 0; i < cpunum; i++){
	  latency_res = latency_res + __cxx11::to_string((int)round(result_arr[i].latency_ema)) + ";";
  }
  write_file <<latency_res;
  write_file.close();
}


void wait_for_workers(){
  pthread_mutex_lock(&heav_prof_mut);
  while(heav_ready != (num_threads-banned_amount)){
    pthread_cond_wait(&heav_prof_cond, &heav_prof_mut);
  }
  pthread_mutex_unlock(&heav_prof_mut);
  heav_ready = 0;
}


void ban_vcpus(vector<profiled_data>& data_arr){
	ifstream file("/sys/fs/cgroup/user.slice/cpuset.cpus");
	if (!file) {
        	cerr << "Check if setup shell script is ran" << endl;
        	return;
    	}
	string bans="";
	for(int i = 0; i<num_threads; i++){
		if(vtop_banlist[i] != 1 &&  (data_arr[i].capacity_perc_ema > straggler_cutoff)){
		 	bans+=to_string(i)+",";
		}
	}
     if (!bans.empty()) {
        bans.pop_back();
    }

    // Write the banned vCPUs to the file
    ofstream outfile("/sys/fs/cgroup/user.slice/cpuset.cpus");
    if (!outfile) {
        cerr << "Check if setup shell script is ran" << endl;
        return;
    }

    outfile << bans;
    outfile.close();
    ofstream outfile1("/sys/fs/cgroup/system.slice/cpuset.cpus");
    if (!outfile1) {
        cerr << "Check if setup shell script is ran" << endl;
        return;
    }

    outfile1 << bans;
    outfile1.close();


    file.close();
}


void updateVectorFromBanlist(string fileLocation) {
    ifstream file(fileLocation);

    if (!file) {
        cerr << "Banlist not found" << endl;
        return;
    }

    // Set all elements of vtop_banned to 0
    fill(vtop_banlist.begin(), vtop_banlist.end(), 0);
    banned_amount = 0;
    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        string item;
        while (getline(iss, item, ',')) {
            // Remove leading/trailing whitespace from the item
            item.erase(0, item.find_first_not_of("\t"));
            item.erase(item.find_last_not_of(" \t") + 1);

            // Convert the item to an integer and update the vector
            try {
                int index = stoi(item);
                if (index >= 0 && (size_t)index < vtop_banlist.size()) {
			vtop_banlist[index] = 1;
                }
            } catch (const invalid_argument& e) {
                // Skip invalid integers
                continue;
            }
        }
    }
    file.close();
}


void disable_straggler_cpus(vector<profiled_data>& result_arr){
    string banlist = "";

    for (int z = 0; z < (int)result_arr.size(); z++) {
        if (result_arr[z].capacity_perc_ema < straggler_cutoff) {
          banlist += to_string(z) + ",";
        }
    }

    if (!banlist.empty()) {
      banlist.pop_back();  // Remove the trailing comma
    }

    ofstream banlistFile(banlistPath / "vcap_strag.txt");
    if (banlistFile.is_open()) {
      banlistFile << banlist;
      banlistFile.close();
    } else {
      cout << "Unable to open banlist file" << endl;
    }
}

bool preemptive_leave(vector<raw_data>& data_begin,vector<raw_data>& data_end){
  int max_preempts = -2;
  bool maxed_capacity = true;
  int preempts = 0;
  bool total_maxed_capacity = false;

  for (int i = 0; i < num_threads; i++) {
    preempts = data_end[i].preempts - data_begin[i].preempts;
    if(preempts> max_preempts) {
      max_preempts = preempts;
    }
    maxed_capacity = ((data_end[i].steal_time - data_begin[i].steal_time) < 1000000);

    if(!maxed_capacity){
	total_maxed_capacity=false;
    }

    if(preempts<3 && !maxed_capacity && !vtop_banlist[i]) {
      return false;
    }
  }

  return (max_preempts>7 || total_maxed_capacity);
}


void do_small_profile(vector<raw_data>& data_begin,vector<raw_data>& data_end, vector<thread_args*> thread_arg){
  while(true){
      //wake up threads and broadcast 
      pthread_cond_broadcast(&prof_cond);
      //endtime = chrono::high_resolution_clock::now() + chrono::milliseconds(20);
      this_thread::sleep_for(chrono::milliseconds(10));
      get_cpu_information(num_threads,data_end,thread_arg);
      if(preemptive_leave(data_begin,data_end) || chrono::high_resolution_clock::now() > endtime){
        endtime = chrono::high_resolution_clock::now();
	break;
      }
    }
}


void do_profile(vector<raw_data>& data_end,vector<thread_args*> thread_arg){
    std::vector<raw_data> data_begin(num_threads);
    std::vector<profiled_data> result_arr(num_threads);
    while(true){
      //If the last interval was heavy, move the threads to low priority. If interval is less then 2, obviously special workload. 
      if ( !(heavy_profile_interval < 2) && ((profiler_iter-1) % heavy_profile_interval == 0)){
        for (int i = 0; i < num_threads; i++) {
          move_thread_to_low_prio(thread_arg[i]->tid);
        }
      }
      updateVectorFromBanlist(banlistPath / "vtop.txt");
      //sleep during sleep
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_length));

      //We want to set the endtime and get data immediately after the threads have woken up in order to minimize innacuracy, this is to keep threads waiting
      endtime = chrono::high_resolution_clock::now() + std::chrono::milliseconds(1000000000);

      //this is for the heavy profile period
      awake_workers_flag=false;

      //wake up threads and broadcast 
      initialized = 1;
      pthread_cond_broadcast(&prof_cond);

      //if it's a heavy profile period wait for the workers to wake up
      if((profiler_iter) % heavy_profile_interval == 0){
        wait_for_workers();
        awake_workers_flag=true;
      }

      //set the endtime and get data
      endtime = chrono::high_resolution_clock::now() + std::chrono::milliseconds(profile_time);
      get_cpu_information(num_threads,data_begin,thread_arg);
      //TODO-sleep every x ms and wake up to see if it's now(potentially)try nano sleep? (do some testing)
      //Wait for processors to finish profiling

      //sleep during profiling
      //std::this_thread::sleep_for(std::chrono::milliseconds(profile_time));
	do_small_profile(data_begin,data_end,thread_arg);
      //wait for everybody to finish reporting data
      if ((profiler_iter) % heavy_profile_interval == 0){
        wait_for_workers();
      }
      get_cpu_information(num_threads,data_end,thread_arg);
      initialized = 0;
      //get actual profiling period
     double test = (profile_time * 1000000
        + static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count())
        - static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(endtime.time_since_epoch()).count()));
    //  test = (profile_time * 1000000);
get_finalized_data(num_threads,test,data_begin,data_end,result_arr,thread_arg);
      give_to_kernel(num_threads,result_arr);
       //If the next interval is heavy, move the threads to high priority.
      if ((profiler_iter+1) % heavy_profile_interval == 0){
        for (int i = 0; i < num_threads; i++) {
          move_thread_to_high_prio(thread_arg[i]->tid);
        }
      }
      ban_vcpus(result_arr);
      disable_straggler_cpus(result_arr);
      profiler_iter++;
      if(verbose){
        print_result(num_threads,result_arr,thread_arg);
      }
    }
}



vector<thread_args*> setup_threads(vector<pthread_t>& thread_array,vector<raw_data>& data_end){
  cpu_set_t cpuset;
  vector<thread_args*> threads_arg(num_threads);
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
  bool allset = true;
  while(true){
    allset = true;
    for (int i = 0; i < num_threads; i++) {
      if(threads_arg[i]->tid == -1 ){
        allset = false;
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
  vtop_banlist.resize(num_cpus, 0); 
  //the threads need to be moved to root level cgroup before they can be distributed to high/low cgroup
  move_current_thread();
  //Setting up arguments
  const vector<string_view> args(argv, argv + argc);
  set_arguments(args);
	
  vector<pthread_t> thread_array(num_threads);
  //note that this needs to be here because the computations and the main thread need to communicate with each other
  vector<raw_data> data_end(num_threads);

  vector<thread_args*> threads_arg = setup_threads(thread_array,data_end);
   move_thread_to_high_prio(syscall(SYS_gettid));

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
  ifstream f("/proc/stat");
  string s;
  for (int i = 0; i <= cpunum; i++) {
    getline(f, s);
  }
  unsigned n;
  string l;
  if(istringstream(s)>> l >> n >> n >> n ) {
    return(n);
  }
  return 0;
}

u64 timespec_diff_to_ns(struct timespec *start, struct timespec *end) {
    u64 start_ns = start->tv_sec * 1000000000LL + start->tv_nsec;
    u64 end_ns = end->tv_sec * 1000000000LL + end->tv_nsec;
    return end_ns - start_ns;
}



void alert_main_thread(){
  pthread_mutex_lock(&heav_prof_mut);
  heav_ready += 1;
  pthread_mutex_unlock(&heav_prof_mut);
  pthread_cond_signal(&heav_prof_cond);
}





void* run_computation(void * arg)
{
    //TODO-Learn how to use kernel shark to visualize whole process
    struct thread_args *args = (struct thread_args *)arg;
    move_thread_to_low_prio(syscall(SYS_gettid));
    args->tid = syscall(SYS_gettid);
    while(true) {
      stick_this_thread_to_core(args->id);
      struct timespec start,end,lstart,lend;
      //here to avoid a race condition
      bool heavy_interval = false;
      pthread_mutex_lock(&args->mutex);
      while (! initialized) {
        pthread_cond_wait(&prof_cond, &args->mutex);
      }
      pthread_mutex_unlock(&args->mutex);
      int addition_calculator = 0;
      if (profiler_iter % heavy_profile_interval == 0){
        alert_main_thread();
        while(!awake_workers_flag){
        }
        clock_gettime(CLOCK_MONOTONIC_RAW, &lstart);
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
        heavy_interval = true;
      }
     if(vtop_banlist[args->id]==0){
      while(chrono::high_resolution_clock::now() < endtime) {
        addition_calculator += 1;
      };
      }else{
	this_thread::sleep_for(chrono::milliseconds(profile_time));
	}
      *args->addition_calc = addition_calculator;
      if(heavy_interval){
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
        clock_gettime(CLOCK_MONOTONIC_RAW, &lend);

        double test = static_cast<double>(timespec_diff_to_ns(&start, &end)) /static_cast<double>(timespec_diff_to_ns(&lstart, &lend));

        args->user_time = test;
        alert_main_thread();

        }
      initialized = 0;
      }
      return NULL;
} 
