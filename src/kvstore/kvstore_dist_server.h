/*!
 * Copyright (c) 2015 by Contributors
 * \file mxnet_node.h
 * \brief implement mxnet nodes
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
#define MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
#include <queue>
#include <string>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>
#include <future>
#include <vector>
#include <fstream>
#include <sstream>
#include <sys/types.h>
#include <unistd.h>
#include "ps/ps.h"
#include "mxnet/kvstore.h"
#include <sys/time.h>

namespace mxnet {
namespace kvstore {

static const int kStopServer = -1;
static const int kSyncMode = -2; // BSP
static const int kSyncByGroupMode = -3; // GSP
static const int kSyncByStaleMode = -4; // SSP

/**
 * \brief executor runs a function using the thread called \ref Start
 */
class Executor {
 public:
  /**
   * \brief start the executor
   */
  void Start() {
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
      cond_.wait(lk, [this]{return !queue_.empty();});
      Block blk = std::move(queue_.front());
      queue_.pop();
      lk.unlock();

      if (blk.f) {
        blk.f(); blk.p->set_value();
      } else {
        blk.p->set_value(); break;
      }
      lk.lock();
    }
  }

  /**
   * \brief function
   */
  typedef std::function<void()> Func;

  /**
   * \brief let the thread called \ref Start to exec a function. threadsafe
   */
  void Exec(const Func& func) {
    Block blk(func);
    auto fut = blk.p->get_future();
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push(std::move(blk));
      cond_.notify_one();
    }
    fut.wait();
  }

  /**
   * \brief stop the thread, threadsafe
   */
  void Stop() {
    Exec(Func());
  }

 private:
  struct Block {
  explicit Block(const Func& func) : f(func), p(std::make_shared<std::promise<void>>()) { }
    Func f;
    std::shared_ptr<std::promise<void>> p;
  };
  std::queue<Block> queue_;
  std::mutex mu_;
  std::condition_variable cond_;
};

class KVStoreDistServer {
 public:
  KVStoreDistServer() {
    using namespace std::placeholders;
    ps_server_ = new ps::KVServer<float>(0);
    static_cast<ps::SimpleApp*>(ps_server_)->set_request_handle(
        std::bind(&KVStoreDistServer::CommandHandle, this, _1, _2));
    ps_server_->set_request_handle(
        std::bind(&KVStoreDistServer::DataHandle, this, _1, _2, _3));
    sync_mode_ = 0;
  }

  ~KVStoreDistServer() {
	file.close();
    delete ps_server_;
  }

  void set_controller(const KVStore::Controller& controller) {
    CHECK(controller);
    controller_ = controller;
  }

  void set_updater(const KVStore::Updater& updater)  {
    CHECK(updater);
    updater_ = updater;
  }

  /**
   * \brief blocked until received the command \a kSyncMode
   */
  void Run() {
    exec_.Start();
  }

 private:
  void CommandHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
    if (recved.head == kStopServer) {
      exec_.Stop();
    } else if (recved.head == kSyncMode) {
      sync_mode_ = kSyncMode;

      //merge and update time count
	  merge_update_filename = std::string("/home/") + std::string(getlogin()) + "/mxnet/example/image-classification/merge_update_time.log";
	  merge_update_file.open(merge_update_filename, std::ios::out | std::ios::app);
	  merge_flag = 0;
	  update_flag = 0;
	} else if (recved.head == kSyncByGroupMode) {
	  sync_mode_ = kSyncByGroupMode;
	  group_filename = std::string("/home/") + std::string(getlogin()) + "/mxnet/example/image-classification/groups";
	  group_file.open(group_filename, std::ios::in);
	  std::string s;
	  std::vector<int> group_nums;
	  std::multimap<std::string, int> ip_to_id;
	  std::string ip;

	  struct Node {
		std::string ip;
		int id;
		bool is_visited;
		Node(std::string ipc, int idc = 0, bool is_visitedc = false): ip(ipc) {}
	  };

	  std::vector<Node> nodes;
	  while (getline(group_file, s)) {
		std::istringstream f(s);
		while (getline(f, ip, ' ')) {
			nodes.push_back(Node(ip));
		}
	  }
	  std::sort(nodes.begin(), nodes.end(), [](const Node& a, const Node& b) {
		return a.ip < b.ip;
	  });
	  int count = nodes.size() - 1;
	  for (unsigned int i = 0; i < nodes.size(); ++i) {
		nodes[i].id = count * 2 + 9;
		--count;
		std::cout << "key = " << nodes[i].ip << "  value = " << nodes[i].id << std::endl;
	  }

	  group_file.close();
	  group_file.open(group_filename, std::ios::in);
	  while (getline(group_file, s)) {
		group_nums.clear();
		std::istringstream f(s);
		while (getline(f, ip, ' ')) {
	      for (auto& node : nodes) {
			if (node.ip == ip && node.is_visited == false) {
			  group_nums.push_back(node.id);
			  node.is_visited = true;
			}
		  }
		}
		groups.push_back(group_nums);
	  }
	  groups_count = groups.size();
	  store_group.resize(groups_count);
	  merge_buf_of_group.resize(groups_count);
	  for (auto group : groups) {
		for (auto num : group) {
		  std::cout << num << " ";
		}
		std::cout << std::endl;
	  }
	  staleness = 0;
	} else if (recved.head == kSyncByStaleMode) {
	  sync_mode_ = kSyncByStaleMode;
	  miniters_filename = std::string("/home/") + std::string(getlogin()) + "/mxnet/example/image-classification/ssp/miniters.log";
	  file.open(miniters_filename, std::ios::out);
	  is_first_push = true;
	  straggler = (ps::NumWorkers() - 1) * 2 + 9;
	  staleness = 0;
	} else {
      // let the main thread to execute ctrl, which is necessary for python
      exec_.Exec([this, recved]() {
        CHECK(controller_);
        controller_(recved.head, recved.body);
      });
      staleness = 0;
    }
    app->Response(recved);
  }

  void DataHandle(const ps::KVMeta& req_meta,
                  const ps::KVPairs<real_t>& req_data,
                  ps::KVServer<real_t>* server) {
    // do some check
    CHECK_EQ(req_data.keys.size(), (size_t)1);
    if (req_meta.push) {
      CHECK_EQ(req_data.lens.size(), (size_t)1);
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
    }

    int parameter_partition_num = ps::Postoffice::Get()->num_parameter_partition(); //yegeyan 2016.12.9

    int key;
    if (parameter_partition_num >= 0) {
      key = req_data.keys[0];
    }
    else {
      key = DecodeKey(req_data.keys[0]);
    }

    auto& stored = store_[key];

    // there used several WaitToRead, this is because \a recved's memory
    // could be deallocated when this function returns. so we need to make sure
    // the operators with \a NDArray are actually finished
    if (req_meta.push) {
      size_t ds[] = {(size_t)req_data.lens[0]};
      TShape dshape(ds, ds + 1);
      TBlob recv_blob((real_t*)req_data.vals.data(), // NOLINT(*)
                      dshape, cpu::kDevMask);
      NDArray recved = NDArray(recv_blob, 0);

      if (stored.is_none()) {
        // initialization
        stored = NDArray(dshape, Context());
        CopyFromTo(recved, &stored, 0);
        server->Response(req_meta);
        stored.WaitToRead();

		if (sync_mode_ == kSyncByGroupMode) {
		  for (unsigned int i = 0; i < groups_count; ++i) {
			store_group[i][key] = NDArray(dshape, Context());
			CopyFromTo(recved, &store_group[i][key], 0);
			server->Response(req_meta);
			store_group[i][key].WaitToRead();
		  }
		}
      } else if (sync_mode_ == kSyncMode) {
        // synced push
        auto& merged = merge_buf_[key];
        if (merged.array.is_none()) {
          merged.array = NDArray(dshape, Context());
        }

        if (merged.request.size() == 0) {
          CopyFromTo(recved, &merged.array, 0);
        } else {
          merged.array += recved;
          /*
		  int total_num = 1000;
		  clock_t start_time = clock();
		  for (int i = 0; i < total_num; ++i) {
			merged.array += recved;
		  }
		  clock_t end_time = clock();
		  std::cout << "--------Merge cost time: " << (end_time - start_time) * 1.0 / (CLOCKS_PER_SEC * total_num)
				  << std::endl;
		  */
        }

        merged.request.push_back(req_meta);

        if (merged.request.size() == (size_t)ps::NumWorkers()) {
          // let the main thread to execute updater_, which is necessary for
          // python
          if (updater_) {
        	worker_num = ps::NumWorkers();
            exec_.Exec([this, key, &merged, &stored](){
              CHECK(updater_);
              updater_(key, merged.array, &stored, worker_num);
            });
          } else {
            // if no updater, just copy
            CopyFromTo(merged.array, &stored);
          }

          for (const auto& req : merged.request) {
            server->Response(req);
          }
          merged.request.clear();
          stored.WaitToRead();
        } else {
          merged.array.WaitToRead();
        }
      } else if (sync_mode_ == kSyncByGroupMode){
          for (unsigned int i = 0; i < groups_count; ++i) {
        	for (unsigned int j = 0; j < groups[i].size(); ++j) {
        	  if (req_meta.sender == groups[i][j]) {
        	    auto& merged = merge_buf_of_group[i][key];
        		if (merged.array.is_none()) {
        		  merged.array = NDArray(dshape, Context());
        		}

        		if (merged.request.size() == 0) {
        		  CopyFromTo(recved, &merged.array, 0);
        		} else {
        		  merged.array += recved;
        		}

        	    merged.request.push_back(req_meta);

        		if (merged.request.size() == groups[i].size()) {
        		  // let the main thread to execute updater_, which is necessary for
        		  // python
        	      if (updater_) { //
        	    	worker_num = groups[i].size();
        			staleness = update_count_total[key] - update_count_group[i][key] + 1;
        			if (staleness <= 0) staleness = 1;
        			//staleness = 1;
        			exec_.Exec([this, key, &merged, &stored](){
        			  CHECK(updater_);
        			  updater_(key, merged.array, &stored, worker_num * staleness);
        			});
        			++update_count_total[key];
        			update_count_group[i][key] = update_count_total[key];
        		  } else {
        			// if no updater, just copy
        			CopyFromTo(merged.array, &stored);
        		  }
        		  for (const auto& req : merged.request) {
        			server->Response(req);
        		  }
        		  merged.request.clear();
        		  store_group[i][key] = stored;
        		  store_group[i][key].WaitToRead();
        		} else {
        		  merged.array.WaitToRead();
        		}
        		return;
        	  }
            }
          }
      } else if (sync_mode_ == kSyncByStaleMode) {
          if (updater_) {
        	// staleness = update_count_total[key] - update_count_worker[req_meta.sender][key] + 1;
        	// if (staleness <= 0) staleness = 1;
        	staleness = 1;
        	exec_.Exec([this, key, &recved, &stored](){
        	  CHECK(updater_);
        	  updater_(key, recved, &stored, staleness);
        	});
        	++update_count_total[key];
          } else {
        	// if no updater, just copy
        	CopyFromTo(recved, &stored);
          }
		  server->Response(req_meta);
		  stored.WaitToRead();

		  if(is_first_push) {
			total_key_num = store_.size();
			is_first_push = false;
		  }

		  if(req_meta.sender == straggler) {
			++min_key_num;
			if(min_key_num % total_key_num == 0) {
			  min_iter_num = min_key_num / total_key_num;
		      file.seekg(0, std::ios::beg);
			  file << min_iter_num;
			}
		  }
      } else {
		// async push
		if (updater_) { //
		  // worker_num = ps::NumWorkers() - 1;
		  // staleness = update_count_total[key] - update_count_worker[req_meta.sender][key] + 1;
		  // if (staleness <= 0) staleness = 1;
		  staleness = 1;
		  exec_.Exec([this, key, &recved, &stored](){
		  	CHECK(updater_);
		  	updater_(key, recved, &stored, staleness);
		  });
		  // ++update_count_total[key];
		  } else {
		   	// if no updater, just copy
		   	CopyFromTo(recved, &stored);
		  }
		  server->Response(req_meta);
		  stored.WaitToRead();
        }
      } else {
		if (sync_mode_ == kSyncByGroupMode) {
          for (unsigned int i = 0; i < groups_count; ++i) {
        	for (unsigned int j = 0; j < groups[i].size(); ++j) {
        	  if (req_meta.sender == groups[i][j]) {
        	    ps::KVPairs<real_t> response;
        		CHECK(!store_group[i][key].is_none()) << "init " << key << " first";
        		int len = store_group[i][key].shape()[0];
        		response.keys = req_data.keys;
        		response.lens = {len};
        		response.vals.CopyFrom(static_cast<const float*>(store_group[i][key].data().dptr_), len);
        		server->Response(req_meta, response);
        		return;
        	  }
        	}
          }
		} else {
		  ps::KVPairs<real_t> response;
		  CHECK(!stored.is_none()) << "init " << key << " first";
		  int len = stored.shape()[0];
		  response.keys = req_data.keys;
		  response.lens = {len};
		  response.vals.CopyFrom(static_cast<const float*>(stored.data().dptr_), len);
		  server->Response(req_meta, response);
		  update_count_worker[req_meta.sender][key] = update_count_total[key];
		}
    }
  }

  int DecodeKey(ps::Key key) {
    auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
    return key - kr.begin();
  }

  /**
   * \brief user defined
   */
  int sync_mode_; //yegeyan 2016.10.6
  KVStore::Controller controller_;
  KVStore::Updater updater_;

  std::unordered_map<int, NDArray> store_;

  std::vector<std::unordered_map<int, NDArray>> store_group;
  struct MergeBuf {
    std::vector<ps::KVMeta> request;
    NDArray array;
  };
  std::unordered_map<int, MergeBuf> merge_buf_;
  std::vector<std::unordered_map<int, MergeBuf>> merge_buf_of_group;
  std::string group_filename;
  std::ifstream group_file;
  std::vector<std::vector<int>> groups;
  unsigned int groups_count;
  int worker_num;

  std::string miniters_filename;
  std::fstream file;
  static long long int min_key_num;
  static long long int min_iter_num;
  static int total_key_num;
  bool is_first_push;
  std::set<int> keys;
  int straggler;

  std::unordered_map<int, long long int> update_count_total;
  std::unordered_map<int, std::unordered_map<int, long long int>> update_count_worker;
  std::unordered_map<int, std::unordered_map<int, long long int>> update_count_group;
  long long int staleness;
  Executor exec_;

  ps::KVServer<float>* ps_server_;

  //merge and update time count
  std::string merge_update_filename;
  std::fstream merge_update_file;
  long long int merge_flag;
  long long int update_flag;
};

long long int KVStoreDistServer::min_key_num = 0;
long long int KVStoreDistServer::min_iter_num = 0;
int KVStoreDistServer::total_key_num = 0;

}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
