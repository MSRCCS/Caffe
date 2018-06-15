#include <gflags/gflags.h>
#include <glog/logging.h>

#include "caffe/clusters.hpp"

namespace Clusters{
	
  int proc_rank_ = -1;   // The rank of this process in the world
  int proc_count_ = -1;  // The number of processes in the world
  int proc_local_rank_ = -1;   // The rank of the process in this node (if 1 process/node this is equal Caffe:solver_rank())
  int proc_local_count_ = -1;  // The number of processes in this node (if 1 process/node this is equal Caffe:solver_count())

  void Init() {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count_);
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &shmcomm);
    MPI_Comm_rank(shmcomm, &proc_local_rank_);
    MPI_Comm_size(shmcomm, &proc_local_count_);
    LOG(INFO) << "MPI world: ("<< Clusters::proc_rank() << "/" << Clusters::proc_count() << ") local:(" << Clusters::proc_local_rank() <<"/" << Clusters::proc_local_count() << ")";
  }
  
  void Finalize() {
    MPI_Finalize();	
  }
  int proc_rank() {
    CHECK_GT(proc_rank_, -1);
    return proc_rank_;
  }

  int proc_count() {
    CHECK_GT(proc_count_, -1);
    return proc_count_;
  }
  
  int proc_local_rank() {
    CHECK_GT(proc_local_rank_, -1);
    return proc_local_rank_;
  }

  int proc_local_count() {
    CHECK_GT(proc_local_count_, -1);
    return proc_local_count_;
  }

}
