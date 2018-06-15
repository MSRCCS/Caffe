#include <gflags/gflags.h>
#include <glog/logging.h>

#include "caffe/clusters.hpp"

namespace Clusters{
	
  int node_rank_ = -1;
  int node_count_ = -1;  
  int node_local_rank_ = -1;
  int node_local_count_ = -1;  

  void Init() {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &node_count_);
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &shmcomm);
    MPI_Comm_rank(shmcomm, &node_local_rank_);
    MPI_Comm_size(shmcomm, &node_local_count_);
    LOG(INFO) << "MPI world: ("<< Clusters::node_rank() << "/" << Clusters::node_count() << ") local:(" << Clusters::node_local_rank() <<"/" << Clusters::node_local_count() << ")";
  }
  
  void Finalize() {
    MPI_Finalize();	
  }
  int node_rank() {
    CHECK_GT(node_rank_, -1);
    return node_rank_;
  }

  int node_count() {
    CHECK_GT(node_count_, -1);
    return node_count_;
  }
  
  int node_local_rank() {
    CHECK_GT(node_local_rank_, -1);
    return node_local_rank_;
  }

  int node_local_count() {
    CHECK_GT(node_local_count_, -1);
    return node_local_count_;
  }

}
