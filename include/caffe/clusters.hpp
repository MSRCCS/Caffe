#ifndef CAFFE_CLUSTER_HPP_
#define CAFFE_CLUSTER_HPP_

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "caffe/proto/caffe.pb.h"

namespace Clusters{
	
  void Init();
  
  void Finalize();

  int proc_rank();

  int proc_count();

  int proc_local_rank();

  int proc_local_count();

}

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_
