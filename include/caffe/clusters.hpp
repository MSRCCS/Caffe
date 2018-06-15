#include <mpi.h>
#include "caffe/proto/caffe.pb.h"

namespace Clusters{
	
  void Init();
  
  void Finalize();

  int node_rank();

  int node_count();

  int node_local_rank();

  int node_local_count();

}
