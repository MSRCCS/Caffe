#include <mpi.h>
#include "caffe/proto/caffe.pb.h"

namespace Clusters{
	
  void Init();
  
  void Finalize();

  int proc_rank();

  int proc_count();

  int proc_local_rank();

  int proc_local_count();

}
