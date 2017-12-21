#include <algorithm>
#include <vector>
#include <math.h>
#include "cuda_runtime.h"
#include "curand.h"
#include "cuda.h"

#include "caffe/layers/nms_filter_layer.hpp"
#include "caffe/region_common.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_iou_mask(
    const Dtype* bbs_data,
    int outer_num, int inner_num, 
    float thresh,
    unsigned int* mask_data) {
    const int block_size = sizeof(unsigned int) * 8;
    // TODO: we could also multiply classes, to paralellize, but should avoid going above MAX_INT
    const int count = inner_num * (inner_num - 1) / 2 * outer_num;
    CUDA_KERNEL_LOOP(block_index, (count + block_size - 1) / block_size) {
        int index = block_index * block_size;
        unsigned int mask = 1u;
        mask_data[block_index] = 0;
        for (int b = 0; b < block_size; ++b, ++index, mask <<= 1u) {
            if (index >= count)
                return;
            int n = index % outer_num;
            int m = index / outer_num;

            // m - i == j * (j - 1) / 2  for 0 <= i < j < inner_num
            const int j = (int)floor((sqrt(8.0f * m + 1.0f) + 1) / 2);
            const int i = m - j * (j - 1) / 2;

            auto i_bb = bbs_data + (n * inner_num + i) * 4;
            auto j_bb = bbs_data + (n * inner_num + j) * 4;
            Dtype curr_iou = TBoxIou<Dtype>(i_bb[0], i_bb[1], i_bb[2], i_bb[3],
                                            j_bb[0], j_bb[1], j_bb[2], j_bb[3]);
            if (curr_iou > thresh)
                mask_data[block_index] |= mask;
        }
    }
}

template <typename Dtype>
__device__ void bottom_up_argmerge(const Dtype* p, 
                                   int left, int right, int end,
                                   const int* src, int* dst) {
    int i = left;
    int j = right;
    // Merge 2 already sorted lists
    for (int k = left; k < end; ++k) {
        if (i < right && (j >= end || p[src[i]] > p[src[j]])) {
            dst[k] = src[i];
            i++;
        } else {
            dst[k] = src[j];
            j++;
        }
    }
}

template <typename Dtype>
__global__ void kernel_channel_argmergesort(
    int outer_num, int channels, int inner_num,
    int width, int chunks,
    int c, 
    const Dtype* data,
    int* src, int* dst) {
    CUDA_KERNEL_LOOP(index, outer_num * chunks) {
        const int cidx = index % chunks;
        const int n = index / chunks;
        int left = cidx * width;
        int right = min(left + width / 2, inner_num);
        int end = min(left + width, inner_num);
        int* src_idx = src + n * inner_num;
        int* dst_idx = dst + n * inner_num;
        if (width == 2) {
            // Initialize the index
            if (right < end)
                src_idx[right] = left + 1;
            src_idx[left] = left + 0;
        }
        bottom_up_argmerge(data + (n * channels + c) * inner_num,
                           left, right, end,
                           src_idx, dst_idx);
    }
}

template <typename Dtype>
__global__ void kernel_nms_filter(
    int outer_num, int channels, int inner_num,
    int c, 
    const unsigned int* mask_data,
    const int* src,
    Dtype* conf_data) {
    int block_size = sizeof(unsigned int) * 8;
    CUDA_KERNEL_LOOP(n, outer_num) {
        const int* src_idx = src + n * inner_num;
        int dim = n * channels * inner_num;
        for (int i_idx = 0; i_idx < inner_num; ++i_idx) {
            int i = src_idx[i_idx];
            if (conf_data[dim + c * inner_num + i] == 0)
                continue;
            for (int j_idx = i_idx + 1; j_idx < inner_num; ++j_idx) {
                int j = src_idx[j_idx];
                int m;
                if (j > i)
                    m = i + j * (j - 1) / 2;
                else
                    m = j + i * (i - 1) / 2;
                int index = m * outer_num + n;
                int block_index = index / block_size;
                int b = index % block_size;
                if (!(mask_data[block_index] & (1u << b)))
                    continue;
                conf_data[dim + c * inner_num + j] = 0;
            }
        }
    }
}

template <typename Dtype>
void NMSFilterLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    int blob_idx = 0;
    auto blob_bbs = bottom[blob_idx++];
    auto blob_conf = bottom[blob_idx++];

    auto top_conf = top[0];
    auto top_conf_data = top_conf->mutable_gpu_data();
    caffe_copy(blob_conf->count(), blob_conf->gpu_data(), top_conf_data);

    auto bbs_data = blob_bbs->gpu_data();

    if (nms_ <= 0 || inner_num_ == 1)
        return;

    int actual_classes = classes_;
    if (actual_classes <= 0)
        actual_classes = channels_;

    int block_size = sizeof(unsigned int) * 8;
    int count = inner_num_ * (inner_num_ - 1) / 2 * outer_num_;
    for (int c = 0; c < actual_classes; ++c) {
        // TODO: Find the mask of all classes at once
        kernel_iou_mask << <CAFFE_GET_BLOCKS((count + block_size - 1) / block_size),
            CAFFE_CUDA_NUM_THREADS >> > (bbs_data,
                                         outer_num_, inner_num_, 
                                         nms_,
                                         mask_.mutable_gpu_data());

        int* idx_data = idx_.mutable_gpu_data();
        int* idx_tmp = idx_.mutable_gpu_diff();
        // Start swapped if loop runs for an odd number
        bool is_swapped = ((int)ceil(log2((double)inner_num_))) % 2 != 0;
        // TODO: Use dynamic parallelism for devices with 3.5 compute capability
        for (int width = 2; width < inner_num_ * 2; width *= 2) {
            int chunks = (inner_num_ + width - 1) / width;
            int* src_idx = is_swapped ? idx_tmp : idx_data;
            int* dst_idx = is_swapped ? idx_data : idx_tmp;
            kernel_channel_argmergesort << <CAFFE_GET_BLOCKS(outer_num_ * chunks),
                CAFFE_CUDA_NUM_THREADS >> > (outer_num_, channels_, inner_num_,
                                             width, chunks,
                                             c, 
                                             blob_conf->gpu_data(),
                                             src_idx, dst_idx);
            is_swapped = !is_swapped;
        }
        // Even if there were just one batch better suppress inside a kernel to avoid having to copy 
        //  the intermediate vectors to the host.
        kernel_nms_filter << <CAFFE_GET_BLOCKS(outer_num_),
            CAFFE_CUDA_NUM_THREADS >> > (outer_num_, channels_, inner_num_,
                                         c, 
                                         mask_.gpu_data(),
                                         idx_.gpu_data(),
                                         top_conf_data
                                         );
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(NMSFilterLayer);

}  // namespace caffe

