#include <vector>

#include <stdlib.h>
#include <float.h>
#include "caffe/layers/region_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

struct box {
    float x, y, w, h;
};

int entry_index(layer l, int batch, int location, int entry)
{
    int n = location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(l.coords + l.classes + 1) + entry*l.w*l.h + loc;
}

void softmax(const float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for (i = 0; i < n; ++i) {
        if (input[i*stride] > largest) largest = input[i*stride];
    }
    for (i = 0; i < n; ++i) {
        float e = exp(input[i*stride] / temp - largest / temp);
        sum += e;
        output[i*stride] = e;
    }
    for (i = 0; i < n; ++i) {
        output[i*stride] /= sum;
    }
}


void softmax_cpu(const float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    for (b = 0; b < batch; ++b) {
        for (g = 0; g < groups; ++g) {
            softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}

static inline float logistic_activate(float x) { 
    if (x > 0) {
        return 1. / (1. + exp(-x)); 
    } else {
        return exp(x) / (1 + exp(x));
    }
}
static inline float logistic_gradient(float x) { return (1 - x)*x; }

float activate(float x, ACTIVATION a)
{
    switch (a) {
    case LOGISTIC:
        return logistic_activate(x);
    }
    return 0;
}

void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    for (i = 0; i < n; ++i) {
        x[i] = activate(x[i], a);
    }
}

float gradient(float x, ACTIVATION a)
{
    switch (a) {
    case LOGISTIC:
        return logistic_gradient(x);
    }
    return 0;
}

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for (i = 0; i < n; ++i) {
        delta[i] *= gradient(x[i], a);
    }
}

float mag_array(float *a, int n)
{
    float sum = 0;

#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        sum += a[i] * a[i];
    }

    return sqrt(sum);
}

box float_to_box(const float *f)
{
    box b;
    b.x = f[0];
    b.y = f[1];
    b.w = f[2];
    b.h = f[3];
    return b;
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(const box &a, const box &b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(const box &a, const box &b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(const box &a, const box &b)
{
    return box_intersection(a, b) / box_union(a, b);
}

box get_region_box(const float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0 * stride]) / w;
    b.y = (j + x[index + 1 * stride]) / h;
    b.w = exp((double)x[index + 2 * stride]) * biases[2 * n] / w;   // adding (double) conversion for parity with linux
    b.h = exp((double)x[index + 3 * stride]) * biases[2 * n + 1] / h;
    return b;
}

float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_region_box(x, biases, n, index, i, j, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w*w / biases[2 * n]);
    float th = log(truth.h*h / biases[2 * n + 1]);

    delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
    delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);
    delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
    delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);
    return iou;
}

void delta_region_class(float *output, float *delta, int index, int cls, int classes, tree *hier, float scale, int stride, float *avg_cat)
{
    int i, n;
    if (hier) {
        float pred = 1;
        while (cls >= 0) {
            pred *= output[index + stride*cls];
            int g = hier->group[cls];
            int offset = hier->group_offset[g];
            for (i = 0; i < hier->group_size[g]; ++i) {
                delta[index + stride*(offset + i)] = scale * (0 - output[index + stride*(offset + i)]);
            }
            delta[index + stride*cls] = scale * (1 - output[index + stride*cls]);

            cls = hier->parent[cls];
        }
        *avg_cat += pred;
    }
    else {
        for (n = 0; n < classes; ++n) {
            delta[index + stride*n] = scale * (((n == cls) ? 1 : 0) - output[index + stride*n]);
            if (n == cls) *avg_cat += output[index + stride*n];
        }
    }
}

void malloc_error()
{
    fprintf(stderr, "Malloc error\n");
    exit(-1);
}

char *fgetl(FILE *fp)
{
    if (feof(fp)) return 0;
    size_t size = 512;
    char *line = (char *)malloc(size * sizeof(char));
    if (!fgets(line, size, fp)) {
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while ((line[curr - 1] != '\n') && !feof(fp)) {
        if (curr == size - 1) {
            size *= 2;
            line = (char *)realloc(line, size * sizeof(char));
            if (!line) {
                printf("%ld\n", (long)size);
                malloc_error();
            }
        }
        size_t readsize = size - curr;
        if (readsize > INT_MAX) readsize = INT_MAX - 1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    if (line[curr - 1] == '\n') line[curr - 1] = '\0';

    return line;
}

tree *read_tree(const char *filename)
{
    tree t;
    FILE *fp = fopen(filename, "r");

    char *line;
    int last_parent = -1;
    int group_size = 0;
    int groups = 0;
    int n = 0;
    while ((line = fgetl(fp)) != 0) {
        char *id = (char *)calloc(256, sizeof(char));
        int parent = -1;
        sscanf(line, "%s %d", id, &parent);
        t.parent = (int *)realloc(t.parent, (n + 1) * sizeof(int));
        t.parent[n] = parent;

        t.child = (int *)realloc(t.child, (n + 1) * sizeof(int));
        t.child[n] = -1;

        t.name = (char **)realloc(t.name, (n + 1) * sizeof(char *));
        t.name[n] = id;
        if (parent != last_parent) {
            ++groups;
            t.group_offset = (int *)realloc(t.group_offset, groups * sizeof(int));
            t.group_offset[groups - 1] = n - group_size;
            t.group_size = (int *)realloc(t.group_size, groups * sizeof(int));
            t.group_size[groups - 1] = group_size;
            group_size = 0;
            last_parent = parent;
        }
        t.group = (int *)realloc(t.group, (n + 1) * sizeof(int));
        t.group[n] = groups;
        if (parent >= 0) {
            t.child[parent] = groups;
        }
        ++n;
        ++group_size;
    }
    ++groups;
    t.group_offset = (int *)realloc(t.group_offset, groups * sizeof(int));
    t.group_offset[groups - 1] = n - group_size;
    t.group_size = (int *)realloc(t.group_size, groups * sizeof(int));
    t.group_size[groups - 1] = group_size;
    t.n = n;
    t.groups = groups;
    t.leaf = (int *)calloc(n, sizeof(int));
    int i;
    for (i = 0; i < n; ++i) t.leaf[i] = 1;
    for (i = 0; i < n; ++i) if (t.parent[i] >= 0) t.leaf[t.parent[i]] = 0;

    fclose(fp);
    tree *tree_ptr = (tree *)calloc(1, sizeof(tree));
    *tree_ptr = t;
    //error(0);
    return tree_ptr;
}

float get_hierarchy_probability(float *x, tree *hier, int c, int stride)
{
    float p = 1;
    while (c >= 0) {
        p = p * x[c*stride];
        c = hier->parent[c];
    }
    return p;
}

void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride)
{
    int j;
    for (j = 0; j < n; ++j) {
        int parent = hier->parent[j];
        if (parent >= 0) {
            predictions[j*stride] *= predictions[parent*stride];
        }
    }
    if (only_leaves) {
        for (j = 0; j < n; ++j) {
            if (!hier->leaf[j]) predictions[j*stride] = 0;
        }
    }
}

float hierarchy_prediction_one(float *predictions, tree *hier, int target, int stride)
{
    float result = 1;
    while (true) {
        result *= predictions[target * stride];
        target = hier->parent[target];
        if (target < 0) {
            break;
        }
    }
    return result;
}

int hierarchy_top_prediction(const float *predictions, tree *hier, float thresh, int stride)
{
    float p = 1;
    int group = 0;
    int i;
    while (1) {
        float max = 0;
        int max_i = 0;

        for (i = 0; i < hier->group_size[group]; ++i) {
            int index = i + hier->group_offset[group];
            float val = predictions[index *stride];
            if (val > max) {
                max_i = index;
                max = val;
            }
        }
        if (p*max > thresh) {
            p = p*max;
            group = hier->child[max_i];
            if (hier->child[max_i] < 0) return max_i;
        }
        else {
            if (group == 0) {
                return max_i;
            } else {
                return hier->parent[hier->group_offset[group]];
            } 
        }
    }
    return 0;
}

void correct_region_boxes(box *boxes, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w = 0;
    int new_h = 0;
    if (((float)netw / w) < ((float)neth / h)) {
        new_w = netw;
        new_h = (h * netw) / w;
    }
    else {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    for (i = 0; i < n; ++i) {
        box b = boxes[i];
        b.x = (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
        b.w *= (float)netw / new_w;
        b.h *= (float)neth / new_h;
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        boxes[i] = b;
    }
}

vector<int> read_map(const char *filename)
{
    vector<int> map;
    char *str;
    FILE *file = fopen(filename, "r");
    CHECK(file) << filename;
    while((str=fgetl(file))){
        map.push_back(atoi(str));
    }
    return map;
}

void get_region_boxes(layer l, int output_classes, int w, int h, int netw, int neth, float thresh, float **probs, box *boxes, int only_objectness, 
        const vector<int> &map, float tree_thresh, int relative)
{
    int i, j, n, z;
    float *predictions = l.output;
    if (l.batch == 2) {
        float *flip = l.output + l.outputs;
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w / 2; ++i) {
                for (n = 0; n < l.n; ++n) {
                    for (z = 0; z < l.classes + l.coords + 1; ++z) {
                        int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                        int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                        float swap = flip[i1];
                        flip[i1] = flip[i2];
                        flip[i2] = swap;
                        if (z == 0) {
                            flip[i1] = -flip[i1];
                            flip[i2] = -flip[i2];
                        }
                    }
                }
            }
        }
        for (i = 0; i < l.outputs; ++i) {
            l.output[i] = (l.output[i] + flip[i]) / 2.;
        }
    }
    for (i = 0; i < l.w*l.h; ++i) {
        int row = i / l.w;
        int col = i % l.w;
        for (n = 0; n < l.n; ++n) {
            int index = n*l.w*l.h + i;
            for (j = 0; j < output_classes; ++j) {
                probs[index][j] = 0;
            }
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
            float scale = predictions[obj_index];
            boxes[index] = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h, l.w*l.h);

            int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1);
            if (l.softmax_tree) {
                if (map.size() > 0) {
                    //hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0, l.w*l.h);
                    float max_prob = 0;
                    for (j = 0; j < map.size(); ++j) {
                        float prob;
                        //int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + map[j]);
                        //prob = scale*predictions[class_index];
                        prob = scale * hierarchy_prediction_one(predictions + class_index, l.softmax_tree, map[j], l.w * l.h);

                        prob = (prob > thresh) ? prob : 0;
                        if (prob > max_prob) {
                            max_prob = prob;
                        }
                        probs[index][j] = prob;
                    }
                    probs[index][map.size()] = max_prob;
                }
                else {
                    int j = hierarchy_top_prediction(predictions + class_index, l.softmax_tree, tree_thresh, l.w*l.h);
                    probs[index][j] = (scale > thresh) ? scale : 0;
                    probs[index][l.classes] = scale;
                }
            }
            else {
                float max = 0;
                for (j = 0; j < l.classes; ++j) {
                    int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + j);
                    float prob = scale*predictions[class_index];
                    probs[index][j] = (prob > thresh) ? prob : 0;
                    if (prob > max) max = prob;
                    // TODO REMOVE
                    // if (j != 15 && j != 16) probs[index][j] = 0; 
                    /*
                    if (j != 0) probs[index][j] = 0;
                    int blacklist[] = {121, 497, 482, 504, 122, 518,481, 418, 542, 491, 914, 478, 120, 510,500};
                    int bb;
                    for (bb = 0; bb < sizeof(blacklist)/sizeof(int); ++bb){
                    if(index == blacklist[bb]) probs[index][j] = 0;
                    }
                    */
                }
                probs[index][l.classes] = max;
            }
            if (only_objectness) {
                probs[index][0] = scale;
            }
        }
    }
    correct_region_boxes(boxes, l.w*l.h*l.n, w, h, netw, neth, relative);
}

typedef struct {
    int index;
    int cls;
    float **probs;
} sortable_bbox;

int nms_comparator(const void *pa, const void *pb)
{
    sortable_bbox a = *(sortable_bbox *)pa;
    sortable_bbox b = *(sortable_bbox *)pb;
    float diff = a.probs[a.index][b.cls] - b.probs[b.index][b.cls];
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

void do_nms_obj(box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    vector<sortable_bbox> s(total);

    for (i = 0; i < total; ++i) {
        s[i].index = i;
        s[i].cls = classes;
        s[i].probs = probs;
    }

    qsort(&s[0], total, sizeof(sortable_bbox), nms_comparator);
    for (i = 0; i < total; ++i) {
        if (probs[s[i].index][classes] == 0) continue;
        box a = boxes[s[i].index];
        for (j = i + 1; j < total; ++j) {
            box b = boxes[s[j].index];
            if (box_iou(a, b) > thresh) {
                for (k = 0; k < classes + 1; ++k) {
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
}

void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    vector<sortable_bbox> s(total);

    for (i = 0; i < total; ++i) {
        s[i].index = i;
        s[i].cls = 0;
        s[i].probs = probs;
    }

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            s[i].cls = k;
        }
        qsort(&s[0], total, sizeof(sortable_bbox), nms_comparator);
        for (i = 0; i < total; ++i) {
            if (probs[s[i].index][k] == 0) continue;
            box a = boxes[s[i].index];
            for (j = i + 1; j < total; ++j) {
                box b = boxes[s[j].index];
                if (box_iou(a, b) > thresh) {
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
}

template <typename Dtype>
void RegionLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    const RegionLossParameter &region_param = this->layer_param().region_loss_param();
    layer &l = this->l_;
    l.coords = region_param.coords();
    l.classes = region_param.classes();
    for (int i = 0; i < region_param.biases_size(); ++i) {
        biases_.push_back(region_param.biases(i));
    }
    CHECK(biases_.size() % 2 == 0) << "the number of biases must be even: " << biases_.size();
    l.n = biases_.size() / 2;
    l.biases = &biases_[0];

    l.object_scale = region_param.object_scale();
    l.noobject_scale = region_param.noobject_scale();
    l.class_scale = region_param.class_scale();
    l.coord_scale = region_param.coord_scale();
    l.thresh = region_param.thresh();
    l.bias_match = region_param.bias_match();
    l.rescore = region_param.rescore();
    l.softmax_tree = region_param.has_tree() ? read_tree(region_param.tree().c_str()) : 0;
    l.softmax = region_param.softmax();
    l.temperature = 1;
    
    anchor_aligned_images_ = region_param.anchor_aligned_images();
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1));
    seen_images_ = this->blobs_[0]->mutable_cpu_data();
    *seen_images_ = 0;
}

template <typename Dtype>
void RegionLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    output_.ReshapeLike(*bottom[0]);
    output_gpu_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void RegionLossLayer<Dtype>::prepare_net_layer(network &net, layer &l, const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    net.truth = (const float *)bottom[1]->cpu_data();
    // We should use bottom[0]->mutable_cpu_diff() here. But the buffer will be cleared before backward due to shared memory optimization.
    // So we use output_.mutable_cpu_diff() instead and restore this to bottom[0] diff in Backward_cpu(...) and Backward_gpu(...)
    // l.delta = (float *)bottom[0]->mutable_cpu_diff();
    l.cost = (float *)top[0]->mutable_cpu_data();
    l.loss_weight = top[0]->cpu_diff()[0];

    l.truths = bottom[1]->shape(1);
    l.boxes = l.truths / (l.coords + 1);

    l.batch = bottom[0]->shape(0);
    l.w = bottom[0]->shape(3);
    l.h = bottom[0]->shape(2);
    l.outputs = l.h * l.w * l.n * (l.classes + l.coords + 1);
    l.inputs = l.outputs;
}

void find_target_for_non_bbox_label(const box &truth, int cls,
        const layer &l, int b, int &target_i, int &target_j, int &target_n)
{
    float best_confidence = -1;
    target_i = -1;
    target_j = -1;
    target_n = -1;
    for (int j = 0; j < l.h; j++) {
        for (int i = 0; i < l.w; i++) {
            for (int n = 0; n < l.n; n++) {
                int obj_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, l.coords);
                int cls_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, l.coords + 1 + cls);
                float confidence = l.output[cls_index] * l.output[obj_index];
                if (confidence > best_confidence) {
                    best_confidence = confidence;
                    target_i = i;
                    target_j = j;
                    target_n = n;
                }
            }
        }
    }
}

bool find_target_for_bbox_label(const box &truth, 
        const layer &l, int b, int &target_i, int &target_j, int &target_n)
{
    target_i = (truth.x * l.w);
    target_j = (truth.y * l.h);
    if (target_i < 0 || target_i >= l.w || target_j < 0 || target_j >= l.h) {
        return false;
    }
    box truth_shift = truth;
    truth_shift.x = 0;
    truth_shift.y = 0;
    float best_iou = -1;

    for (int n = 0; n < l.n; ++n) {
        int box_index = entry_index(l, b, n*l.w*l.h + target_j*l.w + target_i, 0);
        box pred = get_region_box(l.output, l.biases, n, box_index, target_i, target_j, l.w, l.h, l.w*l.h);
        if (l.bias_match) {
            pred.w = l.biases[2 * n] / l.w;
            pred.h = l.biases[2 * n + 1] / l.h;
        }
        pred.x = 0;
        pred.y = 0;
        float iou = box_iou(pred, truth_shift);
        if (iou > best_iou) {
            best_iou = iou;
            target_n = n;
        }
    }
    return true;
}

bool is_global_label_without_box(const box &truth) {
    return truth.x > 100000 && truth.y > 100000;
}

void print_bb_obj_class_loss(layer &l) {
    float loss_xy = 0;
    float loss_wh = 0;
    float loss_objness = 0;
    float loss_class = 0;
    for (int b = 0; b < l.batch; b++) {
        for (int n = 0; n < l.n; n++) {
            int idx_xy = entry_index(l, b, n * l.w * l.h, 0);
            loss_xy += pow(mag_array(l.delta + idx_xy, l.w * l.h * 2), 2);
            int idx_wh = entry_index(l, b, n * l.w * l.h, 2);
            loss_wh += pow(mag_array(l.delta + idx_wh, l.w * l.h * (l.coords - 2)), 2);
            int idx_objness = entry_index(l, b, n * l.w * l.h, l.coords);
            loss_objness += pow(mag_array(l.delta + idx_objness, l.w * l.h), 2);
            int idx_class = entry_index(l, b, n * l.w * l.h, l.coords + 1);
            loss_class += pow(mag_array(l.delta + idx_class, l.w * l.h * l.classes), 2);
        }
    }
    LOG_IF(INFO, Caffe::root_solver()) << "loss_xy: " << loss_xy / l.batch << "; "
        << "loss_wh: " << loss_wh / l.batch << "; "
        << "loss_objness: " << loss_objness / l.batch << "; "
        << "loss_class: " << loss_class / l.batch;
}

template <typename Dtype>
void RegionLossLayer<Dtype>::forward_for_loss(network &net, layer &l)
{
    seen_images_ = this->blobs_[0]->mutable_cpu_data();

    const RegionLossParameter &region_param = this->layer_param().region_loss_param();
    bool display = region_param.debug_info() > 0 && ((((int)*seen_images_) / l.batch) % region_param.debug_info()) == 0;

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    float avg_iou = 0;
    float recall = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    if (l.softmax_tree) {
        for (int b = 0; b < l.batch; ++b) {
            for (int t = 0; t < l.boxes; ++t) {
                box truth = float_to_box(net.truth + t * (l.coords + 1) + b*l.truths);
                if (!truth.x) 
                    break;
                if (is_global_label_without_box(truth)) {
                    assert(t == 0);

                    int cls = net.truth[t * (l.coords + 1) + b*l.truths + l.coords];
                    float maxp = 0;
                    int maxi = 0;
                    //#pragma omp parallel for reduction(max:maxp)
                    for (int n = 0; n < l.n*l.w*l.h; ++n) {
                        int class_index = entry_index(l, b, n, l.coords + 1);
                        int obj_index = entry_index(l, b, n, l.coords);
                        float scale = l.output[obj_index];
                        l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                        float p = scale*get_hierarchy_probability(l.output + class_index, l.softmax_tree, cls, l.w*l.h);
                        if (p > maxp) {
                            maxp = p;
                            maxi = n;
                        }
                    }
                    int class_index = entry_index(l, b, maxi, l.coords + 1);
                    int obj_index = entry_index(l, b, maxi, l.coords);
                    delta_region_class(l.output, l.delta, class_index, cls, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat);
                    if (l.output[obj_index] < .3) 
                        l.delta[obj_index] = l.object_scale * (.3 - l.output[obj_index]);
                    else  
                        l.delta[obj_index] = 0;
                    ++class_count;

                    // The entire of this batch is class-only
                    break;
                }
            }
        }
    }
#pragma omp parallel for reduction(+:avg_anyobj)
    for (long long idx = 0; idx < l.batch * l.h * l.w * l.n; idx++) {
        int b = idx / (l.h * l.w * l.n);
        long long left = idx % (l.h * l.w * l.n);
        int j = left / (l.w * l.n);
        left = left % (l.w * l.n);
        int i = left / l.n;
        int n = left % l.n;
        int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
        box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
        float best_iou = 0;
        int best_cls = 0;
        for (int t = 0; t < l.boxes; ++t) {
            box truth = float_to_box(net.truth + t * (l.coords + 1) + b*l.truths);
            if (is_global_label_without_box(truth)) {
                continue; // this is a class label without bbox
            }
            if (!truth.x) 
                break;
            float iou = box_iou(pred, truth);
            if (iou > best_iou) {
                best_iou = iou;
                best_cls = (int)(*(net.truth + t * (l.coords + 1) + b * l.truths));
            }
        }
        int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, l.coords);
        avg_anyobj += l.output[obj_index];
        l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
        if (best_iou > l.thresh) {
            l.delta[obj_index] = 0;
        }
    }
    if ((*seen_images_) * Caffe::solver_count() < this->anchor_aligned_images_) {
#pragma omp parallel for
        for (long long idx = 0; idx < l.batch * l.h * l.w * l.n; idx++) {
            int b = idx / (l.h * l.w * l.n);
            long long left = idx % (l.h * l.w * l.n);
            int j = left / (l.w * l.n);
            left = left % (l.w * l.n);
            int i = left / l.n;
            int n = left % l.n;
            int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
            box truth = { 0 };
            truth.x = (i + .5) / l.w;
            truth.y = (j + .5) / l.h;
            truth.w = l.biases[2 * n] / l.w;
            truth.h = l.biases[2 * n + 1] / l.h;
            delta_region_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, l.delta, .01, l.w*l.h);
        }
    }
    for (int b = 0; b < l.batch; ++b) {
        for (int t = 0; t < l.boxes; ++t) {
            const box truth = float_to_box(net.truth + t * (l.coords + 1) + b*l.truths);
            int cls = net.truth[t * (l.coords + 1) + b*l.truths + l.coords];
            int best_n = 0;
            if (!truth.x) 
                break;

            float iou; // this is the target iou
            bool is_global_label = is_global_label_without_box(truth);
            int i, j;
            if (is_global_label) {
                find_target_for_non_bbox_label(truth, cls, l, b, i, j, best_n);
            } else {
                bool valid = find_target_for_bbox_label(truth, l, b, i, j, best_n);
                if (!valid) {
                    continue;
                }

                int box_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 0);
                iou = delta_region_box(truth, l.output, l.biases, best_n, 
                        box_index, i, j, l.w, l.h, l.delta, l.coord_scale *  (2 - truth.w*truth.h), l.w*l.h);
                if (iou > .5) recall += 1;
                avg_iou += iou;
            }

            //l.delta[best_index + l.coords] = iou - l.output[best_index + l.coords];
            int obj_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords);
            avg_obj += l.output[obj_index];
            if (!is_global_label) {
                l.delta[obj_index] = l.object_scale * (1 - l.output[obj_index]);
                if (l.rescore) {
                    l.delta[obj_index] = l.object_scale * (iou - l.output[obj_index]);
                }
            } else  {
                if (l.output[obj_index] < 0.3) {
                    l.delta[obj_index] = l.object_scale * (0.3 - l.output[obj_index]);
                } else {
                    l.delta[obj_index] = 0;
                }
            }

            //if (l.map) cls = l.map[cls];
            int class_index;
            class_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords + 1);
            delta_region_class(l.output, l.delta, class_index, cls, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat);
            ++count;
            ++class_count;
        }
    }
    
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);

    if (display)
    {
        char msg[1024];
        sprintf(msg, "Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", 
                avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, count);
        LOG_IF(INFO, Caffe::root_solver()) << msg;
        print_bb_obj_class_loss(l);
    }

    (*seen_images_) += l.batch;
}

template <typename Dtype>
void RegionLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // prepare wrapper environment for forward computation
    network &net = this->net_;
    layer &l = this->l_;

    net.input = (const float *)bottom[0]->cpu_data();
    prepare_net_layer(net, l, bottom, top);
    l.output = (float *)output_.mutable_cpu_data();
    l.delta = (float *)output_.mutable_cpu_diff();

    // perform computation
    memcpy(l.output, net.input, l.outputs*l.batch * sizeof(float));

    for (int b = 0; b < l.batch; ++b) {
        for (int n = 0; n < l.n; ++n) {
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            activate_array(l.output + index, l.w*l.h, LOGISTIC);
        }
    }
    if (l.softmax_tree) {
        int count = l.coords + 1;
        for (int i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            int index = entry_index(l, 0, 0, count);
            softmax_cpu(net.input + index, group_size, l.batch * l.n, l.inputs / l.n, l.w*l.h, 1, l.w*l.h, l.temperature, l.output + index);
            count += group_size;
        }
    }
    else if (l.softmax) {
        int index = entry_index(l, 0, 0, l.coords + 1);
        softmax_cpu(net.input + index, l.classes, l.batch*l.n, l.inputs / l.n, l.w*l.h, 1, l.w*l.h, 1, l.output + index);
    }

    if (this->phase_ == TEST) return;

    // compute the remaining part for loss
    forward_for_loss(net, l);

    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    // multiplicate delta with -loss_weight to fit for caffe's sgd solver.
    caffe_cpu_scale(l.outputs*l.batch, -l.loss_weight / l.batch, l.delta, l.delta);
    *(l.cost) /= l.batch;

    if (top.size() == 2) {
        top[1]->CopyFrom(output_);
    }
}

template <typename Dtype>
void RegionLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    // prepare wrapper environment for backward computation
    network &net = this->net_;
    layer &l = this->l_;
    net.input = (const float *)bottom[0]->cpu_data();
    prepare_net_layer(net, l, bottom, top);
    l.output = (float *)output_.mutable_cpu_data();
    l.delta = (float *)bottom[0]->mutable_cpu_diff();
    memcpy(l.delta, output_.cpu_diff(), l.batch * l.outputs * sizeof(float));

    int b, n;
    for (b = 0; b < l.batch; ++b) {
        for (n = 0; n < l.n; ++n) {
            int index = entry_index(l, b, n*l.w*l.h, 0);
            gradient_array(l.output+ index, 2 * l.w*l.h, LOGISTIC, l.delta+ index);
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            gradient_array(l.output+ index, l.w*l.h, LOGISTIC, l.delta+ index);
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(RegionLossLayer);
#endif

INSTANTIATE_CLASS(RegionLossLayer);
REGISTER_LAYER_CLASS(RegionLoss);

template <typename Dtype>
void RegionOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    Layer<Dtype>::LayerSetUp(bottom, top);

    const RegionOutputParameter &region_param = this->layer_param().region_output_param();
    thresh_ = region_param.thresh();
    hier_thresh_ = region_param.hier_thresh();
    nms_ = region_param.nms();
    classes_ = region_param.classes();
    feat_stride_ = region_param.feat_stride();

    layer &l = this->l_;
    l.classes = region_param.classes();
    l.coords = region_param.coords(); 
    for (int i = 0; i < region_param.biases_size(); ++i) {
        biases_.push_back(region_param.biases(i));
    }
    CHECK(biases_.size() % 2 == 0) << "the number of biases must be even: " << biases_.size();
    l.n = biases_.size() / 2;
    l.biases = &biases_[0];
    l.softmax = true;
    l.softmax_tree = region_param.has_tree() ? read_tree(region_param.tree().c_str()) : 0;
    if (region_param.has_map()) {
        this->map_ = read_map(region_param.map().c_str());
        classes_ = this->map_.size();
    } else if (l.softmax_tree) {
        CHECK_EQ(l.softmax_tree->n, l.classes);
    }
    this->class_specific_nms_ = region_param.class_specific_nms();
    l.temperature = 1;

    vector<int> shape(3);
    shape[0] = 1;
    shape[1] = 1;  // placeholder which will be updated in Reshape(...)
    shape[2] = 4;
    top[0]->Reshape(shape);

    shape[2] = classes_ + 1;
    top[1]->Reshape(shape);
    CHECK_EQ(bottom[0]->channels(), l.n * (l.classes + 1 + l.coords));
}

template <typename Dtype>
void RegionOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    layer &l = this->l_;

    l.batch = bottom[0]->shape(0);
    l.w = bottom[0]->shape(3);
    l.h = bottom[0]->shape(2);
    l.outputs = l.h * l.w * l.n * (l.classes + l.coords + 1);
    l.inputs = l.outputs;

    CHECK(l.batch == 1 || l.batch == 2) << "RegionOutputLayer only accepts batch size 1 or 2 (for flip)";

    net_w_ = l.w * feat_stride_;
    net_h_ = l.h * feat_stride_;

    output_.ReshapeLike(*bottom[0]);
    output_gpu_.ReshapeLike(*bottom[0]);

    vector<int> shape = top[0]->shape();
    shape[1] = l.w*l.h*l.n;
    top[0]->Reshape(shape);
    shape = top[1]->shape();
    shape[1] = l.w*l.h*l.n;
    top[1]->Reshape(shape);
}

template <typename Dtype>
void RegionOutputLayer<Dtype>::GetRegionBoxes(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    layer &l = this->l_;
    l.output = (float *)output_.mutable_cpu_data();

    const float *im_info = (const float *)bottom[1]->cpu_data();
    int im_w = im_info[1];
    int im_h = im_info[0];
    // when used for Caffe timing, im_w and im_h might be 0 and we need to give them valid values.
    if (im_w == 0)
        im_w = net_w_;
    if (im_h == 0)
        im_h = net_h_;

    float *box_output = (float *)top[0]->mutable_cpu_data();
    float *prob_output = (float *)top[1]->mutable_cpu_data();
    box *boxes = (box *)box_output;
    vector<float*> probs(l.w*l.h*l.n);
    for (int j = 0; j < l.w*l.h*l.n; ++j)
        probs[j] = prob_output + (classes_ + 1) * j;
    
    get_region_boxes(l, classes_, im_w, im_h, net_w_, net_h_, thresh_, &probs[0], boxes, 0, this->map_, hier_thresh_, 0);//1);
    if (nms_ > 0) {
        if (this->class_specific_nms_) {
            do_nms_sort(boxes, &probs[0], l.w*l.h*l.n, classes_, nms_);//0.4);
        } else {
            do_nms_obj(boxes, &probs[0], l.w*l.h*l.n, classes_, nms_);//0.4);
        }
    }
}

template <typename Dtype>
void RegionOutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    layer &l = this->l_;
    l.output = (float *)output_.mutable_cpu_data();
    network net;
    net.input = (float *)bottom[0]->cpu_data();

    // As in RegionLossLayer, apply sigmoid or softmax operations on bottom data.
    memcpy(l.output, net.input, l.outputs*l.batch * sizeof(float));
    for (int b = 0; b < l.batch; ++b) {
        for (int n = 0; n < l.n; ++n) {
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            activate_array(l.output + index, l.w*l.h, LOGISTIC);
        }
    }
    if (l.softmax_tree) {
        int count = l.coords + 1;
        for (int i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            int index = entry_index(l, 0, 0, count);
            softmax_cpu(net.input + index, group_size, l.batch * l.n, l.inputs / l.n, l.w*l.h, 1, l.w*l.h, l.temperature, l.output + index);
            count += group_size;
        }
    }
    else if (l.softmax) {
        int index = entry_index(l, 0, 0, l.coords + 1);
        softmax_cpu(net.input + index, l.classes, l.batch*l.n, l.inputs / l.n, l.w*l.h, 1, l.w*l.h, 1, l.output + index);
    }
    
    GetRegionBoxes(bottom, top);
}

#ifdef CPU_ONLY
STUB_GPU(RegionOutputLayer);
#endif

INSTANTIATE_CLASS(RegionOutputLayer);
REGISTER_LAYER_CLASS(RegionOutput);

}  // namespace caffe
