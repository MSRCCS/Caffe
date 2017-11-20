#include "caffe/blob.hpp"
#include "caffe/tree_common.hpp"

#include <vector>

namespace {

char *fgetl(FILE *fp) {
    if (feof(fp))
        return 0;
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
            CHECK(line) << size << " could not be allocated";
        }
        size_t readsize = size - curr;
        if (readsize > INT_MAX)
            readsize = INT_MAX - 1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    if (line[curr - 1] == '\n')
        line[curr - 1] = '\0';

    return line;
}

}

namespace caffe {

void Tree::read(const char *filename) {
    FILE *fp = fopen(filename, "r");
    CHECK(fp) << "Cannot open the tree file: " << filename;

    char *line;
    int last_parent = -1;
    int group_size = 0;
    int groups = 0;
    int n = 0;
    while ((line = fgetl(fp)) != 0) {
        char *id = (char *)calloc(256, sizeof(char));
        int parent = -1;
        sscanf(line, "%s %d", id, &parent);
        parent_cpu_ptr_ = (int *)realloc(parent_cpu_ptr_, (n + 1) * sizeof(int));
        parent_cpu_ptr_[n] = parent;

        child_ = (int *)realloc(child_, (n + 1) * sizeof(int));
        child_[n] = -1;

        name_ = (char **)realloc(name_, (n + 1) * sizeof(char *));
        name_[n] = id;
        if (parent != last_parent) {
            ++groups;
            group_offset_cpu_ptr_ = (int *)realloc(group_offset_cpu_ptr_, groups * sizeof(int));
            group_offset_cpu_ptr_[groups - 1] = n - group_size;
            group_size_cpu_ptr_ = (int *)realloc(group_size_cpu_ptr_, groups * sizeof(int));
            group_size_cpu_ptr_[groups - 1] = group_size;
            group_size = 0;
            last_parent = parent;
        }
        group_cpu_ptr_ = (int *)realloc(group_cpu_ptr_, (n + 1) * sizeof(int));
        group_cpu_ptr_[n] = groups;
        if (parent >= 0) {
            child_[parent] = groups;
        }
        ++n;
        ++group_size;
    }
    fclose(fp);

    ++groups;
    group_offset_cpu_ptr_ = (int *)realloc(group_offset_cpu_ptr_, groups * sizeof(int));
    group_offset_cpu_ptr_[groups - 1] = n - group_size;
    group_size_cpu_ptr_ = (int *)realloc(group_size_cpu_ptr_, groups * sizeof(int));
    group_size_cpu_ptr_[groups - 1] = group_size;

    n_ = n;
    groups_ = groups;
    leaf_ = (int *)calloc(n, sizeof(int));
    int i;
    for (i = 0; i < n; ++i)
        leaf_[i] = 1;
    for (i = 0; i < n; ++i)
        if (parent_cpu_ptr_[i] >= 0)
            leaf_[parent_cpu_ptr_[i]] = 0;

    std::vector<int> shape{ groups };
    group_offset_.Reshape(shape);
    group_offset_.set_cpu_data(group_offset_cpu_ptr_);
    group_size_.Reshape(shape);
    group_size_.set_cpu_data(group_size_cpu_ptr_);
    shape[0] = n;
    parent_.Reshape(shape);
    parent_.set_cpu_data(parent_cpu_ptr_);
    group_.Reshape(shape);
    group_.set_cpu_data(group_cpu_ptr_);
}


}  // namespace caffe
