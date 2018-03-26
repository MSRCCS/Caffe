#include "caffe/blob.hpp"
#include "caffe/tree_common.hpp"

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
    int last_sub_group = -1;
    int group_size = 0;
    int groups = 0;
    int sub_groups = 0;
    int n = 0;
    while ((line = fgetl(fp)) != 0) {
        char *id = (char *)calloc(256, sizeof(char));
        int parent = -1;
        int sub_group = -1;
        int count = sscanf(line, "%s %d %d", id, &parent, &sub_group);
        CHECK_GE(count, 2) << "Error reading node: " << n << " in tree:" << filename;
        if (count == 2)
            sub_group = -1;
        CHECK_LT(parent, n) << "Out of order parent for node: " << n << " in tree:" << filename;
        parent_cpu_ptr_ = (int *)realloc(parent_cpu_ptr_, (n + 1) * sizeof(int));
        parent_cpu_ptr_[n] = parent;

        child_cpu_ptr_ = (int *)realloc(child_cpu_ptr_, (n + 1) * sizeof(int));
        child_cpu_ptr_[n] = -1;
        child_size_cpu_ptr_ = (int *)realloc(child_size_cpu_ptr_, (n + 1) * sizeof(int));
        child_size_cpu_ptr_[n] = 0;

        name_ = (char **)realloc(name_, (n + 1) * sizeof(char *));
        name_[n] = id;
        bool new_group = false;
        bool new_sub_group = false;
        if (parent != last_parent) {
            new_group = true;
            last_parent = parent;
            last_sub_group = sub_group;
            sub_groups = 0;
        } else if (sub_group != last_sub_group) {
            CHECK_GT(sub_group, last_sub_group) << "node: " << n << " out of order sub-groups in tree:" << filename;
            new_sub_group = true;
            last_sub_group = sub_group;
            sub_groups++;
        }
        if (new_group || new_sub_group) {
            ++groups;
            group_offset_cpu_ptr_ = (int *)realloc(group_offset_cpu_ptr_, groups * sizeof(int));
            group_offset_cpu_ptr_[groups - 1] = n - group_size;
            group_size_cpu_ptr_ = (int *)realloc(group_size_cpu_ptr_, groups * sizeof(int));
            group_size_cpu_ptr_[groups - 1] = group_size;
            group_size = 0;
        }
        group_cpu_ptr_ = (int *)realloc(group_cpu_ptr_, (n + 1) * sizeof(int));
        group_cpu_ptr_[n] = groups;
        if (parent >= 0) {
            if (new_group) {
                CHECK_EQ(child_cpu_ptr_[parent], -1) << "node: " << n << " parent discontinuity in tree:" << filename;
                child_cpu_ptr_[parent] = groups;
            } else if (new_sub_group) {
                child_size_cpu_ptr_[parent] = sub_groups;
            }
        } else if (new_sub_group) {
            root_size_ = sub_groups;
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
    child_.Reshape(shape);
    child_.set_cpu_data(child_cpu_ptr_);
    child_size_.Reshape(shape);
    child_size_.set_cpu_data(child_size_cpu_ptr_);
}

void read_map(const char *filename, int max_label, Blob<int>& label_map) {
    int* label_map_cpu_ptr_ = NULL;
    char* str;
    FILE* file = fopen(filename, "r");
    CHECK(file) << "Cannot open the label map file: " << filename;
    int n = 0;
    while ((str = fgetl(file))) {
        auto label = atoi(str);
        CHECK_GE(label, 0) << "Invalid label: " << label << " in label map file:" << filename;
        CHECK_LT(label, max_label) << "Invalid label: " << label << " in label map file:" << filename;

        label_map_cpu_ptr_ = (int *)realloc(label_map_cpu_ptr_, (n + 1) * sizeof(int));
        label_map_cpu_ptr_[n] = label;
        n++;
    }

    CHECK_GT(n, 0) << "Label map file must be non-empty";
    label_map.Reshape({ n });
    label_map.set_cpu_data(label_map_cpu_ptr_);
}


}  // namespace caffe
