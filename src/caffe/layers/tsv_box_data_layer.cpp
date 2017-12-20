#define _USE_MATH_DEFINES
#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layers/tsv_box_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/random_helper.h"

#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>

using boost::property_tree::ptree;
using boost::property_tree::read_json;

namespace caffe {

struct image {
    int h;
    int w;
    int c;
    float *data;
};

struct box_label {
    int id;
    float x, y, w, h;
    float left, right, top, bottom;
};


image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w, h, c);
    out.data = new float[h*w*c];
    return out;
}

void free_image(image m)
{
    if (m.data) {
        delete[] m.data;
        m.data = 0;
    }
}

image cvmat_to_image(cv::Mat* src)
{
    int h = src->rows;
    int w = src->cols;
    int c = src->channels();
    image out = make_image(w, h, c);
    int i, j, k, count = 0;;

    for (k = 0; k < c; ++k) {
        for (i = 0; i < h; ++i) {
            const unsigned char* ptr = src->ptr<unsigned char>(i);
            for (j = 0; j < w; ++j) {
                out.data[count++] = ptr[j*c + k] / 255.;
            }
        }
    }
    return out;
}

void rgbgr_image(image im)
{
    int i;
    for (i = 0; i < im.w*im.h; ++i) {
        float swap = im.data[i];
        im.data[i] = im.data[i + im.w*im.h * 2];
        im.data[i + im.w*im.h * 2] = swap;
    }
}

float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}
float get_pixel_extend(image m, int x, int y, int c)
{
    if (x < 0) x = 0;
    if (x >= m.w) x = m.w - 1;
    if (y < 0) y = 0;
    if (y >= m.h) y = m.h - 1;
    if (c < 0 || c >= m.c) return 0;
    return get_pixel(m, x, y, c);
}
void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}
void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}

image resize_image(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < im.h; ++r) {
            for (c = 0; c < w; ++c) {
                float val = 0;
                if (c == w - 1 || im.w == 1) {
                    val = get_pixel(im, im.w - 1, r, k);
                }
                else {
                    float sx = c*w_scale;
                    int ix = (int)sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix + 1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < h; ++r) {
            float sy = r*h_scale;
            int iy = (int)sy;
            float dy = sy - iy;
            for (c = 0; c < w; ++c) {
                float val = (1 - dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if (r == h - 1 || im.h == 1) continue;
            for (c = 0; c < w; ++c) {
                float val = dy * get_pixel(part, c, iy + 1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}

image ipl_to_image(IplImage* src)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    image out = make_image(w, h, c);
    int i, j, k, count = 0;;

    for (k = 0; k < c; ++k) {
        for (i = 0; i < h; ++i) {
            for (j = 0; j < w; ++j) {
                out.data[count++] = data[i*step + j*c + k] / 255.;
            }
        }
    }
    return out;
}

// imbuf is image stream in memory
image load_image_cv(vector<unsigned char>& imbuf, int channels)
{
    int cv_read_flag = (channels == 3 ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat src = cv::imdecode(cv::Mat(imbuf), cv_read_flag);

    image out = cvmat_to_image(&src);
    // Here we directly work in the BGR format, not as in https://github.com/pjreddie/darknet/blob/master/src/image.c#L526
    // rgbgr_image(out);
    return out;
}

// imbuf is image stream in memory
image load_image_cv(const string &filename, int channels) 
{
    cv::Mat src = ReadImageToCVMat(filename, channels > 1);

    image out = cvmat_to_image(&src);
    // Here we directly work in the BGR format, not as in https://github.com/pjreddie/darknet/blob/master/src/image.c#L526
    // rgbgr_image(out);
    return out;
}

image load_image(vector<unsigned char>& imbuf, int w, int h, int c)
{
    image out = load_image_cv(imbuf, c);

    if ((h && w) && (h != out.h || w != out.w)) {
        image resized = resize_image(out, w, h);
        free_image(out);
        out = resized;
    }
    return out;
}

image load_image(const string &filename, int w, int h, int c) 
{
    image out = load_image_cv(filename, c);

    if ((h && w) && (h != out.h || w != out.w)) {
        image resized = resize_image(out, w, h);
        free_image(out);
        out = resized;
    }
    return out;
}

image load_image_color(vector<unsigned char>& imbuf, int w, int h)
{
    return load_image(imbuf, w, h, 3);
}

image load_image_color(const string &filename, int w, int h)
{
    return load_image(filename, w, h, 3);
}

void fill_image(image m, float s)
{
    int i;
    for (i = 0; i < m.h*m.w*m.c; ++i) m.data[i] = s;
}

float bilinear_interpolate(image im, float x, float y, int c)
{
    int ix = (int)floorf(x);
    int iy = (int)floorf(y);

    float dx = x - ix;
    float dy = y - iy;

    float val = (1 - dy) * (1 - dx) * get_pixel_extend(im, ix, iy, c) +
        dy     * (1 - dx) * get_pixel_extend(im, ix, iy + 1, c) +
        (1 - dy) *   dx   * get_pixel_extend(im, ix + 1, iy, c) +
        dy     *   dx   * get_pixel_extend(im, ix + 1, iy + 1, c);
    return val;
}

void original_image_to_network_input(float x3, float y3, 
        float cos_rad, float sin_rad,
        int orig_img_w, int orig_img_h, 
        int nw, int nh, 
        int network_w, int network_h,
        float dx, float dy,
        float &rotate_x3, float &rotate_y3) {
    float offset_x3 = x3 * nw / orig_img_w - 0.5 * nw;
    float offset_y3 = y3 * nh / orig_img_h - 0.5 * nh; 
    rotate_x3 = cos_rad * offset_x3 - sin_rad * offset_y3;
    rotate_y3 = sin_rad * offset_x3 + cos_rad * offset_y3;
    rotate_x3 += (float)nw / 2. + dx;
    rotate_y3 += (float)nh / 2. + dy;
}

void network_input_to_original_image(float in_x, float in_y,
        float cos_rad, float sin_rad,
        int orig_img_w, int orig_img_h,
        int nw, int nh,
        int network_w, int network_h,
        float dx, float dy,
        float &out_x, float &out_y) {
    float rx = in_x - 0.5 * nw - dx;
    float ry = in_y - 0.5 * nh - dy;
    out_x = cos_rad * rx + sin_rad * ry; 
    out_y = -sin_rad * rx + cos_rad * ry;
    out_x *= orig_img_w / (float)nw;
    out_y *= orig_img_h / (float)nh;
    out_x += orig_img_w * .5;
    out_y += orig_img_h * .5;
}

void place_image(image im, int w, int h, int dx, int dy, image canvas, float rad)
{
    if (rad) {
        const float fill_value = 0.5;
        float cos_rad = cos(rad);
        float sin_rad = sin(rad);
        for (int c = 0; c < im.c; c++) {
            for (int y1 = 0; y1 < canvas.h; y1++) {
                for (int x1 = 0; x1 < canvas.w; x1++) {
                    float m_x, m_y;
                    network_input_to_original_image((float)x1, (float)y1,
                            cos_rad, sin_rad,
                            im.w, im.h,
                            w, h,
                            canvas.w, canvas.h,
                            dx, dy,
                            m_x, m_y);
                    float val;
                    if (m_x < 0 || m_x >= im.w - 1 || m_y < 0 || m_y >= im.h - 1) {
                        val = fill_value;
                    } else {
                        val = bilinear_interpolate(im, m_x, m_y, c);
                    }
                    set_pixel(canvas, x1, y1, c, val);
                }
            }
        }
    } else {
        int x, y, c;
        for (c = 0; c < im.c; ++c) {
            for (y = 0; y < h; ++y) {
                for (x = 0; x < w; ++x) {
                    float rx = ((float)x / w) * im.w;
                    float ry = ((float)y / h) * im.h;
                    float val = bilinear_interpolate(im, rx, ry, c);
                    set_pixel(canvas, x + dx, y + dy, c, val);
                }
            }
        }
    }
}

float three_way_max(float a, float b, float c)
{
    return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
}

// http://www.cs.rit.edu/~ncs/color/t_convert.html
void rgb_to_hsv(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float h, s, v;
    for (j = 0; j < im.h; ++j) {
        for (i = 0; i < im.w; ++i) {
            r = get_pixel(im, i, j, 2);
            g = get_pixel(im, i, j, 1);
            b = get_pixel(im, i, j, 0);
            float max = three_way_max(r, g, b);
            float min = three_way_min(r, g, b);
            float delta = max - min;
            v = max;
            if (max == 0) {
                s = 0;
                h = 0;
            }
            else {
                s = delta / max;
                if (max == min) {
                    h = 0;
                }
                else if (r == max) {
                    h = (g - b) / delta;
                }
                else if (g == max) {
                    h = 2 + (b - r) / delta;
                }
                else {
                    h = 4 + (r - g) / delta;
                }
                if (h < 0) h += 6;
                h = h / 6.;
            }
            set_pixel(im, i, j, 0, h);
            set_pixel(im, i, j, 1, s);
            set_pixel(im, i, j, 2, v);
        }
    }
}

void hsv_to_rgb(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float h, s, v;
    float f, p, q, t;
    for (j = 0; j < im.h; ++j) {
        for (i = 0; i < im.w; ++i) {
            h = 6 * get_pixel(im, i, j, 0);
            s = get_pixel(im, i, j, 1);
            v = get_pixel(im, i, j, 2);
            if (s == 0) {
                r = g = b = v;
            }
            else {
                int index = floor(h);
                f = h - index;
                p = v*(1 - s);
                q = v*(1 - s*f);
                t = v*(1 - s*(1 - f));
                if (index == 0) {
                    r = v; g = t; b = p;
                }
                else if (index == 1) {
                    r = q; g = v; b = p;
                }
                else if (index == 2) {
                    r = p; g = v; b = t;
                }
                else if (index == 3) {
                    r = p; g = q; b = v;
                }
                else if (index == 4) {
                    r = t; g = p; b = v;
                }
                else {
                    r = v; g = p; b = q;
                }
            }
            set_pixel(im, i, j, 2, r);
            set_pixel(im, i, j, 1, g);
            set_pixel(im, i, j, 0, b);
        }
    }
}

void scale_image_channel(image im, int c, float v)
{
    int i, j;
    for (j = 0; j < im.h; ++j) {
        for (i = 0; i < im.w; ++i) {
            float pix = get_pixel(im, i, j, c);
            pix = pix*v;
            set_pixel(im, i, j, c, pix);
        }
    }
}

void scale_image(image im, float v)
{
    int i;
    for (i = 0; i < im.w*im.h*im.c; ++i) {
        im.data[i] = im.data[i]*v;
    }
}

void constrain_image(image im)
{
    int i;
    for (i = 0; i < im.w*im.h*im.c; ++i) {
        if (im.data[i] < 0) im.data[i] = 0;
        if (im.data[i] > 1) im.data[i] = 1;
    }
}

void distort_image(image im, float hue, float sat, float val)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    scale_image_channel(im, 2, val);
    int i;
    for (i = 0; i < im.w*im.h; ++i) {
        im.data[i] = im.data[i] + hue;
        if (im.data[i] > 1) im.data[i] -= 1;
        if (im.data[i] < 0) im.data[i] += 1;
    }
    hsv_to_rgb(im);
    constrain_image(im);
}

float rand_uniform(float min, float max)
{
    if (min == max) {
        return min;
    }
    if (max < min) {
        float swap = min;
        min = max;
        max = swap;
    }
    return (float)random_helper::uniform_real(min, max);
}

float rand_scale(float s)
{
    float scale = rand_uniform(1, s);
    if (random_helper::uniform_int(0, 1)) return scale;
    return 1. / scale;
}

void random_distort_image(image im, float hue, float saturation, float exposure)
{
    float dhue = rand_uniform(-hue, hue);
    float dsat = rand_scale(saturation);
    float dexp = rand_scale(exposure);
    distort_image(im, dhue, dsat, dexp);
}

void flip_image(image a)
{
    int i, j, k;
    for (k = 0; k < a.c; ++k) {
        for (i = 0; i < a.h; ++i) {
            for (j = 0; j < a.w / 2; ++j) {
                int index = j + a.w*(i + a.h*(k));
                int flip = (a.w - j - 1) + a.w*(i + a.h*(k));
                float swap = a.data[flip];
                a.data[flip] = a.data[index];
                a.data[index] = swap;
            }
        }
    }
}

void image_subtract_mean(image im, float r, float g, float b)
{
    float *r_channel, *g_channel, *b_channel;
    b_channel = im.data;
    g_channel = im.data + im.w * im.h;
    r_channel = im.data + im.w * im.h * 2;
    for (int i = 0; i < im.w * im.h; ++i) {
        b_channel[i] -= b;
        g_channel[i] -= g;
        r_channel[i] -= r;
    }
}

vector<box_label> read_boxes(const string &input_label_data, map<string, int> &labelmap, int orig_img_w, int orig_img_h)
{
    vector<box_label> boxes;
    if (input_label_data.length() == 0)
        return boxes;

    // Read json.
    ptree pt;
    std::istringstream is(input_label_data);
    read_json(is, pt);

    for (boost::property_tree::ptree::iterator it = pt.begin(); it != pt.end(); ++it)
    {
        boost::optional<int> diff = it->second.get_optional<int>("diff");         // 0: normal, 1: difficult
        if (diff && diff.get())
            continue;

        string cls = it->second.get<string>("class");
        CHECK(labelmap.find(cls) != labelmap.end()) << "class name not found in the labelmap: " << cls;
        std::vector<float> rect;
        ptree pt_rect = it->second.get_child("rect");   // format: x1, y1, x2, y2
        for (ptree::iterator iter = pt_rect.begin(); iter != pt_rect.end(); ++iter)
            rect.push_back(iter->second.get_value<float>());

        // https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py#L15
        // in yolo, x is decreased by 1 because coordinates in pascal voc are 1-based.
        // but in the tsv data, coordinates are already 0-based.
        float x = ((rect[0] + rect[2]) / 2) / orig_img_w;
        float y = ((rect[1] + rect[3]) / 2) / orig_img_h;
        float w = (rect[2] - rect[0]) / orig_img_w;
        float h = (rect[3] - rect[1]) / orig_img_h;

        box_label box;
        box.id = labelmap[cls];
        // https://github.com/pjreddie/darknet/blob/master/src/data.c#L149
        box.x = x;
        box.y = y;
        box.h = h;
        box.w = w;
        box.left = x - w / 2;
        box.right = x + w / 2;
        box.top = y - h / 2;
        box.bottom = y + h / 2;
        boxes.push_back(box);
    }

    return boxes;
}

void randomize_boxes(box_label *b, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        box_label swap = b[i];
        int index = rand() % n;
        b[i] = b[index];
        b[index] = swap;
    }
}

float constrain(float min, float max, float a)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip,
        float rad,
        int orig_img_w, int orig_img_h, 
        int nw, int nh,
        int network_w, int network_h, float odx, float ody)
{
    int i;
    float cos_rad = cos(rad);
    float sin_rad = sin(rad);
    for (i = 0; i < n; ++i) {
        if (boxes[i].x == 0 && boxes[i].y == 0) {
            boxes[i].x = 999999;
            boxes[i].y = 999999;
            boxes[i].w = 999999;
            boxes[i].h = 999999;
            continue;
        }
        if (rad) {
            float x0 = boxes[i].left * orig_img_w;
            float y0 = boxes[i].top * orig_img_h;
            float x3 = boxes[i].right * orig_img_w;
            float y3 = boxes[i].bottom * orig_img_h;
            float x1 = x3;
            float y1 = y0;
            float x2 = x0;
            float y2 = y3;
            float area_in_new = (boxes[i].right - boxes[i].left) * 
                (boxes[i].bottom - boxes[i].top) * nw * nh;
            original_image_to_network_input(x0, y0, cos_rad, sin_rad, orig_img_w, orig_img_h,
                    nw, nh, network_w, network_h,
                    odx, ody,
                    x0, y0);
            original_image_to_network_input(x1, y1, cos_rad, sin_rad, orig_img_w, orig_img_h,
                    nw, nh, network_w, network_h,
                    odx, ody,
                    x1, y1);
            original_image_to_network_input(x2, y2, cos_rad, sin_rad, orig_img_w, orig_img_h,
                    nw, nh, network_w, network_h,
                    odx, ody,
                    x2, y2);
            original_image_to_network_input(x3, y3, cos_rad, sin_rad, orig_img_w, orig_img_h,
                    nw, nh, network_w, network_h,
                    odx, ody,
                    x3, y3);
            float left = fmin(fmin(fmin(x0, x1), x2), x3);
            float right = fmax(fmax(fmax(x0, x1), x2), x3);
            float top = fmin(fmin(fmin(y0, y1), y2), y3);
            float bottom = fmax(fmax(fmax(y0, y1), y2), y3);
            {
                // make the area the same with the aspect ratio held
                float shrink = sqrt(area_in_new / (right - left) / (bottom - top));
                float final_width = (right - left) * shrink;
                float final_height = (bottom - top) * shrink;
                float final_cx = (left + right) / 2.;
                float final_cy = (top + bottom) / 2.;
                boxes[i].left = (final_cx - final_width / 2.) / network_w;
                boxes[i].right = (final_cx + final_width / 2.) / network_w;
                boxes[i].top = (final_cy - final_height / 2.) / network_h;
                boxes[i].bottom = (final_cy + final_height / 2.) / network_h;
                
            }
        } else {
            boxes[i].left = boxes[i].left  * sx - dx;
            boxes[i].right = boxes[i].right * sx - dx;
            boxes[i].top = boxes[i].top   * sy - dy;
            boxes[i].bottom = boxes[i].bottom* sy - dy;
        }

        if (flip) {
            float swap = boxes[i].left;
            boxes[i].left = 1. - boxes[i].right;
            boxes[i].right = 1. - swap;
        }

        boxes[i].left = constrain(0, 1, boxes[i].left);
        boxes[i].right = constrain(0, 1, boxes[i].right);
        boxes[i].top = constrain(0, 1, boxes[i].top);
        boxes[i].bottom = constrain(0, 1, boxes[i].bottom);

        boxes[i].x = (boxes[i].left + boxes[i].right) / 2;
        boxes[i].y = (boxes[i].top + boxes[i].bottom) / 2;
        boxes[i].w = (boxes[i].right - boxes[i].left);
        boxes[i].h = (boxes[i].bottom - boxes[i].top);

        boxes[i].w = constrain(0, 1, boxes[i].w);
        boxes[i].h = constrain(0, 1, boxes[i].h);
    }
}

void fill_truth_detection(const string &input_label_data, float *truth, int num_boxes, map<string, int> &labelmap, 
        int orig_img_w, int orig_img_h, int flip, float dx, float dy, float sx, float sy,
        float rad,
        int nw, int nh, int network_w, int network_h, float odx, float ody)
{
    vector<box_label> boxes = read_boxes(input_label_data, labelmap, orig_img_w, orig_img_h);
    auto count = boxes.size();
    if (!count)
        return;
    randomize_boxes(&boxes[0], count);
    correct_boxes(&boxes[0], count, dx, dy, sx, sy, flip, rad,
            orig_img_w, orig_img_h, nw, nh, network_w, network_h, odx, ody);
    if (count > num_boxes) 
        count = num_boxes;
    float x, y, w, h;
    int id;
    int i;

    for (i = 0; i < count; ++i) {
        x = boxes[i].x;
        y = boxes[i].y;
        w = boxes[i].w;
        h = boxes[i].h;
        id = boxes[i].id;

        if ((w < .001 || h < .001)) continue;

        truth[i * 5 + 0] = x;
        truth[i * 5 + 1] = y;
        truth[i * 5 + 2] = w;
        truth[i * 5 + 3] = h;
        truth[i * 5 + 4] = id;
    }
}

void load_data_detection(const string &input_b64coded_data, const string &input_label_data, float *output_image_data, float *output_label_data,
                         map<string, int> labelmap,
                         int w, int h, int boxes, float jitter, float hue, float saturation, float exposure,
                         float mean_r, float mean_g, float mean_b,
                         float pixel_value_scale,
                         const BoxDataParameter &box_param,
                         bool is_image_path)
{
    image orig;
    // load image in BGR format with values ranging in [0,1]
    if (is_image_path) {
        orig = load_image_color(input_b64coded_data, 0, 0);
    } else {
        vector<BYTE> imbuf = base64_decode(input_b64coded_data);
        orig = load_image_color(imbuf, 0, 0);
    }

    image sized = make_image(w, h, orig.c);
    fill_image(sized, .5);

    float dw = jitter * orig.w;
    float dh = jitter * orig.h;

    float new_ar = (orig.w + rand_uniform(-dw, dw)) / (orig.h + rand_uniform(-dh, dh));
    float random_scale_min = box_param.random_scale_min();
    float random_scale_max = box_param.random_scale_max();
    float scale = rand_uniform(random_scale_min, random_scale_max);

    float nw, nh;

    if (new_ar < 1) {
        nh = scale * h;
        nw = nh * new_ar;
    }
    else {
        nw = scale * w;
        nh = nw / new_ar;
    }

    float dx;
    float dy;
    if (box_param.fix_offset()) {
        dx = (w - nw) / 2;
        dy = (h - nh) / 2;
    } else {
        dx = rand_uniform(0, w - nw);
        dy = rand_uniform(0, h - nh);
    }

    float rad = rand_uniform(-box_param.rotate_max(),
            box_param.rotate_max());
    rad *= M_PI / 180.;

    place_image(orig, nw, nh, dx, dy, sized, rad);

    random_distort_image(sized, hue, saturation, exposure);
    int flip = random_helper::uniform_int(0, 1) && !box_param.fix_offset();
    if (flip) flip_image(sized);

    // scale values back to [0,255]
    scale_image(sized, pixel_value_scale);

    // mean subtraction
    image_subtract_mean(sized, mean_r, mean_g, mean_b);

    memcpy(output_image_data, sized.data, sizeof(float) * h * w * orig.c);
    free_image(sized);

    fill_truth_detection(input_label_data, output_label_data, boxes, labelmap, orig.w, orig.h, flip, 
            -dx / w, -dy / h, nw / w, nh / h,
            rad,
            nw, nh, sized.w, sized.h, dx, dy);
    free_image(orig);
}

template <typename Dtype>
void TsvBoxDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    TsvDataLayer<Dtype>::DataLayerSetUp(bottom, top);

    const BoxDataParameter &box_param = this->layer_param().box_data_param();

    iter_ = 0;
    dim_ = box_param.random_min();

    // reshape label
    vector<int> shape = top[1]->shape();
    iter_for_resize_ = box_param.iter_for_resize();
    CHECK(box_param.has_labelmap()) << "labelmap is expected in TsvBoxDataLayer";
    load_labelmap(box_param.labelmap());
    shape[1] = box_param.max_boxes() * 5;
    top[1]->Reshape(shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->label_.Reshape(shape);
    }
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
        std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
}

template <typename Dtype>
void TsvBoxDataLayer<Dtype>::load_labelmap(const string &filename)
{
    std::ifstream labelmap_file;
    labelmap_file.open(filename.c_str());
    CHECK(!labelmap_file.fail()) << "Failed to open labelmap file: " << filename << ", error: " << strerror(errno);

    int id = 0;
    labelmap_.clear();
    while (!labelmap_file.eof())
    {
        std::string line;
        std::getline(labelmap_file, line);
        rtrim(line);
        if (line.length() == 0)
            break;

        labelmap_[line] = id++;
    }

    labelmap_file.close();
}

template <typename Dtype>
void TsvBoxDataLayer<Dtype>::on_load_batch(Batch<Dtype>* batch)
{
    // change size
    if (iter_++ % this->iter_for_resize_ == 0)
    {
        const BoxDataParameter &box_param = this->layer_param().box_data_param();
        int rand_step = box_param.random_step();
        int rand_min = box_param.random_min() / rand_step;
        int rand_max = box_param.random_max() / rand_step;

        if (rand_min == rand_max)
        {
            dim_ = box_param.random_min();
        }
        else
        {
            dim_ = random_helper::uniform_int(rand_min, rand_max) * rand_step;
            LOG(INFO) << "Box data reshape to " << dim_ << "x" << dim_;
        }
    }
    vector<int> shape = batch->data_.shape();
    shape[2] = dim_;
    shape[3] = dim_;
    batch->data_.Reshape(shape);
}

template <typename Dtype>
void TsvBoxDataLayer<Dtype>::process_one_image_and_label(const string &input_b64coded_data, const string &input_label_data, const TsvDataParameter &tsv_param, Dtype *output_image_data, Dtype *output_label_data)
{
    const BoxDataParameter &box_param = this->layer_param().box_data_param();
    float jitter = box_param.jitter();
    float exposure = box_param.exposure();
    float hue = box_param.hue();
    float saturation = box_param.saturation();
    int max_boxes = box_param.max_boxes();
    float pixel_value_scale = this->layer_param().tsv_data_param().pixel_value_scale();
    
    load_data_detection(input_b64coded_data, input_label_data, (float*)output_image_data, (float*)output_label_data,
        labelmap_,
        dim_, dim_, max_boxes, jitter, hue, saturation, exposure,
        this->mean_values_[2], this->mean_values_[1], this->mean_values_[0], pixel_value_scale,
        box_param, 
        tsv_param.data_format() == TsvDataParameter_DataFormat_ImagePath);
}

INSTANTIATE_CLASS(TsvBoxDataLayer);
REGISTER_LAYER_CLASS(TsvBoxData);

}  // namespace caffe

