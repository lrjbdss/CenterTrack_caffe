#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>

cv::Scalar color_map(int64_t n);
// {
//     auto bit_get = [](int64_t x, int64_t i) { return x & (1 << i); };

//     int64_t r = 0, g = 0, b = 0;
//     int64_t i = n;
//     for (int64_t j = 7; j >= 0; --j)
//     {
//         r |= bit_get(i, 0) << j;
//         g |= bit_get(i, 1) << j;
//         b |= bit_get(i, 2) << j;
//         i >>= 3;
//     }
//     return cv::Scalar(b, g, r);
// }

void draw_text(cv::Mat &img, const std::string &str,
               const cv::Scalar &color, cv::Point pos);
// {
//     auto t_size = cv::getTextSize(str, cv::FONT_HERSHEY_PLAIN, 1, 1, nullptr);
//     cv::Point bottom_left, upper_right;

//     bottom_left = pos;
//     upper_right = cv::Point(bottom_left.x + t_size.width, bottom_left.y - t_size.height);

//     cv::rectangle(img, bottom_left, upper_right, color, -1);
//     cv::putText(img, str, bottom_left, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255) - color);
// }

void draw_bbox(cv::Mat &img, cv::Rect2f bbox,
               const std::string &label = "",
               const cv::Scalar &color = {0, 0, 0});
// {
//     auto img_box = cv::Rect2f(bbox.x * img.cols,
//                               bbox.y * img.rows,
//                               bbox.width * img.cols,
//                               bbox.height * img.rows);

//     cv::rectangle(img, img_box, color);

//     if (!label.empty())
//     {
//         draw_text(img, label, color, img_box.tl());
//     }
// }

template <typename T>
std::vector<int> sort_index(const std::vector<T> &v)
{
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) { return v[i1] > v[i2]; });
    return idx;
}

void draw_gaussian(cv::Mat &input_hm, cv::Rect2f bbox);