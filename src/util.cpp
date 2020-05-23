#include "util.h"
#include <opencv2/opencv.hpp>

void draw_gaussian(cv::Mat &input_hm, cv::Rect2f bbox)
{
    float height = bbox.height;
    float width = bbox.width;

    float min_overlap = 0.7;
    float a1 = 1;
    float b1 = height + width;
    float c1 = width * height * (1 - min_overlap) / (1 + min_overlap);
    float sq1 = sqrtf(powf(b1, 2) - 4 * a1 * c1);
    float r1 = (b1 + sq1) / 2;

    float a2 = 4;
    float b2 = 2 * (height + width);
    float c2 = (1 - min_overlap) * width * height;
    float sq2 = sqrtf(powf(b2, 2) - 4 * a2 * c2);
    float r2 = (b2 + sq2) / 2;

    float a3 = 4 * min_overlap;
    float b3 = -2 * min_overlap * (height + width);
    float c3 = (min_overlap - 1) * width * height;
    float sq3 = sqrtf(powf(b3, 2) - 4 * a3 * c3);
    float r3 = (b3 + sq3) / 2;
    float radius = std::min(r1, r2);
    radius = std::min(radius, r3);

    float diameter = radius * 2 + 1;
    float sigma = diameter * 6;
    cv::Mat gaussian_kernel = cv::getGaussianKernel(diameter, sigma);

    cv::Mat roi = input_hm(bbox);
    cv::MatExpr expr = cv::max(roi, gaussian_kernel);
    expr.op->assign(expr, roi);
}

cv::Scalar color_map(int64_t n)
{
    auto bit_get = [](int64_t x, int64_t i) { return x & (1 << i); };

    int64_t r = 0, g = 0, b = 0;
    int64_t i = n;
    for (int64_t j = 7; j >= 0; --j)
    {
        r |= bit_get(i, 0) << j;
        g |= bit_get(i, 1) << j;
        b |= bit_get(i, 2) << j;
        i >>= 3;
    }
    return cv::Scalar(b, g, r);
}

void draw_text(cv::Mat &img, const std::string &str,
               const cv::Scalar &color, cv::Point pos)
{
    auto t_size = cv::getTextSize(str, cv::FONT_HERSHEY_PLAIN, 1, 1, nullptr);
    cv::Point bottom_left, upper_right;

    bottom_left = pos;
    upper_right = cv::Point(bottom_left.x + t_size.width, bottom_left.y - t_size.height);

    cv::rectangle(img, bottom_left, upper_right, color, -1);
    cv::putText(img, str, bottom_left, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255) - color);
}

void draw_bbox(cv::Mat &img, cv::Rect2f bbox,
               const std::string &label,
               const cv::Scalar &color)
{
    // auto img_box = cv::Rect2f(bbox.x * img.cols,
    //                           bbox.y * img.rows,
    //                           bbox.width * img.cols,
    //                           bbox.height * img.rows);

    cv::rectangle(img, bbox, color);

    if (!label.empty())
    {
        draw_text(img, label, color, bbox.tl());
    }
}