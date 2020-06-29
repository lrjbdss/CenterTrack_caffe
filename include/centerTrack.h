#pragma once
#include "caffe/caffe.hpp"
// #include "util.h"
#include "tracker.h"
#include <string>
#include <vector>

class CenterTrack
{
public:
    CenterTrack(const std::string prototxtPath,
                const std::string caffeModelPath,
                float score_threshold = 0.3,
                bool input_pre_hm = false,
                int num_class = 1,
                int topK = 25,
                // std::array<float, 3> mean = {105.9422484375, 105.21612951388889, 105.52638559027778},
                // float std = 78.21761860831944);
                std::array<float, 3> mean = {0.40789655, 0.44719303, 0.47026116},
                std::array<float, 3> std = {0.279, 0.279, 0.279});

    std::vector<Track> tracking(cv::Mat &img, cv::Mat pre_img);

private:
    void WrapInputLayer(std::vector<cv::Mat> *input_channels,
                        caffe::Blob<float> *input_layer);

    void Preprocess(const cv::Mat &img,
                    std::vector<cv::Mat> &input_channels);
    cv::Mat img_preprocess(cv::Mat img);
    std::vector<Track> post_process();

private:
    std::array<float, 3> img_mean;
    std::array<float, 3> img_std;
    cv::Mat pre_img;
    cv::Mat pre_hm;
    bool input_pre_hm;
    std::shared_ptr<caffe::Net<float>> net_;
    float score_threshold;
    int inp_w, inp_h, ori_w, ori_h;
    int num_class;
    int topK;
    Tracker tracker;
};
