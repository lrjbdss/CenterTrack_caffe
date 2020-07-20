#include "centerTrack.h"
#include "util.h"
#include <iostream>
#include <array>
// #include "caffe/caffe.hpp"

using boost::shared_ptr;
using caffe::Blob;
using caffe::Caffe;
using caffe::Layer;
using caffe::Net;

CenterTrack::CenterTrack(const std::string prototxtPath,
                         const std::string caffeModelPath,
                         float score,
                         bool add_pre_hm,
                         int classes,
                         int K,
                         std::array<float, 3> mean,
                         std::array<float, 3> std)
    : net_(new Net<float>(prototxtPath, caffe::Phase::TEST)),
      score_threshold(score),
      num_class(classes),
      topK(K),
      input_pre_hm(add_pre_hm),
      img_mean(mean),
      img_std(std),
      tracker()
{
    Caffe::set_mode(Caffe::GPU);
    net_->CopyTrainedLayersFrom(caffeModelPath);

    int num_inputs = 2;
    if (input_pre_hm)
        num_inputs = 3;
    CHECK_EQ(net_->num_inputs(), num_inputs) << "Network should have exactly %d input.", num_inputs;
    CHECK_EQ(net_->num_outputs(), 4) << "Network should have exactly 4 output.";
    Blob<float> *input_layer = net_->input_blobs()[0];
    inp_w = input_layer->width();
    inp_h = input_layer->height();
    // pre_hm = tracker.gen_pre_hm(inp_w, inp_h);
    // input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

cv::Mat CenterTrack::img_preprocess(cv::Mat img)
{
    if (img.size() != cv::Size(inp_w, inp_h))
    {
        ori_h = img.rows;
        ori_w = img.cols;

        cv::Point2f ori[3], inp[3];
        ori[0] = cv::Point2f(ori_w / 2., ori_h / 2.);
        inp[0] = cv::Point2f(inp_w / 2., inp_h / 2.);
        if (ori_w / float(ori_h) >= inp_w / float(inp_h))
        {
            float delta_h = (inp_h / float(inp_w) * ori_w - ori_h) / 2.;
            ori[1] = cv::Point2f(0., ori_h / 2.);
            inp[1] = cv::Point2f(0., inp_h / 2.);
            ori[2] = cv::Point2f(0., -delta_h);
            inp[2] = cv::Point2f(0., 0.);
        }
        else
        {
            float delta_w = (inp_w / float(inp_h) * ori_h - ori_w) / 2.;
            ori[1] = cv::Point2f(ori_w / 2., 0);
            inp[1] = cv::Point2f(inp_w / 2., 0);
            ori[2] = cv::Point2f(-delta_w, 0);
            inp[2] = cv::Point2f(0., 0.);
        }
        cv::Mat trans = cv::getAffineTransform(ori, inp);
        cv::Mat resized_img;
        cv::warpAffine(img, resized_img, trans, cv::Size2d(inp_w, inp_h));
        return resized_img;
    }
    return img;
}

std::vector<Track> CenterTrack::tracking(cv::Mat &img, cv::Mat input_pre_img)
{

    cv::Mat resized_img = img_preprocess(img);

    // cv::imshow("resized_img", resized_img);
    // cv::waitKey(0);

    std::vector<cv::Mat> img_channels;
    WrapInputLayer(&img_channels, net_->input_blobs()[0]);
    Preprocess(resized_img, img_channels);
    printf("***img_channels[0].at<float>(0, 0): %f\n", img_channels[0].at<float>(0, 0));
    printf("***resized_img.data[0, 0, 0]: %d, resized: %f\n",
           resized_img.data[0, 0, 0], (resized_img.data[0, 0, 0] / 255. - img_mean[0]) / img_std[0]);
    for (int i = 0; i < img_channels.size(); ++i)
        img_channels[i] = (img_channels[i] / 255. - img_mean[i]) / img_std[i];
    // img_channels[i] = (img_channels[i] - img_mean[i]) / img_std;
    printf("***img_channels[0].at<float>(0, 0): %f\n", img_channels[0].at<float>(0, 0));
    if (input_pre_img.data)
        pre_img = img_preprocess(input_pre_img);
    if (!pre_img.data)
        pre_img = resized_img;
    std::vector<cv::Mat> pre_img_channels;

    WrapInputLayer(&pre_img_channels, net_->input_blobs()[1]);
    Preprocess(pre_img, pre_img_channels);
    for (int i = 0; i < pre_img_channels.size(); ++i)
        pre_img_channels[i] = (pre_img_channels[i] / 255. - img_mean[i]) / img_std[i];
    // pre_img_channels[i] = (pre_img_channels[i] - img_mean[i]) / img_std;
    pre_img = resized_img;

    if (input_pre_hm)
    {
        std::vector<cv::Mat> pre_hm_channels;
        WrapInputLayer(&pre_hm_channels, net_->input_blobs()[2]);
        pre_hm = img_preprocess(pre_hm);
        Preprocess(pre_hm, pre_hm_channels);
    }

    net_->Forward();

    // *** test
    // Blob<float> *add_blob10 = net_->blob_by_name("add_blob10").get();
    // const float *add_blob10_data = add_blob10->cpu_data();

    std::vector<Track> dets = post_process();
    auto results = tracker.step(dets);
    // pre_hm = tracker.gen_pre_hm(ori_w, ori_h);
    return results;
}

std::vector<Track> CenterTrack::post_process()
{
    Blob<float> *hm = net_->blob_by_name("sigmoid_blob1").get();
    float *hm_dat = hm->mutable_cpu_data();
    // Blob<float> *hm_pool = net_->blob_by_name("max_pool_blob2").get();
    // const float *hm_pool_data = hm_pool->cpu_data();
    // int hm_w = int(inp_w / 4);
    // int hm_h = int(inp_h / 4);
    int hm_w = 88; // 本来应该320下采样4倍变成80,因为caffe的pooling只有ceil_mode,
    int hm_h = 48; // 结果变成了88,与训练时有差异
    CHECK_EQ(hm->channels(), num_class) << "output hm channel should equal to num_class.";
    CHECK_EQ(hm->width(), hm_w) << "output hm width should have equal to input_width/4.";
    CHECK_EQ(hm->height(), hm_h) << "output hm height should have equal to input_height/4.";

    // nms
    int row, col;
    float hm_data[num_class * hm_w * hm_h] = {0};
    for (int i = 0; i < num_class * hm_w * hm_h; ++i)
    {
        // if (hm_data[i] != hm_pool_data[i])
        //     hm_data[i] = 0.;
        row = i / hm_w % hm_h;
        col = i % hm_w;
        if (col > 0)
        {
            // left
            if (hm_dat[i] < hm_dat[i - 1])
                continue;
        }
        if (col < hm_w - 1)
        {
            // right
            if (hm_dat[i] <= hm_dat[i + 1])
                continue;
        }

        if (row > 0)
        {
            // up
            if (hm_dat[i] < hm_dat[i - hm_w])
                continue;

            // up-left
            if (col > 0)
            {
                if (hm_dat[i] < hm_dat[i - hm_w - 1])
                    continue;
            }
            // up-right
            if (col < hm_h - 1)
            {
                if (hm_dat[i] <= hm_dat[i - hm_w + 1])
                    continue;
            }
        }

        if (row < hm_h - 1)
        {
            // down
            if (hm_dat[i] <= hm_dat[i + hm_w])
                continue;

            // down-left
            if (col > 0)
            {
                if (hm_dat[i] < hm_dat[i + hm_w - 1])
                    continue;
            }
            // down-right
            if (col < hm_w - 1)
            {
                if (hm_dat[i] <= hm_dat[i + hm_w + 1])
                    continue;
            }
        }
        hm_data[i] = hm_dat[i];
    }

    const std::vector<float> hm_vec(hm_data, hm_data + num_class * hm_w * hm_h);
    std::vector<int> max_index = sort_index(hm_vec);

    int channel_size = hm_w * hm_h;

    Blob<float> *reg = net_->blob_by_name("conv_blob25").get();
    const float *reg_data = reg->cpu_data();

    Blob<float> *wh = net_->blob_by_name("conv_blob27").get();
    const float *wh_data = wh->cpu_data();

    Blob<float> *track = net_->blob_by_name("conv_blob29").get();
    const float *track_data = track->cpu_data();

    // 下面三个变量用来坐标转换
    float scale = std::max(ori_w / 80., ori_h / 45.);
    // float delta_w = 0.;
    // float delta_h = 95.;
    float delta_w = (scale * 80 - ori_w) / 2.;
    float delta_h = (scale * 45 - ori_h) / 2.;

    std::vector<float> topK_score;
    std::vector<float> topK_x, topK_y;
    std::vector<int> topK_class;
    std::vector<float> reg_x, reg_y;
    std::vector<float> wh_x, wh_y;
    std::vector<float> track_x, track_y;
    std::vector<cv::Rect2f> topK_bbox;
    int obj_count = 0;
    for (; obj_count < topK; ++obj_count)
    {
        int idx = max_index[obj_count];
        if (hm_vec[idx] < score_threshold)
            break;
        topK_score.push_back(hm_vec[idx]);
        topK_class.push_back(int(idx / channel_size));
        idx %= channel_size;
        topK_y.push_back(idx / hm_w * scale - delta_h);
        topK_x.push_back(idx % hm_w * scale - delta_w);
        reg_x.push_back(reg_data[idx] * scale);
        reg_y.push_back(reg_data[idx + channel_size] * scale);
        wh_x.push_back(wh_data[idx] * scale);
        wh_y.push_back(wh_data[idx + channel_size] * scale);
        track_x.push_back(track_data[idx] * scale);
        track_y.push_back(track_data[idx + channel_size] * scale);
        topK_bbox.push_back(cv::Rect2f(topK_x[obj_count] + reg_x[obj_count] - wh_x[obj_count] / 2.,
                                       topK_y[obj_count] + reg_y[obj_count] - wh_y[obj_count] / 2.,
                                       wh_x[obj_count], wh_y[obj_count]));
    }
    // printf("obj_count: %d\n", obj_count);
    std::vector<Track> dets;
    for (int i = 0; i < obj_count; ++i)
    {
        Track item;
        item.score = topK_score[i];
        item.class_id = topK_class[i];
        item.center = cv::Point2i(topK_x[i], topK_y[i]);
        item.tracking = cv::Vec2f(track_x[i], track_y[i]);
        item.bbox = topK_bbox[i];
        dets.push_back(item);
    }
    return dets;
}

void CenterTrack::WrapInputLayer(std::vector<cv::Mat> *input_channels,
                                 Blob<float> *input_layer)
{
    // Blob<float> *input_layer = net_->input_blobs()[0];

    float *input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat channel(inp_h, inp_w, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += inp_w * inp_h;
    }
}

void CenterTrack::Preprocess(const cv::Mat &img,
                             std::vector<cv::Mat> &input_channels)
{
    cv::Mat sample_resized;
    if (img.size() != cv::Size(inp_w, inp_h))
    {
        printf("img size: (%d, %d)\n", img.cols, img.rows);
        cv::resize(img, sample_resized, cv::Size(inp_w, inp_h));
    }
    else
        sample_resized = img;

    cv::Mat sample_float;
    if (img.channels() == 3)
    {
        sample_resized.convertTo(sample_float, CV_32FC3);
    }
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
    cv::split(sample_float, input_channels);
}
