#pragma once
#include <opencv2/opencv.hpp>
#include <array>
#include <vector>
#include <map>
#include <math.h>
#include "util.h"

class Tracker
{
public:
    Tracker();

    void reset();

    std::vector<Track> step(std::vector<Track> results);

    cv::Mat gen_pre_hm(int w, int h);

private:
    int id_count;
    std::vector<Track> tracks;

    // void add_track(Track det);
    float cal_dist(cv::Point2f p1, cv::Point2f p2);
};