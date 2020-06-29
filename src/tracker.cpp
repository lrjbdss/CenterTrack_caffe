#include "tracker.h"

Tracker::Tracker()
    : id_count(1)
{
}

void Tracker::reset()
{
    id_count = 1;
    tracks.clear();
}

// void Tracker::add_track(Track result)
// {
//     result.track_id = id_count;
//     tracks[id_count++] = result;
// }

float Tracker::cal_dist(cv::Point2f p1, cv::Point2f p2)
{
    float distance = powf((p1.x - p2.x), 2) + powf((p1.y - p2.y), 2);
    // return sqrtf(distance);
    return distance;
}

std::vector<Track> Tracker::step(std::vector<Track> dets)
{
    float first_thresh = 0.3; // 对首次出现的目标,设置更高的阈值,来增加跟踪的稳定性
    // for (auto &det : dets)
    for (auto iter = dets.begin(); iter != dets.end();)
    {
        cv::Point2f last_ct(iter->center);
        last_ct.x += iter->tracking[0];
        last_ct.y += iter->tracking[1];
        float det_size = iter->bbox.area();
        float min_dist = 1e18;
        int match_idx = -1;

        for (int i = 0; i < tracks.size(); ++i)
        {
            auto track = tracks[i];
            if (track.class_id != iter->class_id)
                continue;
            float track_size = track.bbox.area();

            float dist = cal_dist(track.center, last_ct);
            if (dist < track_size && dist < det_size && dist < min_dist)
            {
                min_dist = dist;
                match_idx = i;
            }
        }
        if (match_idx >= 0)
        {
            iter->track_id = tracks[match_idx].track_id;
            iter->active = tracks[match_idx].active + 1;
            ++iter;
            tracks.erase(tracks.begin() + match_idx);
        }
        else
        {
            if (iter->score > first_thresh)
            {
                iter->track_id = id_count++;
                iter->active = 1;
                ++iter;
            }
            else
                iter = dets.erase(iter);
        }
    }
    tracks = dets;
    return tracks;
}

cv::Mat Tracker::gen_pre_hm(int w, int h)
{
    cv::Mat pre_hm = cv::Mat(h, w, CV_32FC1) * 0;

    for (auto item : tracks)
    {
        cv::Rect2f bbox = item.bbox;
        draw_gaussian(pre_hm, bbox);
    }
    return pre_hm;
}