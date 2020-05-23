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

// std::map<int, Track> Tracker::step(std::vector<Track> dets)
// {
//     auto iter = tracks.begin();
//     while (iter != tracks.end())
//     {
//         auto track_ct = iter->second.center;
//         float track_size = iter->second.bbox.area();
//         float min_dist = 1e18;
//         int match_idx = -1;
//         // for (const auto &det : dets)
//         for (int i = 0; i < dets.size(); ++i)
//         {
//             auto det = dets[i];
//             if (det.class_id != iter->second.class_id)
//                 continue;
//             float det_size = det.bbox.area();
//             cv::Point2f last_ct(det.center);
//             last_ct.x += det.tracking[0];
//             last_ct.y += det.tracking[1];
//             float dist = cal_dist(track_ct, last_ct);
//             if (dist < track_size && dist < det_size && dist < min_dist)
//             {
//                 min_dist = dist;
//                 match_idx = i;
//             }
//         }
//         if (match_idx >= 0)
//         {
//             iter->second.update(dets[match_idx]);
//             dets.erase(dets.begin() + match_idx);
//             ++iter;
//         }
//         else
//         {
//             iter = tracks.erase(iter);
//         }
//     }
//     for (auto &det : dets)
//     {
//         add_track(det);
//     }

//     return tracks;
// }

std::vector<Track> Tracker::step(std::vector<Track> dets)
{

    for (auto &det : dets)
    {
        cv::Point2f last_ct(det.center);
        last_ct.x += det.tracking[0];
        last_ct.y += det.tracking[1];
        float det_size = det.bbox.area();
        float min_dist = 1e18;
        int match_idx = -1;

        for (int i = 0; i < tracks.size(); ++i)
        {
            auto track = tracks[i];
            if (track.class_id != det.class_id)
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
            det.track_id = tracks[match_idx].track_id;
            det.active = tracks[match_idx].active + 1;
            tracks.erase(tracks.begin() + match_idx);
        }
        else
        {
            det.track_id = id_count++;
        }
    }
    tracks = dets;
    return tracks;
}

cv::Mat Tracker::gen_pre_hm(int w, int h)
{
    cv::Mat pre_hm = cv::Mat(w, h, CV_32FC1) * 0;

    for (auto item : tracks)
    {
        cv::Rect2f bbox = item.bbox;
        draw_gaussian(pre_hm, bbox);
    }
    return pre_hm;
}