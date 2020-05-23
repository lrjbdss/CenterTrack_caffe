#include "centerTrack.h"
// #include "util.h"
#include <array>

using namespace std;

int main(int argc, const char *argv[])
{

    auto input_path = string("/home/lx/work/centerTrack_SDK/videos/10.avi");

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened())
    {
        throw runtime_error("Cannot open cv::VideoCapture");
    }

    std::string prototxtPath = "/home/lx/work/centerTrack_SDK/model/res18_centerTrack.prototxt";
    std::string caffeModelPath = "/home/lx/work/centerTrack_SDK/model/res18_centerTrack.caffemodel";
    CenterTrack centerTrack(prototxtPath, caffeModelPath);
    // Tracker tracker = Tracker();

    cv::Size orig_dim(int(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    cv::VideoWriter outputCap("../output_video/res18_centerTrack.avi",
                              CV_FOURCC('M', 'P', '4', '2'), 25.0, orig_dim);
    auto image = cv::Mat();
    while (cap.read(image))
    // while (true)
    {
        // image = cv::imread("/media/lx/5f5da3ff-2264-4479-82cc-7053cf09f640/datasets/vehicle_type_data/convert_to_coco/image/0-2017-10-23-11-01-13-323-01-300972.jpg");
        auto results = centerTrack.tracking(image, cv::Mat());

        for (auto track : results)
        {
            draw_bbox(image, track.bbox, to_string(track.track_id), color_map(track.track_id));
        }

        cv::imshow("image", image);
        cv::waitKey(1);

        outputCap << image;
    }
    outputCap.release();

    return 0;
}
