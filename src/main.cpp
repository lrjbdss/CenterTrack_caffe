#include "centerTrack.h"
// #include "util.h"
#include <array>

using namespace std;

int main(int argc, const char *argv[])
{

    auto input_path = string("/home/lx/work/hi3519/nfs100.69/centerTrack/data/500.avi");

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened())
    {
        throw runtime_error("Cannot open cv::VideoCapture");
    }

    std::string prototxtPath = "/home/lx/work/centerTrack_SDK/model/mosaic1_1533.prototxt";
    std::string caffeModelPath = "/home/lx/work/centerTrack_SDK/model/mosaic1_1533.caffemodel";
    CenterTrack centerTrack(prototxtPath, caffeModelPath);
    // Tracker tracker = Tracker();

    cv::Size orig_dim(int(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    cv::VideoWriter outputCap("../output_video/res18_centerTrack.avi",
                              CV_FOURCC('M', 'P', '4', '2'), 25.0, orig_dim);
    auto image = cv::Mat();
    int frame_count = 0;
    while (cap.read(image))
    // while (true)
    {
        // image = cv::imread("/media/lx/5f5da3ff-2264-4479-82cc-7053cf09f640/datasets/vehicle_type_data/convert_to_coco/image/0-2017-10-23-11-01-13-323-01-300972.jpg");
        auto results = centerTrack.tracking(image, cv::Mat());

        for (auto track : results)
        {
            draw_result(image, track);
        }
        cv::putText(image, std::to_string(frame_count++), cv::Point(0, 20), CV_FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 0));
        outputCap << image;
        // cv::imshow("image", image);
        // if (cv::waitKey(0) == 27)
        //     break;
    }
    outputCap.release();

    return 0;
}
