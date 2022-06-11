#include <tuple>
#include <vector>

#include "frame_matching.h"

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

static bool break_for_key(int delay) {
    int key = cv::waitKey(delay);

    switch (key) {
        case 27:
            return true;
        case 32:
            while (true) {
                if (cv::waitKey(delay) == 32) {
                    return false;
                } else if (cv::waitKey(delay) == 27) {
                    return true;
                }
            }
        default:
            return false;
    }
}

static std::tuple<cv::Mat, cv::Mat, std::vector<cv::KeyPoint>> grab_on_image(cv::VideoCapture &capture,
              cv::Ptr<cv::FeatureDetector> &detector,
              cv::Ptr<cv::DescriptorExtractor> &extractor,
              bool using_adaptive_alignment = false) {
    cv::Mat src;
    cv::Mat deser;
    std::vector<cv::KeyPoint> keys;

    capture.read(src);

    if(!src.empty()) {
        if (using_adaptive_alignment) {
            cv::cvtColor(src, src, cv::COLOR_RGB2GRAY);

            cv::Ptr<cv::CLAHE> pClahe = createCLAHE(80, cv::Size(5, 5));
            pClahe->setClipLimit(1);

            pClahe->apply(src, src);
        }

        detector->detect(src, keys);
        extractor->compute(src, keys, deser);
    }

    return std::make_tuple(src, deser, keys);
};

void FrameMatching(const std::string &path, bool using_adaptive_alignment) {
    cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::BRISK::create();
    cv::BFMatcher matcher;

    std::vector<cv::KeyPoint> keys;
    std::vector<cv::KeyPoint> prev_keys;

    cv::Mat src;
    cv::Mat prev_src;

    cv::Mat deser;
    cv::Mat prev_deser;

    cv::Mat image_with_points;

    cv::VideoCapture cap(path);
    if (!cap.isOpened()) throw std::invalid_argument("incorrect path: " + path);

    int delay = static_cast<int>(1000 / cap.get(cv::CAP_PROP_FPS));

    std::tie(prev_src, prev_deser, prev_keys) = grab_on_image(cap, detector, extractor, using_adaptive_alignment);

    while (true) {
        std::vector<cv::DMatch> matches;

        std::tie(src, deser, keys) = grab_on_image(cap, detector, extractor, using_adaptive_alignment);

        if (src.empty()) break;

        matcher.match(deser, prev_deser, matches);
        cv::drawMatches(src, keys, prev_src, prev_keys, matches, image_with_points);

        cv::imshow(typeid(cv::BRISK).name(), image_with_points);
        cv::imshow("Origin", src);

        prev_src = std::move(src);
        prev_keys = std::move(keys);
        prev_deser = std::move(deser);

        if (break_for_key(delay)) break;
    }

    cap.release();
}