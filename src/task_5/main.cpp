#include <string>
#include <fstream>

#include <opencv2/opencv.hpp>

std::vector<cv::Rect> get_region_of_interest(const cv::Mat &original, cv::Scalar lover, cv::Scalar upper) {
    cv::Mat thresholdedMat;
    cv::cvtColor(original, thresholdedMat, cv::COLOR_BGR2HSV_FULL);
    cv::imshow("test", thresholdedMat);
    cv::inRange(thresholdedMat,lover,upper,thresholdedMat);
    cv::erode(thresholdedMat,thresholdedMat,
              cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    cv::dilate(thresholdedMat,thresholdedMat,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    cv::dilate(thresholdedMat,thresholdedMat,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    cv::erode(thresholdedMat,thresholdedMat,
              cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    cv::Canny(thresholdedMat, thresholdedMat, 100, 50, 5);

    std::vector<std::vector<cv::Point>> countours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(thresholdedMat,countours,hierarchy,cv::RETR_TREE,cv::CHAIN_APPROX_SIMPLE,
            cv::Point(0, 0));

    std::vector<cv::Rect> rects;
    for (uint i = 0; i < countours.size(); ++i) {
        if (0 <= hierarchy[i][3]) {
            continue;
        }
        rects.push_back(cv::boundingRect(countours[i]));
    }

    return rects;
}

void detect_text(std::string input) {
    cv::Mat original = cv::imread(input);
    cv::imshow("Source", original);

    auto red_regions = get_region_of_interest(original, cv::Scalar(0, 100, 100), cv::Scalar(5, 255, 255));
    for(const auto& red_region: red_regions) {
        cv::Mat red_roi = original(red_region);

        auto black_regions = get_region_of_interest(red_roi, cv::Scalar(0, 0, 0), cv::Scalar(140, 140, 60));
        for(const auto& black_region: black_regions) {
            cv::Mat black_roi = red_roi(black_region);
            cv::Mat small;
            cvtColor(black_roi, small, cv::COLOR_BGR2GRAY);

            cv::Mat grad;
            cv::Mat morphKernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::morphologyEx(small, grad, cv::MORPH_GRADIENT, morphKernel);

            cv::Mat bw;
            cv::threshold(grad, bw, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);

            cv::Mat connected;
            morphKernel = getStructuringElement(cv::MORPH_RECT, cv::Size(9, 1));
            cv::morphologyEx(bw, connected, cv::MORPH_CLOSE, morphKernel);

            cv::Mat mask = cv::Mat::zeros(bw.size(), CV_8UC1);
            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(connected, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

            if(hierarchy.empty()) continue;

            for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
                cv::Rect rect = boundingRect(contours[idx]);
                cv::Mat maskROI(mask, rect);
                drawContours(mask, contours, idx, cv::Scalar(255, 255, 255), cv::FILLED);
                double r = (double) countNonZero(maskROI) / (rect.width * rect.height);
                if (r > 0.45 && (rect.height > 8 && rect.width > 8)) {
                    rectangle(black_roi, rect, cv::Scalar(0, 255, 0), 2);
                }
            }
        }
    }
    cv::imshow("text", original);
    cv::waitKey(0);
}

int main() {
    detect_text(std::string("../test_marker.jpg"));

    return 0;
}