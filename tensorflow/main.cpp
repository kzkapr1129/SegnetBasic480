#include "Detector.h"
#include <opencv2/opencv.hpp>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s img_path\n", argv[0]);
        return 0;
    }

    cv::Mat img = cv::imread(argv[1]);
    if (img.empty()) {
        printf("couldn't read %s\n", argv[1]);
        return 0;
    }

    Detector detector(480, 480, 2);
    detector.open("python/frozen_graph.pb");
    detector.setClassColor(1, cv::Vec3b(0, 255, 0));

    cv::Point pos;
    cv::Mat* output;

    // 推論の開始
    auto start = std::chrono::system_clock::now();
    detector.predict(img, pos, output);
    auto end = std::chrono::system_clock::now();

    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    printf("elepsed: %lf msec\n", elapsed);

    cv::imshow("frame", img);
    cv::imshow("output", *output);
    cv::waitKey();
    return 0;
}