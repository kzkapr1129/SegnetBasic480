#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>

class Detector {
public:
    Detector(int img_width, int img_height, int num_classes);
    ~Detector();

    bool open(const std::string& frozen_graph_filename);
    void release();

    void setClassColor(int id, const cv::Vec3b& color);

    bool predict(const cv::Mat& img, cv::Point& pos, cv::Mat*& output);

private:
    void preprocess(const cv::Mat& img);
    static TF_Buffer* read_file(const std::string& filename);

    const int INPUT_IMG_HEIGHT;
    const int INPUT_IMG_WIDTH;
    const int NUM_CLASSES;
    std::vector<cv::Vec3b> mClassColors;
    TF_Graph* mGraph;
    TF_Session* mSession;
    TF_Status* mStatus;
    cv::Mat mInput;
    cv::Mat mOutput;
    cv::Mat mMean;
};