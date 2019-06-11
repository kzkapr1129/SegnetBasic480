#include "Detector.h"

static void free_buffer(void* data, size_t /*length*/) {
    free(data);
}

static void nonfree_dealloc_tensor(void* /*data*/, size_t /*len*/, void* /*arg*/) {
    // IGNORE
}

Detector::Detector(int img_width, int img_height, int num_classes)
        : NUM_CLASSES(num_classes), INPUT_IMG_HEIGHT(img_height), INPUT_IMG_WIDTH(img_width),
        mGraph(NULL), mSession(NULL), mStatus(NULL) {
    mClassColors.assign(NUM_CLASSES, cv::Vec3b(0, 0, 0));
}

Detector::~Detector() {
    release();
}

bool Detector::open(const std::string& filename) {
    TF_SessionOptions* sess_opts = NULL;
    TF_ImportGraphDefOptions* graph_opts = NULL;
    TF_Buffer* graph_def = NULL;
    bool success = false;

    release();

    mGraph = TF_NewGraph();
    mStatus = TF_NewStatus();
    if (mGraph == NULL || mStatus == NULL) {
        printf("failed to allocate mGraph or mStatus\n");
        goto release;
    }

    graph_opts = TF_NewImportGraphDefOptions();
    if (graph_opts == NULL) {
        printf("failed to allocate graph_opts\n");
        goto release;
    }

    graph_def = read_file(filename.c_str());
    if (graph_def == NULL) {
        goto release;
    }

    TF_GraphImportGraphDef(mGraph, graph_def, graph_opts, mStatus);
    if (TF_GetCode(mStatus) != TF_OK) {
        printf("Unable to import graph: %s\n", TF_Message(mStatus));
        goto release;
    }

    sess_opts = TF_NewSessionOptions();
    if (sess_opts == NULL) {
        printf("failed to allocate sess option\n");
        goto release;
    }

    mSession = TF_NewSession(mGraph, sess_opts, mStatus);
    if (TF_GetCode(mStatus) != TF_OK) {
        printf("Unable to create session: %s\n", TF_Message(mStatus));
        goto release;
    }

    success = true;
    printf("TF API Version: %s\n", TF_Version());

release:
    TF_DeleteSessionOptions(sess_opts);
    TF_DeleteBuffer(graph_def);
    TF_DeleteImportGraphDefOptions(graph_opts);

    if (!success) {
        release();
    }

    return success;
}

void Detector::release() {
    if (mSession) {
        TF_CloseSession(mSession, mStatus);
        TF_DeleteSession(mSession, mStatus);
        mSession = NULL;
    }

    if (mGraph) {
        TF_DeleteGraph(mGraph);
        mGraph = NULL;
    }

    if (mStatus) {
        TF_DeleteStatus(mStatus);
        mStatus = NULL;
    }
}

void Detector::setClassColor(int id, const cv::Vec3b& color) {
    mClassColors[id] = color;
}

bool Detector::predict(const cv::Mat& img, cv::Point& pos, cv::Mat*& output) {
    output = NULL;
    preprocess(img);

    const int64_t input_dims[4] = {1, INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, 3};
    const size_t input_len = 1 * INPUT_IMG_HEIGHT * INPUT_IMG_WIDTH * 3 * sizeof(float);
    std::vector<float> input_buffer;
    input_buffer.resize(INPUT_IMG_HEIGHT * INPUT_IMG_WIDTH * 3);

    TF_Tensor* input_tensor = NULL;
    if (mInput.isContinuous()) {
        float* pixels = (float*)mInput.ptr();
        input_tensor = TF_NewTensor(
            TF_FLOAT,
            input_dims, sizeof(input_dims) / sizeof(input_dims[0]),
            reinterpret_cast<void*>(pixels), input_len,
            nonfree_dealloc_tensor, nullptr);
    } else {
        printf("No supported yet: mInput wasn't continuous");
        return false;
    }

    if (input_tensor == NULL) {
        printf("failed to allocate input_tensor\n");
        return false;
    }

    TF_Output input_op = {TF_GraphOperationByName(mGraph, "input"), 0};
    TF_Output output_op = {TF_GraphOperationByName(mGraph, "output"), 0};
    TF_Tensor* output_tensor = nullptr;

    TF_SessionRun(
        mSession,
        nullptr,
        &input_op, &input_tensor, 1,
        &output_op, &output_tensor, 1,
        nullptr, 0,
        nullptr,
        mStatus
    );

    if (TF_GetCode(mStatus) != TF_OK) {
        fprintf(stderr, "failed TF_SessionRun %s\n", TF_Message(mStatus));
    }

    printf("byte size=%ld\n", TF_TensorByteSize(output_tensor));

    float* output_ptr = static_cast<float*>(TF_TensorData(output_tensor));

    if (mOutput.size() != cv::Size(INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT)) {
        mOutput = cv::Mat::zeros(INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, CV_8UC3);
    }

    for (int y = 0; y < mInput.rows; y++) {
        for (int x = 0; x < mInput.cols; x++) {
            int basei = (y * mInput.cols + x) * NUM_CLASSES;

            float channels[NUM_CLASSES];
            float max = -1;
            int max_class = -1;
            for (int c = 0; c < NUM_CLASSES; c++) {
                float val = output_ptr[basei + c];
                if (max < val) {
                    max = val;
                    max_class = c;
                }
            }

            if (0 <= max_class && max_class < mClassColors.size()) {
                mOutput.at<cv::Vec3b>(y,x) = mClassColors[max_class];
            }
        }
    }

    output = &mOutput;
    return true;
}

void Detector::preprocess(const cv::Mat& img) {
    cv::Mat sample;
    if (img.channels() == 1) {
        cv::cvtColor(img, sample, cv::COLOR_GRAY2RGB);
    } else if (img.channels() == 3) {
        cv::cvtColor(img, sample, cv::COLOR_BGR2RGB);
    } else {
        sample = img;
    }

    cv::Mat sample_resized;
    const cv::Size& img_size = img.size();
    if (img_size.width != INPUT_IMG_WIDTH || img_size.height != INPUT_IMG_HEIGHT) {
        cv::resize(sample, sample_resized, cv::Size(INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT));
    } else {
        sample_resized = sample;
    }

    cv::Mat sample_float;
    sample_resized.convertTo(sample_float, CV_32FC3);

    if (mMean.size() != cv::Size(INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT)) {
        mMean = cv::Mat::zeros(INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, CV_32FC3);
        mMean = cv::Scalar(128, 128, 128);
    }

    cv::Mat sample_norm;
    cv::subtract(sample_float, mMean, mInput);
    mInput *= 0.0078125f;
}

TF_Buffer* Detector::read_file(const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        printf("couldn't open %s\n", filename.c_str());
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (fsize < 16) {
        printf("invalid file size: %ld\n", fsize);
        fclose(fp);
        return NULL;
    }

    void* data = malloc(fsize);
    size_t n = fread(data, fsize, 1, fp);
    fclose(fp);

    if (n != 1) {
        printf("fread err n=%lu\n", n);
        free(data);
        return NULL;
    }

    TF_Buffer* buf = TF_NewBuffer();
    if (buf == NULL) {
        printf("failed to alloc TF_BUFFER\n");
        free(data);
        return NULL;
    }

    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = free_buffer;
    return buf;
}