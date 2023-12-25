#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>

int main() {
    // Chemins vers les fichiers YOLO
    std::string weightsPath = "yolov4.weights";
    std::string configPath = "yolov4.cfg";
    std::string classesFile = "coco.names";

    std::vector<std::string> classes;
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (getline(ifs, line)) classes.push_back(line);

    cv::dnn::Net net = cv::dnn::readNet(configPath, weightsPath);
    
    cv::Mat image = cv::imread("example001.png");
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1/255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;


    for (size_t i = 0; i < outputs.size(); ++i) {
        float* data = (float*)outputs[i].data;
        for (int j = 0; j < outputs[i].rows; ++j, data += outputs[i].cols) {
            cv::Mat scores = outputs[i].row(j).colRange(5, outputs[i].cols);
            cv::Point classIdPoint;
            double confidence;

            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > 0.5) {
                int centerX = (int)(data[0] * image.cols);
                int centerY = (int)(data[1] * image.rows);
                int width = (int)(data[2] * image.cols);
                int height = (int)(data[3] * image.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 3); 
        std::string label = classes[classIds[idx]] + ": " + cv::format("%.2f", confidences[idx]);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(box.y, labelSize.height);
        cv::rectangle(image, cv::Point(box.x, top - labelSize.height),
                      cv::Point(box.x + labelSize.width, top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(image, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
    }

    cv::imwrite("result_detection9+.png", image);

    return 0;
}
