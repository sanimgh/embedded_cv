#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/ioctl.h>
#include <unistd.h>
#include <linux/fb.h>
#include <linux/kd.h>
#include <fcntl.h>
#include <unordered_set>


struct framebuffer_info
{
    uint32_t bits_per_pixel;    // framebuffer depth
    uint32_t xres_virtual;      // how many pixel in a row in virtual screen
      uint32_t xres;
};

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);

int main() {
    // Chemins vers les fichiers YOLO
    std::string weightsPath = "yolov3-tiny.weights";
    std::string configPath = "yolov3-tiny.cfg";
    std::string classesFile = "coco.names";

    std::vector<std::string> classes;
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (getline(ifs, line)) classes.push_back(line);

    cv::dnn::Net net = cv::dnn::readNet(configPath, weightsPath);
    std::vector<std::string> targeted_classes = {"mouse", "keyboard", "monitor"};
    std::unordered_set<std::string> target_set(targeted_classes.begin(), targeted_classes.end());
    
   cv::VideoCapture camera_stream(2);
    if(!camera_stream.isOpened())
    {
	printf("camera pas ouverte\n");
	return -1;
    }
    camera_stream.set(cv::CAP_PROP_FPS, 2);
    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
    std::ofstream ofs("/dev/fb0");
    
    cv::Mat image;
    cv::Size2f image_size;
    std::cout << fb_info.xres_virtual << std::endl;
    //std::cout << fb_info.xres << std::endl;

    bool isProcessing = false;

    while(1)
      {

	    camera_stream >> image;
	    auto start = std::chrono::high_resolution_clock::now();
	    cv::Mat blob;
	    cv::dnn::blobFromImage(image, blob, 1/255.0, cv::Size(320, 320), cv::Scalar(0, 0, 0), true, false);
	    net.setInput(blob);
	    std::vector<cv::Mat> outputs;
	    net.forward(outputs, net.getUnconnectedOutLayersNames());
	    auto end = std::chrono::high_resolution_clock::now();
	    std::cout << "model processing : ";
	    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	    std::cout << "ms" << std::endl;

	        start = std::chrono::high_resolution_clock::now();


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
		if (confidence > 0.5 && target_set.find(classes[classIdPoint.x]) != target_set.end()) {
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
	      cv::rectangle(image, cv::Point(box.x, top - labelSize.height),cv::Point(box.x + labelSize.width, top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
	      cv::putText(image, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
	    }
	    cv::cvtColor(image,image,cv::COLOR_BGR2BGR565);
	    image_size = image.size();
	    //std::cout << image_size.width << std::endl;
	    int x_offset =0;
	    //std::cout << x_offset << std::endl;
	    for (int y = 0; y < image_size.height; y++)
	      {
		ofs.seekp(y * (fb_info.xres_virtual * fb_info.bits_per_pixel / 8)+ (x_offset*fb_info.bits_per_pixel/8));	
		ofs.write(reinterpret_cast<char*>(image.ptr(y)),image_size.width*(fb_info.bits_per_pixel/8));
	      }
	    end = std::chrono::high_resolution_clock::now();
	    std::cout << "post processing & displaying : ";
	    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	    std::cout << "ms" << std::endl;
	  
      }
    return 0;
}

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path)
{
    struct framebuffer_info fb_info;        // Used to return the required attrs.
    struct fb_var_screeninfo screen_info;   // Used to get attributes of the device from OS kernel.

	int fd = open(framebuffer_device_path, O_RDWR);

	if(ioctl(fd,FBIOGET_VSCREENINFO,&screen_info)<0)
	{
		printf("ioctl fail\n");	
	}
	fb_info.xres_virtual=screen_info.xres_virtual;
	fb_info.bits_per_pixel=screen_info.bits_per_pixel;
	fb_info.xres=screen_info.xres;


    return fb_info;
};
