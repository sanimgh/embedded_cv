#include "fastest.h"
#include <math.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/ioctl.h>
#include <unistd.h>
#include <linux/fb.h>
#include <linux/kd.h>
#include <fcntl.h>

//模型的参数配置
yoloFastestv2::yoloFastestv2()
{   
    printf("Creat yoloFastestv2 Detector...\n");
    //输出节点数
    numOutput = 2;
    //推理线程数
    numThreads = 4;
    //anchor num
    numAnchor = 3;
    //类别数目
    numCategory = 80;
    //NMS阈值
    nmsThresh = 0.25;

    //模型输入尺寸大小
    inputWidth = 352;
    inputHeight = 352;

    //模型输入输出节点名称
    inputName = "input.1";
    outputName1 = "794"; //22x22
    outputName2 = "796"; //11x11

    //打印初始化相关信息
    printf("numThreads:%d\n", numThreads);
    printf("inputWidth:%d inputHeight:%d\n", inputWidth, inputHeight);

    //anchor box w h
    std::vector<float> bias {12.64,19.39, 37.88,51.48, 55.71,138.31, 
                             126.91,78.23, 131.57,214.55, 279.92,258.87};

    anchor.assign(bias.begin(), bias.end());
}

yoloFastestv2::~yoloFastestv2()
{
    printf("Destroy yoloFastestv2 Detector...\n");
}

//ncnn 模型加载
int yoloFastestv2::loadModel(const char* paramPath, const char* binPath)
{
    printf("Ncnn mode init:\n%s\n%s\n", paramPath, binPath);

    net.load_param(paramPath);
    net.load_model(binPath);    

    printf("Ncnn model init sucess...\n");

    return 0;
}

float intersection_area(const TargetBox &a, const TargetBox &b)
{
    if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);

    return inter_width * inter_height;
}

bool scoreSort(TargetBox a, TargetBox b) 
{ 
    return (a.score > b.score); 
}

//NMS处理
int yoloFastestv2::nmsHandle(std::vector<TargetBox> &tmpBoxes, 
                             std::vector<TargetBox> &dstBoxes)
{
    std::vector<int> picked;
    
    sort(tmpBoxes.begin(), tmpBoxes.end(), scoreSort);

    for (int i = 0; i < tmpBoxes.size(); i++) {
        int keep = 1;
        for (int j = 0; j < picked.size(); j++) {
            //交集
            float inter_area = intersection_area(tmpBoxes[i], tmpBoxes[picked[j]]);
            //并集
            float union_area = tmpBoxes[i].area() + tmpBoxes[picked[j]].area() - inter_area;
            float IoU = inter_area / union_area;

            if(IoU > nmsThresh && tmpBoxes[i].cate == tmpBoxes[picked[j]].cate) {
                keep = 0;
                break;
            }
        }

        if (keep) {
            picked.push_back(i);
        }
    }
    
    for (int i = 0; i < picked.size(); i++) {
        dstBoxes.push_back(tmpBoxes[picked[i]]);
    }

    return 0;
}

//检测类别分数处理
int yoloFastestv2::getCategory(const float *values, int index, int &category, float &score)
{
    float tmp = 0;
    float objScore  = values[4 * numAnchor + index];

    for (int i = 0; i < numCategory; i++) {
        float clsScore = values[4 * numAnchor + numAnchor + i];
        clsScore *= objScore;

        if(clsScore > tmp) {
            score = clsScore;
            category = i;

            tmp = clsScore;
        }
    }
    
    return 0;
}

//特征图后处理
int yoloFastestv2::predHandle(const ncnn::Mat *out, std::vector<TargetBox> &dstBoxes, 
                              const float scaleW, const float scaleH, const float thresh)
{    //do result
    for (int i = 0; i < numOutput; i++) {   
        int stride;
        int outW, outH, outC;

        outH = out[i].c;
        outW = out[i].h;
        outC = out[i].w;

        assert(inputHeight / outH == inputWidth / outW);
        stride = inputHeight / outH;

        for (int h = 0; h < outH; h++) {
            const float* values = out[i].channel(h);

            for (int w = 0; w < outW; w++) {
                for (int b = 0; b < numAnchor; b++) {                    
                    //float objScore = values[4 * numAnchor + b];
                    TargetBox tmpBox;
                    int category = -1;
                    float score = -1;

                    getCategory(values, b, category, score);

                    if (score > thresh) {
                        float bcx, bcy, bw, bh;

                        bcx = ((values[b * 4 + 0] * 2. - 0.5) + w) * stride;
                        bcy = ((values[b * 4 + 1] * 2. - 0.5) + h) * stride;
                        bw = pow((values[b * 4 + 2] * 2.), 2) * anchor[(i * numAnchor * 2) + b * 2 + 0];
                        bh = pow((values[b * 4 + 3] * 2.), 2) * anchor[(i * numAnchor * 2) + b * 2 + 1];
                        
                        tmpBox.x1 = (bcx - 0.5 * bw) * scaleW;
                        tmpBox.y1 = (bcy - 0.5 * bh) * scaleH;
                        tmpBox.x2 = (bcx + 0.5 * bw) * scaleW;
                        tmpBox.y2 = (bcy + 0.5 * bh) * scaleH;
                        tmpBox.score = score;
                        tmpBox.cate = category;

                        dstBoxes.push_back(tmpBox);
                    }
                }
                values += outC;
            } 
        } 
    }
    return 0;
}

int yoloFastestv2::detection(const cv::Mat srcImg, std::vector<TargetBox> &dstBoxes, const float thresh)
{   
    dstBoxes.clear();

    float scaleW = (float)srcImg.cols / (float)inputWidth;
    float scaleH = (float)srcImg.rows / (float)inputHeight;
    
    //resize of input image data
    ncnn::Mat inputImg = ncnn::Mat::from_pixels_resize(srcImg.data, ncnn::Mat::PIXEL_BGR,\
                                                       srcImg.cols, srcImg.rows, inputWidth, inputHeight); 

    //Normalization of input image data
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    inputImg.substract_mean_normalize(mean_vals, norm_vals);  

    //creat extractor
    ncnn::Extractor ex = net.create_extractor();
    ex.set_num_threads(numThreads);

    //set input tensor
    ex.input(inputName, inputImg);

    //forward
    ncnn::Mat out[2]; 
    ex.extract(outputName1, out[0]); //22x22
    ex.extract(outputName2, out[1]); //11x11

    std::vector<TargetBox> tmpBoxes;
    //特征图后处理
    predHandle(out, tmpBoxes, scaleW, scaleH, thresh);

    //NMS
    nmsHandle(tmpBoxes, dstBoxes);
    
    return 0;
}

struct framebuffer_info
{
    uint32_t bits_per_pixel;    // framebuffer depth
    uint32_t xres_virtual;      // how many pixel in a row in virtual screen
      uint32_t xres;
};

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

int main()
{   
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };
    
    yoloFastestv2 api;


    api.loadModel("./model/yolo-fastestv2-opt.param",
                  "./model/yolo-fastestv2-opt.bin");

    cv::VideoCapture camera_stream(2);
    cv::VideoWriter videoWriter;
    int frame_width = camera_stream.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = camera_stream.get(cv::CAP_PROP_FRAME_HEIGHT);
    videoWriter.open("/run/media/mmcblk1p1/screenshot/video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(frame_width, frame_height));

    if(!camera_stream.isOpened())
    {
	printf("camera pas ouverte\n");
	return -1;
    }
    camera_stream.set(cv::CAP_PROP_FPS, 5);
    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
    std::ofstream ofs("/dev/fb0");

    cv::Mat cvImg;
    cv::Size2f image_size;

    while(1)
      {
	auto start = std::chrono::high_resolution_clock::now();

	camera_stream >> cvImg;
	std::vector<TargetBox> boxes;
	api.detection(cvImg, boxes);

	for (int i = 0; i < boxes.size(); i++) {
	  if (class_names[boxes[i].cate] == std::string("mouse") || 
                class_names[boxes[i].cate] == std::string("keyboard") || 
                class_names[boxes[i].cate] == std::string("tv")) {
	    std::cout<<boxes[i].x1<<" "<<boxes[i].y1<<" "<<boxes[i].x2<<" "<<boxes[i].y2
		     <<" "<<boxes[i].score<<" "<<boxes[i].cate<<std::endl;
        
	    char text[256];
	    // sprintf(text, "%s %.1f%%", class_names[boxes[i].cate], boxes[i].score * 100);

	    int baseLine = 0;
	    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

	    int x = boxes[i].x1;
	    int y = boxes[i].y1 - label_size.height - baseLine;
	    if (y < 0)
	      y = 0;
	    if (x + label_size.width > cvImg.cols)
	      x = cvImg.cols - label_size.width;

	    cv::rectangle(cvImg, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
			  cv::Scalar(255, 255, 255), -1);

	    cv::putText(cvImg, text, cv::Point(x, y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

	    cv::rectangle (cvImg, cv::Point(boxes[i].x1, boxes[i].y1), 
			   cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255, 255, 0), 2, 2, 0);
	  }
	}
	/*
	cv::cvtColor(cvImg,cvImg,cv::COLOR_BGR2BGR565);
	image_size=cvImg.size();
	int x_offset =(fb_info.xres_virtual-image_size.width)/2;
	for (int y = 0; y < image_size.height; y++)
	  {
	    ofs.seekp(y * (fb_info.xres_virtual * fb_info.bits_per_pixel / 8)+ (x_offset*fb_info.bits_per_pixel/8));	
	    ofs.write(reinterpret_cast<char*>(cvImg.ptr(y)),image_size.width*(fb_info.bits_per_pixel/8));
	  }
	*/
	videoWriter.write(cvImg);
	auto end = std::chrono::high_resolution_clock::now();

	std::cout << "frame processed in : ";
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "ms" << std::endl;
      }

    return 0;
}


