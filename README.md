# embedded_cv
Embedded Computer Vision using Tencent NCNN running on EmbedSky E9v3 

|   Model   |   Inference Time |   Result |
|---    |:-:    |:-:    |
|   Yolo FastestV2   |   233 ms  |   ![image](images/fastest.gif) |  
|   Yolov4 Tiny   |   2390ms   |   ![image](images/result-detection-yolov4tiny.png) |  
|   Yolov7 Tiny   |   6659 ms  |   ![image](images/result-detection-yolov7-tiny.png) |  
|   YoloX   |   11655 ms  |   ![image](images/result-detection-yolox.png) |
|   YoloXnano   |   1272 ms  |   ![image](images/result-detection-yoloXnano.png) |  


Using:
 https://github.com/Tencent/ncnn
 https://github.com/dog-qiuqiu/Yolo-FastestV2
