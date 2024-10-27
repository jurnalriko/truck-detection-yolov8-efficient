#  Efficiency of YOLOv8 Custom Trained Model for Truck Detection as Part of Bridge Structure Health Monitoring System
This project is an object detection algorithm for truck detection as a part of the bridge structure health monitoring system. The algorithm
has been optimized using the model compression method with quantization and pruning techniques applied. Based on the testing result, the
object detection model achieved an accuracy of 94.9%. Additionally, tests on quantization and pruning showed success by reducing model
size by 50.93% and 68.44%.

## Dataset
This project is using dataset from https://universe.roboflow.com/myfirstworkspace-jbsfo/truck-not-truck which has 2 classes, heavy-vehicle and light-vehicle. The example of the dataset if below:

![image](https://github.com/user-attachments/assets/78eb43f1-2071-479a-9841-1da3df159413)

## Quantization
This project is using ONNX Runtime as an Quantization API. ONNX Runtime documentations can be seen at https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

## Pruning
This project is using Pruning API from https://github.com/VainF/Torch-Pruning. 

## Testing
Based on the test that has been done, the graphics can be seen below.
![image](https://github.com/user-attachments/assets/baa4ce63-3e68-474b-8f9b-53e82c94cb38)

The experiments were conducted using identical parameters and environment to evaluate the performance of each model 
under similar conditions. Based on the research conducted and the results presented in Table 4.5 and Figure 4.11, the 
implementation of model compression techniques was able to reduce model size and accelerate inference time, thereby reducing 
computational requirements. In terms of model size, the pruning technique with a prune rate of 0.7 achieved the most significant 
reduction, decreasing the model size by 68.44% from the original 22.5MB to 7.1MB. On the other hand, in terms of inference time, 
the quantization technique yielded the best execution time at 75.11ms, representing a reduction of 64.75% compared to the 
original 213.1ms required for the model to detect a single image. Although the pruning technique with a prune rate of 0.7 was 
more effective in reducing model size than quantization, it also experienced the most significant performance degradation compared to other 
techniques. This trade-off needs to be carefully considered when selecting the most suitable model compression technique for 
optimizing a custom-trained YOLOv8 model. Therefore, it is essential to compare the reductions in model size and inference time with 
the resulting evaluation metrics. These comparisons are visualized in the following graphs.

![image](https://github.com/user-attachments/assets/bb912303-ae64-42d7-b238-4878789dbfb6)

Figure above presents a comparison of the performance of different model compression techniques based on two parameters: final model size and 
model performance as reflected by the mAP50-95 metric. According to the graph, pruning with a rate of 0.7 results in the smallest model size
but also achieves the lowest mAP50-95 score. Subsequently, pruning with a rate of 0.5 yields a considerable reduction in model size but experiences 
a significant drop in mAP50-95. Furthermore, pruning with a rate of 0.3 results in a minimal reduction in model size and experiences a less 
significant decrease in mAP50-95 compared to pruning rates of 0.7 and 0.5. Additionally, INT8 quantization effectively reduces model size while 
maintaining the highest mAP50-95 score among all techniques. Therefore, based on the data, model compression using quantization outperforms pruning 
techniques as it effectively reduces model size without a significant decrease in accuracy compared to pruning.
