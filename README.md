# CenterTrack 2D tracking C++ caffe 版本

## 根据[**CenterTrack官方pytorch版本**](https://github.com/xingyizhou/CenterTrack)改写.
> 话说CenteTrack是真的丝滑,检测加跟踪一步到位,检测效果比yolov3还好,跟踪效果跟deepsort有的一拼,屏内跟踪的不二之选.

## 使用[**此版本的CenterTrack**](https://github.com/lrjbdss/CenterTrack_2D_train)进行训练,然后使用[**模型转换工具**](https://github.com/xxradon/PytorchToCaffe)转成caffemodel
> 有坑,caffe的pooling只有ceil_mode模式,但pytorch默认ceil_model为False,转模型会报错.屏蔽掉ceil_model关键字,打开ceil_mode模式即可,因为模型是基于floor_model模式训练的,输出尺寸跟训练时不同,后处理的坐标转换有点麻烦.  
>另外,考虑到简化处理,mns阶段的sigmoid和maxpooling被放到了模型中,但caffe模型不支持CenterTrack的nms操作,而且量化后的海思wk模型也不能正常的maxpooling(pooling后最大像素值也会改变),所以模型输出的是sigmoid层,后续的nms由cpu完成.

## 主干网络使用resnet18 
> resnest18也试了,没感觉有明显提升