# detection-ripness-and-distance-measurement-of-strawberry
Strawberry detection by using the YOLOv8 model, Ripeness measurement of strawberry by using OpenCV and Measure distance of strawberry from the Depth Camera for Strawberry harvesting robots.

* Before you start, I recommend that you study the Physical Properties of Strawberry or Object that you want to detect, OpenCV, Artificial Intelligence, Machine Learning, Deep Learning, various color values, study measuring the distance between the object and the camera. and research related to your work. The scope of my work involves strawberry harvesting robots.

# Install Package & Library

* Install the libraries that will be used for the detection system and measure distance.
  Ex. os, datetime, sqrt, ultralytics, openni, cv2 or numpy etc.

# Select Framework for train model

* I select Model [YOLOv8 ultralytics](https://github.com/ultralytics/ultralytics)
(You can learn the details of the model I chose from the link.)

# Prepare Data

* Prepare your dataset. You should get a lot of images of the objects you are interested in detecting in order to train the model. Because the bigger the data, the better the performance of your model.

* Store data in the same folder.

![image](https://github.com/smartfarmdiy/detection-ripeness-and-distance-measurement-of-strawberry/assets/63504401/911c6a78-71c9-42ed-a431-0505409facd0)

# Annotation

* You will need to annotate the data set of your images. This can be done in [Roboflow](https://www.youtube.com/watch?v=wuZtUMEiKWY&list=PLrQjg-8WJZpOnfbFXyOfLlVfkQnRUCjCO&index=13), [CVAT](https://www.youtube.com/watch?v=m9fH9OWn8YM&list=PLrQjg-8WJZpOnfbFXyOfLlVfkQnRUCjCO&index=12), or any other appropriate platform. You can see an example from the link I attached.

![Screenshot 2023-07-26 142839](https://github.com/smartfarmdiy/detection-ripeness-and-distance-measurement-of-strawberry/assets/63504401/50861953-ac16-4693-873b-20d38857f787)

* After that, you need to divide the data into Train and Test, then add Augmentation to add flexibility to our dataset, and then generate data to prepare for the next train model (you can follow the link from above)

# Train Model

* In my part of the model training process, I will do it in Google Colab Pro (Because my data is big), where you need to import the dataset you have already created into Google Drive first.

* After that you need to connect Colab Pro to your Google Drive and import the library as per me.

'import os'
'import torch'
'from ultralytics import YOLO'

