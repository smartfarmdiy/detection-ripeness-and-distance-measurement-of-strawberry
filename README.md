# detection-ripness-and-distance-measurement-of-strawberry
Strawberry detection by using the YOLOv8 model, Ripeness measurement of strawberry by using OpenCV and Measure distance of strawberry from the Depth Camera for Strawberry harvesting robots.

* Before you start, I recommend that you study the Physical Properties of Strawberry or Object that you want to detect, OpenCV, Artificial Intelligence, Machine Learning, Deep Learning, various color values, study measuring the distance between the object and the camera. and research related to your work. The scope of my work involves strawberry harvesting robots.

# Install Package & Library

* Install the libraries that will be used for the detection system and measure distance.
  Ex. os, datetime, sqrt, ultralytics, openni, cv2 or numpy etc.

`pip install opencv-python numpy openni-python ultralytics`

I used python 3.9.7 for this work.

# Select Framework for train model

* I select Model [YOLOv8 ultralytics](https://github.com/ultralytics/ultralytics)
(You can learn the details of the model I chose from the link.)

# Prepare Data

* Prepare your dataset. You should get a lot of images of the objects you are interested in detecting in order to train the model or A dataset that someone has already created can be found on the website [OpenimageDataset](https://storage.googleapis.com/openimages/web/index.html) or [Roboflow](https://roboflow.com/) But you can only use it for example and not for actual use because the free dataset is not very specific to your work. Because the bigger the data, the better the performance of your model.

* Store data in the same folder.

![image](https://github.com/smartfarmdiy/detection-ripeness-and-distance-measurement-of-strawberry/assets/63504401/911c6a78-71c9-42ed-a431-0505409facd0)

# Annotation

* You will need to annotate the data set of your images. This can be done in [Roboflow Example](https://www.youtube.com/watch?v=wuZtUMEiKWY&list=PLrQjg-8WJZpOnfbFXyOfLlVfkQnRUCjCO&index=13), [CVAT Example](https://www.youtube.com/watch?v=m9fH9OWn8YM&list=PLrQjg-8WJZpOnfbFXyOfLlVfkQnRUCjCO&index=12), or any other appropriate platform. You can see an example from the link I attached.

![Screenshot 2023-07-26 142839](https://github.com/smartfarmdiy/detection-ripeness-and-distance-measurement-of-strawberry/assets/63504401/50861953-ac16-4693-873b-20d38857f787)

* After that, you need to divide the data into Train and Test, then add Augmentation to add flexibility to our dataset, and then generate data to prepare for the next train model (you can follow the link from above)

# Train Model

* In my part of the model training process, I will do it in Google Colab Pro (Because my data is big), where you need to import the dataset you have already created into Google Drive first.

* After that you need to connect Colab Pro to your Google Drive and import the library. [Train Docs](https://docs.ultralytics.com/modes/train/)

<pre>
import os
import torch
from ultralytics import YOLO
</pre>

* Code for train

Choose a model based on the capabilities of your computer or GPU. The larger the model, the higher the power required to run. Or the smaller the model, the less power is required to run.

![image](https://github.com/smartfarmdiy/detection-ripeness-and-distance-measurement-of-strawberry/assets/63504401/e8f39c00-a7b5-4313-96c0-a54c65d053a5)

![image](https://github.com/smartfarmdiy/detection-ripeness-and-distance-measurement-of-strawberry/assets/63504401/9178a6ff-9a1c-42bd-834d-b417b7d399a1)

from above info you can click [YOLOv8 ultralytics](https://github.com/ultralytics/ultralytics) to lerning more.
<pre>
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")

# Use the model
results = model.train(data="/content/drive/MyDrive/Strawberry_dataset/google_colab_config.yaml", epochs=200, imgsz =640)
</pre>

* If error this code is Load Checkpoint to continuous

<pre>
from ultralytics import YOLO

# Load a model
model = YOLO("/content/drive/MyDrive/Strawberry_dataset/33resume/detect/train/weights/last.pt")

# Resume training
results = model.train(resume = True)
</pre>

# Detection for Strawberry

* After training the model is finished, you will get the model in the form of best.pt and last.pt files for use in our object detection. You can set the file path and put it in my code and try to detect the objects you want.

![image](https://github.com/smartfarmdiy/detection-ripeness-and-distance-measurement-of-strawberry/assets/63504401/57a3d459-cbaa-4d11-9e5f-9dab30093158)

![image](https://github.com/smartfarmdiy/detection-ripeness-and-distance-measurement-of-strawberry/assets/63504401/6e59a357-8862-4417-aed6-851a240e9e1b)

# Ripeness Measurement

To measure the ripeness of my strawberries I will use the color value technique, within my code it will give the ripeness level as follows:

- Unripe will be set <= 25% of ripeness.

- Turning will be set <= 50% of ripeness

- Partially Ripe is will be set <= 75% of ripeness

- Fully Ripe will be set <= 100% of ripeness

- Rotten will be set 0% of ripeness

* Define the minimum and maximum color values for class

<pre>
min_color_unripe = (0, 128, 0)
max_color_turning = (255, 255, 255)

min_color_PartiallyRipe = (0, 35, 129)
max_color_turning1 = (255, 255, 255)
</pre>

* Function to calculate percentage within a range

<pre>
def calculate_percentage(value, min_value, max_value):
    # Ensure that the value is within the specified range
    value = (max(min_value[0], min(max_value[0], value[0])),
             max(min_value[1], min(max_value[1], value[1])),
             max(min_value[2], min(max_value[2], value[2])))

    # Calculate the percentage for each channel (B, G, R)
    percentage_b = ((value[0] - min_value[0]) / (max_value[0] - min_value[0])) * 100
    percentage_g = ((value[1] - min_value[1]) / (max_value[1] - min_value[1])) * 100
    percentage_r = ((value[2] - min_value[2]) / (max_value[2] - min_value[2])) * 100

    # Calculate the average percentage across channels
    percentage = (percentage_b + percentage_g + percentage_r) / 3.0
 
    return percentage
</pre>

* Code for specifying the color range There will be an average of the color values within the Bounding Box of that strawberry. This is then compared to a given color range where I will give the fruit green to white at 0 % ripeness to 25 % ripeness and from white to red at 25 % ripeness to 100 % ripeness. Rotten results will be given as 0 % ripeness (ripeness color value or percentage of ripeness Different for each fruit which you can use my code as a reference in your work)

<pre>
if (min_color_unripe <= avg_color).all() and (avg_color <= max_color_turning).all():

    percentage = calculate_percentage(avg_color, min_color_unripe, max_color_turning)
    color_rec = (0, int(255 * (1 - percentage / 100)), int(255 * (percentage / 100)))
    ripeness_percentage = percentage * (25 / 100)

elif (min_color_PartiallyRipe <= avg_color).all() and (avg_color <= max_color_turning1).all():

    percentage = calculate_percentage(avg_color, min_color_PartiallyRipe, max_color_turning1)
    color_rec = (0, int(255 * (1 - percentage / 100)), int(255 * (percentage / 100)))
    ripeness_percentage = ((100 - percentage) * (75 / 100)) + 25
</pre>

# Orbbec Astra Pro Camera for Distance Measurement

* In my work, I use an Orbbec Astra Pro Camera to measure the distance between the camera and a strawberry. The camera's Dapth function is used to measure the distance. Before starting the test, you will need to install the camera driver and SDK for camera. [Device Support](https://www.orbbec.com/developers/device-support/) and [Git astra](https://github.com/orbbec/ros_astra_camera)

* Initialize OpenNI SDK (this code for start depth stream)

<pre>
redistPath = "OpenNI\Win64-Release\sdk\libs"
openni2.initialize(redistPath)
device = openni2.Device.open_any()
depth_stream = device.create_depth_stream()
depth_stream.start()
</pre>

* Define constants for smoothing and temporal filtering

<pre>
SMOOTHING_FACTOR = 0.9  # Adjust this value for the desired smoothing strength
TEMPORAL_FACTOR = 0.8  # Adjust this value for the desired temporal filtering strength
prev_avg_depth = None  # Initialize the previous average depth
</pre>

* function for calculate distance from depth frame in bounding box of strawberry

<pre>
def calculate_distance(depth_frame, x1, y1, x2, y2):
    global prev_avg_depth

    # Get depth values within the bounding box
    p_depth = depth_frame.get_buffer_as_uint16()
    depth_data = np.array(p_depth) 
    depth_image = depth_data.reshape(depth_frame.height, depth_frame.width)

    # Calculate the average depth within the bounding box
    depth_roi = depth_image[y1:y2, x1:x2]
    avg_depth = np.mean(depth_roi)

    # Apply smoothing filter
    if prev_avg_depth is not None:
        avg_depth = (1 - SMOOTHING_FACTOR) * prev_avg_depth + SMOOTHING_FACTOR * avg_depth

    # Apply temporal filtering
    if prev_avg_depth is not None:
        avg_depth = TEMPORAL_FACTOR * prev_avg_depth + (1 - TEMPORAL_FACTOR) * avg_depth

    # Update previous average depth
    prev_avg_depth = avg_depth

    # Convert depth value to distance in centimeters
    depth_scale = 0.1  # Conversion factor specific to the depth camera
    distance = avg_depth * depth_scale

    # Calculate individual depth values for each corner of the bounding box
    depth_x1y1 = depth_image[y1, x1]
    depth_x1y2 = depth_image[y2, x1]
    depth_x2y1 = depth_image[y1, x2]
    depth_x2y2 = depth_image[y2, x2]

    return distance, depth_x1y1, depth_x1y2, depth_x2y1, depth_x2y2
    # return distance
</pre>
