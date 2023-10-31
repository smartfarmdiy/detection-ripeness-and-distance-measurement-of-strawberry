import os
import datetime
from math import sqrt
from ultralytics import YOLO
from openni import openni2
import cv2
import numpy as np
import torch
ripeness_percent = 0

def countSTB(detections, num_stb=0):
    for _ in detections:
        num_stb += 1
    cv2.putText(frame, f"Number(s) of Strawberry : {num_stb}", (250, 40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 3)
    cv2.putText(frame, f"Number(s) of Strawberry : {num_stb}", (250, 40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 127, 0), 1)
    print("Numder(s) of STB in countSTB function : " + str(num_stb))
    return num_stb

def analyze_frame(frame):
    p_depth = frame.get_buffer_as_uint16()
    return p_depth

# Define constants for smoothing and temporal filtering
SMOOTHING_FACTOR = 0.9  # Adjust this value for the desired smoothing strength
TEMPORAL_FACTOR = 0.8  # Adjust this value for the desired temporal filtering strength
prev_avg_depth = None  # Initialize the previous average depth

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

# Define the minimum and maximum color values for class 1 (Stb)
min_color_unripe = (0, 128, 0)  # Unripe
max_color_turning = (255, 255, 255)  # Turning

min_color_PartiallyRipe = (0, 35, 129)  # Partially Ripe
max_color_turning1 = (255, 255, 255)  # Turning (for Partially Ripe)

# Function to calculate percentage within a range
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

# Initialize OpenNI SDK
redistPath = "OpenNI\Win64-Release\sdk\libs"
openni2.initialize(redistPath)
device = openni2.Device.open_any()
depth_stream = device.create_depth_stream()
depth_stream.start()

# Initialize the camera
cap = cv2.VideoCapture(0)
check, frame = cap.read()
H, W, _ = frame.shape

#Save Video
out = cv2.VideoWriter("todlong3_50cm.avi", cv2.VideoWriter_fourcc(*'MP4V'),
     int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Load the YOLO model
model_path = os.path.join('.', 'last.pt')
model = YOLO(model_path).to(torch.device(0))

# Class names
class_name_dict = {0: 'Rotten_Stb', 1: 'Stb'}

# Frame count and FPS calculation
frame_count = 0
start_time = cv2.getTickCount()

# Main Loop
while True:
    start = datetime.datetime.now()         # define start time for calculate fps
    # Detect objects using the YOLO model
    results = model(frame, conf=0.5)[0]

    # Get the current date and time
    currentdate = str(datetime.datetime.now())

    # Put text on the frame
    cv2.putText(frame, currentdate, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, currentdate, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "Smart AI Solution", (10, 55), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, "Smart AI Solution", (10, 55), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)

    # Read a frame from the depth stream
    depth_frame = depth_stream.read_frame()
    detections = []
    for det in results.boxes.xyxy.tolist():
        detections.append(det)
    # Process detected objects
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if class_id == 0:

            color_bounding = (255, 153, 51)  # Color for unripe strawberries
            ripeness_percent = 0
            distance, depth_x1y1, depth_x1y2, depth_x2y1, depth_x2y2 = calculate_distance(
                depth_frame, int(x1 + 11), int(y1 + 11), int(x2 - 11), int(y2 - 11))

        elif class_id == 1:

            roi = frame[int(y1 + 11):int(y2 - 11), int(x1 + 11):int(x2 - 11)]
            avg_color = np.mean(roi, axis=(0, 1))
            if np.isnan(avg_color).any():
                avg_color = np.nan_to_num(avg_color, nan= (255,255,255))

            # Map the average color to the bounding box color
            color_bounding = tuple(int(c) for c in avg_color)

            if (min_color_unripe <= avg_color).all() and (avg_color <= max_color_turning).all():

                percentage = calculate_percentage(avg_color, min_color_unripe, max_color_turning)
                color_rec = (0, int(255 * (1 - percentage / 100)), int(255 * (percentage / 100)))
                ripeness_percent = percentage * (25 / 100)

            elif (min_color_PartiallyRipe <= avg_color).all() and (avg_color <= max_color_turning1).all():

                percentage = calculate_percentage(avg_color, min_color_PartiallyRipe, max_color_turning1)
                color_rec = (0, int(255 * (1 - percentage / 100)), int(255 * (percentage / 100)))
                ripeness_percent = ((100 - percentage) * (75 / 100)) + 25

            # Calculate the distance and individual depth values
            distance, depth_x1y1, depth_x1y2, depth_x2y1, depth_x2y2 = calculate_distance(
                depth_frame, int(x1 + 11), int(y1 + 11), int(x2 - 11), int(y2 - 11))

        # Draw bounding box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bounding, 2)

        # Put text with class and score
        cv2.putText(frame, f"{class_name_dict[int(class_id)].upper()} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, color_bounding, 1, cv2.LINE_AA)

        if ripeness_percent <= 25:
            color = (200, 255, 255)  # Color for Unripe
            text = "Unripe"
        elif ripeness_percent <= 50:
            color = (0, 255, 255)  # Color for Turning
            text = "Turning"
        elif ripeness_percent <= 75:
            color = (176, 255, 198)  # Color for Partially Ripe
            text = "Partially Ripe"
        else:
            color = (0, 255, 0)  # Color for Fully Ripe
            text = "Fully Ripe"

        Ripeness_text = f"{text} {ripeness_percent:.2f}%"
        cv2.putText(frame, Ripeness_text, (x1, y1 - 25), cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1, cv2.LINE_AA)


        # Calculate and display x, y, and z coordinates
        xy_text = f"X: {int((x1 + x2) / 2)}, Y: {int((y1 + y2) / 2)}"
        cv2.putText(frame, xy_text, (int(x2 + 10), int(y1 + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255),
                    1, cv2.LINE_AA)

        if 30 <= distance <= 50:
            z_text = f"Z: {distance:.2f} cm"
            color_z = (255,255,255)
        else:
            z_text = "Z: Not in range"
            color_z = (255,255,255)

        cv2.putText(frame, z_text, (int(x2 + 10), int(y1 + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_z, 1,
                    cv2.LINE_AA)

        # Print x, y, z coordinates and class information
        print(f"X: {int((x1 + x2) / 2)}, Y: {int((y1 + y2) / 2)}, Z: {distance:.2f}")
        print(f"Ripeness {ripeness_percent:.2f}%")
        print(f"{class_name_dict[int(class_id)].upper()} {score:.2f}")
        print(distance)
    
    countSTB(detections, num_stb=0)

    # Update FPS every 10 frames
    # if frame_count % 10 == 0:
    #     end_time = cv2.getTickCount()
    #     elapsed_time = (end_time - start_time) / cv2.getTickFrequency()
    #     fps = 10 / elapsed_time
    #     start_time = cv2.getTickCount()

    #     fps_text = f"FPS: {fps:.2f}"
    #     cv2.putText(frame, fps_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    #     cv2.putText(frame, fps_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    #     print(f"FPS: {fps:.2f}")

    # end time to compute the fps
    end  = datetime.datetime.now()

    # show the time it took to process 1 frame
    print(f"Time to process 1 frame : {(end - start).total_seconds() * 1000:.0f} milliseconds")

    # calulate the frame per second and draw it on the frame
    fps = f"FPS : {1 / (end - start).total_seconds():.2f}"

    cv2.putText(frame, fps, (10, 80), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 3)
    cv2.putText(frame, fps, (10, 80), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0 ,255), 1)
        

    cv2.imshow("Smart Ai Solution", frame)
    out.write(frame)
    check, frame = cap.read()

    if cv2.waitKey(25) & 0xFF == ord ("e"):
        break

# Clean up
depth_stream.stop()
depth_stream.destroy()
device.close()
openni2.unload()

cap.release()
out.release()
cv2.destroyAllWindows()