# Real-Time Object Detection and Tracking using YOLOv8 and Deep SORT

## Overview

This project implements a real-time object detection and multi-object tracking system using Python, OpenCV, YOLOv8, and Deep SORT. It captures live video from a webcam (or video file), detects multiple objects in each frame, assigns a unique tracking ID to every detected object, and displays the results with bounding boxes and labels in real time.

## Features

* Real-time webcam/video processing
* Object detection using the YOLOv8 pre-trained model
* Multi-object tracking using Deep SORT
* Bounding boxes with object labels
* Unique tracking IDs for each detected object
* Live video display using OpenCV

## Technologies Used

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* Deep SORT
* NumPy

## Project Structure

```text
CodeAlpha_ObjectDetectionTracking/
│
├── main.py
├── yolov8n.pt
├── requirements.txt
├── README.md

```

## Installation

1. Clone the repository.

```bash
git clone https://github.com/YOUR_USERNAME/CodeAlpha_ObjectDetectionTracking.git
```

2. Open the project folder.

```bash
cd CodeAlpha_ObjectDetectionTracking
```

3. Create a virtual environment.

```bash
python -m venv venv
```

4. Activate the virtual environment.

**Windows**

```bash
venv\Scripts\activate
```

5. Install the required packages.

```bash
pip install -r requirements.txt
```

6. Download the `yolov8n.pt` model (if it is not already present) and place it in the project folder.

## Running the Project

Run the following command:

```bash
python main.py
```

Press **Q** to exit the application.

## Workflow

1. Capture live video using OpenCV.
2. Detect objects in each frame using YOLOv8.
3. Extract bounding box coordinates and confidence scores.
4. Track detected objects using Deep SORT.
5. Assign a unique tracking ID to each object.
6. Display the tracked objects with labels and IDs.

## Sample Output

The application displays:

* Bounding boxes around detected objects.
* Object labels (Person, Laptop, Bottle, Chair, etc.).
* Unique tracking IDs.
* Real-time video output.

## Future Enhancements

* Save processed video automatically.
* Display FPS (Frames Per Second).
* Count objects crossing a virtual line.
* Support custom-trained YOLO models.
* Add email or alert notifications for specific objects.


## License

This project is developed for educational and internship purposes.
