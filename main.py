import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=30)

# Open webcam (use "input_video.mp4" instead of 0 for a video file)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'Q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO object detection
    results = model(frame, verbose=False)

    detections = []

    # Process detections
    for result in results:
        for box in result.boxes:

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Confidence
            confidence = float(box.conf[0])

            # Class ID
            class_id = int(box.cls[0])

            # Class name
            class_name = model.names[class_id]

            # Ignore low-confidence detections
            if confidence > 0.5:
                width = x2 - x1
                height = y2 - y1

                detections.append(
                    ([x1, y1, width, height], confidence, class_name)
                )

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracking results
    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        class_name = track.get_det_class()

        label = f"{class_name} ID:{track_id}"

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Object Detection and Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()