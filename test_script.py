import cv2
import time
from plitter.detector import detector, draw_boxes_and_count_on_image

# Load the riverine model ('cctv' mode)
model = detector('cctv')

# Path to your riverine video
video_path = "river.mov"
cap = cv2.VideoCapture(video_path)

frame_count = 0
total_count = 0
processing_times = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Run detection on the current frame
    preds = model(frame, size=1280)
    boxes = preds.xyxy[0][:, :4].tolist()
    scores = preds.xyxy[0][:, 4].tolist()
    class_ids = preds.xyxy[0][:, 5].tolist()
    classes = [model.names[int(i)] for i in class_ids]

    # Annotate the frame with boxes and an overlaid object count
    annotated_frame = draw_boxes_and_count_on_image(frame.copy(), boxes, classes, class_ids, scores)

    # Calculate the count (ignoring 'Trash bin' detections)
    frame_count_value = sum(1 for cls in classes if cls.lower() != "trash bin")
    total_count += frame_count_value

    # Log the count for this frame
    print(f"Frame {frame_count}: Count = {frame_count_value}")

    processing_time = time.time() - start_time
    processing_times.append(processing_time)
    print(f"Frame {frame_count} processing time: {processing_time:.3f} seconds")

    # Display the annotated frame (press 'q' to quit)
    cv2.imshow("Riverine Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# Quantitative Evaluation Summary
if frame_count > 0:
    avg_count = total_count / frame_count
    avg_time = sum(processing_times) / frame_count
    print(f"\nProcessed {frame_count} frames.")
    print(f"Average object count per frame: {avg_count:.2f}")
    print(f"Average processing time per frame: {avg_time:.3f} seconds")
