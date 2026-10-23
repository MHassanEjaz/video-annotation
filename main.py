import cv2
from ultralytics import YOLO


def annotate_video(input_path, output_path):
    # Load a pretrained CNN-based object detection model
    model = YOLO("yolov8n.pt")

    # Open the input video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Could not open input video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    # Configure the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("Error: Could not create output video.")
        cap.release()
        return

    frame_count = 0

    # Process video frame by frame
    while True:
        success, frame = cap.read()

        if not success:
            break

        # Detect objects in the current frame
        results = model(frame, conf=0.35, verbose=False)

        # Draw detected object boxes and labels
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                label = (
                    f"{model.names[class_id]} "
                    f"{confidence:.2f}"
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

        writer.write(frame)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    writer.release()

    print(f"Processing complete: {frame_count} frames")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    annotate_video("input.mp4", "output.mp4")
