import cv2
import os


def annotate_video(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Video not found: {input_path}")
        return

    # Read the video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Could not open the video.")
        cap.release()
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    print(f"Video resolution: {width}x{height}")
    print(f"Frames per second: {fps}")

    # Configure the output video writer
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

    # Read and process the video frame by frame
    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            # Draw a green rectangle on every frame
            x1, y1 = 50, 50
            x2 = min(300, width - 1)
            y2 = min(250, height - 1)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Add text to every frame
            cv2.putText(
                frame,
                "AI/ML Internship - Week 1",
                (50, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # Save the annotated frame
            writer.write(frame)
            frame_count += 1

    finally:
        # Release video resources
        cap.release()
        writer.release()

    print(f"Processing complete! Frames processed: {frame_count}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    annotate_video("input.mp4", "output.mp4")