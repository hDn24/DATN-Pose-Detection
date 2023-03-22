import cv2
import argparse
import time
from ml.movenet import Movenet
import utils

fps_avg_frame_count = 10


def run(
    estimation_model: str,
    label_file: str,
    camera_id: int,
    width: int,
    height: int,
) -> None:
    """Continuously run inference on images acquired from the camera.

    Args:
        estimation_model: Name of the TFLite pose estimation model.
        label_file: Path to the label file for the pose classification model. Class
        camera_id: The camera id to be passed to OpenCV.
        width: Width of camera. Defaults to 640.
        height: Height of camera. Defaults to 480.
    """
    pose_detector = Movenet(estimation_model)

    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    max_detection_results = 3
    fps_avg_frame_count = 10

    while cap.isOpened():
        _, image = cap.read()

        counter += 1
        image = cv2.flip(image, 1)

        list_persons = [pose_detector.detect(image)]

        # Draw keypoints and edges on input image
        image = utils.visualize(image, list_persons)

        # Calculate the FPS
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        # Show the FPS
        fps_text = "FPS = " + str(int(fps))
        text_location = (left_margin, row_size)
        cv2.putText(
            image,
            fps_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            font_size,
            text_color,
            font_thickness,
        )

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

        cv2.imshow("image", image)

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        help="Name of estimation model.",
        required=False,
        default="movenet_lightning",
    )
    parser.add_argument(
        "--label_file",
        help="Label file for classification.",
        required=False,
        default="labels.txt",
    )
    parser.add_argument("--cameraId", help="Id of camera.", required=False, default=0)
    parser.add_argument(
        "--frameWidth",
        help="Width of frame to capture from camera.",
        required=False,
        default=640,
    )
    parser.add_argument(
        "--frameHeight",
        help="Height of frame to capture from camera.",
        required=False,
        default=480,
    )
    args = parser.parse_args()

    run(
        args.model,
        args.label_file,
        int(args.cameraId),
        args.frameWidth,
        args.frameHeight,
    )


if __name__ == "__main__":
    main()
