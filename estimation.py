import cv2


def main(width: int = 640, height: int = 480) -> None:
    camera_id = 0
    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while cap.isOpened():
        _, image = cap.read()

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

        cv2.imshow("image", image)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
