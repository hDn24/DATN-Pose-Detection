import cv2


def main():
    camera_id = 0
    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)

    while cap.isOpened():
        _, img = cap.read()

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

        cv2.imshow("img", img)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
