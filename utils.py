import math
from typing import List, Tuple

import cv2
from data import Person
import numpy as np

# Map edges to a RGB color
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (147, 20, 255),
    (0, 2): (255, 255, 0),
    (1, 3): (147, 20, 255),
    (2, 4): (255, 255, 0),
    (0, 5): (147, 20, 255),
    (0, 6): (255, 255, 0),
    (5, 7): (147, 20, 255),
    (7, 9): (147, 20, 255),
    (6, 8): (255, 255, 0),
    (8, 10): (255, 255, 0),
    (5, 6): (0, 255, 255),
    (5, 11): (147, 20, 255),
    (6, 12): (255, 255, 0),
    (11, 12): (0, 255, 255),
    (11, 13): (147, 20, 255),
    (13, 15): (147, 20, 255),
    (12, 14): (255, 255, 0),
    (14, 16): (255, 255, 0),
}


def visualize(
    image: np.ndarray,
    list_persons: List[Person],
    keypoint_color: Tuple[int, ...] = (0, 255, 0),
    keypoint_threshold: float = 0.05,
    instance_threshold: float = 0.1,
) -> np.ndarray:
    """Draws landmarks and edges on the input image and return it.
    Args:
        image: The input RGB image.
        list_persons: The list of all "Person" entities to be visualize.
        keypoint_color: the colors in which the landmarks should be plotted.
        keypoint_threshold: minimum confidence score for a keypoint to be drawn.
        instance_threshold: minimum confidence score for a person to be drawn.
    Returns:
        Image with keypoints and edges.
    """
    for person in list_persons:
        if person.score < instance_threshold:
            continue

        keypoints = person.keypoints
        bounding_box = person.bounding_box

        # Draw all the landmarks
        for i in range(len(keypoints)):
            if keypoints[i].score >= keypoint_threshold:
                cv2.circle(image, keypoints[i].coordinate, 2, keypoint_color, 4)

        # Draw all the edges
        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (
                keypoints[edge_pair[0]].score > keypoint_threshold
                and keypoints[edge_pair[1]].score > keypoint_threshold
            ):
                cv2.line(
                    image,
                    keypoints[edge_pair[0]].coordinate,
                    keypoints[edge_pair[1]].coordinate,
                    color,
                    2,
                )

    return image
