import os
from typing import List

from tflite_runtime.interpreter import Interpreter


class Classifier(object):
    def __init(self, model_name: str, label_file: str, score_threshold: float) -> None:
        """Initialize a pose classification model.

        Args:
          model_name: Name of the TFLite pose classification model.
          label_file: Path of the label list file.
          score_threshold: The minimum keypoint score to run classification.
        """
        # Append TFLITE extension to model_name if there's no extension
        _, ext = os.path.splitext(model_name)
        if not ext:
            model_name += ".tflite"

        # Initialize the TFLite model.
        interpreter = Interpreter(model_path=model_name, num_threads=4)
        interpreter.allocate_tensors()

        self._input_index = interpreter.get_input_details()[0]["index"]
        self._output_index = interpreter.get_output_details()[0]["index"]
        self._interpreter = interpreter

        self.pose_class_names = self._load_labels(label_file)
        self.score_threshold = score_threshold

    def _load_labels(self, label_path: str) -> List[str]:
        """Load label list from file.

        Args:
            label_path: Full path of label file.

        Returns:
            An array contains the list of labels.
        """
        with open(label_path, "r") as f:
            return [line.strip() for _, line in enumerate(f.readlines())]

    def classify(self):
        pass
