import os
from typing import List
from data import Category, Person


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

    def classify_pose(self, person: Person) -> List[Category]:
        """Run classification on an input.

        Args:
          person: A data.Person instance.

        Returns:
          A list of prediction result in the data.Class format.
          Sorted by probability descending.
        """
        # Check if all keypoints are detected before running the classifier.
        # If there's a keypoint below the threshold, return zero probability for all
        # class.
        min_score = min([keypoint.score for keypoint in person.keypoints])
        if min_score < self.score_threshold:
            return [
                Category(label=class_name, score=0)
                for class_name in self.pose_class_names
            ]

        # Flatten the input and add an extra dimension to match with the requirement
        # of the TFLite model.
        input_tensor = [
            [keypoint.coordinate.y, keypoint.coordinate.x, keypoint.score]
            for keypoint in person.keypoints
        ]
        input_tensor = np.array(input_tensor).flatten().astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # Set the input and run inference.
        self._interpreter.set_tensor(self._input_index, input_tensor)
        self._interpreter.invoke()

        # Extract the output and squeeze the batch dimension
        output = self._interpreter.get_tensor(self._output_index)
        output = np.squeeze(output, axis=0)

        # Sort output by probability descending.
        prob_descending = sorted(
            range(len(output)), key=lambda k: output[k], reverse=True
        )
        prob_list = [
            Category(label=self.pose_class_names[idx], score=output[idx])
            for idx in prob_descending
        ]

        return prob_list
