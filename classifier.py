import os
from typing import List

from tflite_runtime.interpreter import Interpreter


class Classifier(object):
    def __init(self, model_name, label_file, score_threshold):
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

    def classify(self):
        pass
