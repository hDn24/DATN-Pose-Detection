# Pose estimation final project.

This is the final project about `computer vision` using `deep learning` to train model and deploy on `edge device`.

## Pose detection, estimation and classification.

### Flow

![Logic](test_data/follow_logic.png)

### 17 Landmarks keypoints from estimation model

![17 landmarks](test_data/17_landmarks.png)

### Structure:

- Project features will be

  - Input image:

    ![Detection output](test_data/input.png)

  - Detection: Output expected.

    ![Detection output](test_data/detection.png)

  - Estimation: Output expected.

    ![Estimation output](test_data/estimation.png)

  - Classification: Output expected.

    ![Estimation output](test_data/classification.jpg)

### Step by step

- Collect data
- Using TFLite model to estimation yoga pose and get landmarks into csv file
- Define model to training with input as embedding vector from landmark in csv file
- Evaluate model
- Convert model -> TFLite format to deploy on Edge device
- Testing
