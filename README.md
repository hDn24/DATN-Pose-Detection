# Research and Deployment about Edge AI.
This is the final project about Computer Vision using Deep learning to train model and deploy on Edge Device.

## Pose detection, estimation and classification.

### Follow

![Logic](test_data/follow_logic.png)

### 17 Landmarks Keypoint from esimation model

![17 landmarks](test_data/17_landmarks.png)

### Structure:
*   Project features will be
    *   Input image:
    
        ![Detection output](test_data/input.png)

    *   Detection: Output expected.
    
        ![Detection output](test_data/detection.png)

    *   Estimation: Output expected.
    
        ![Estimation output](test_data/estimation.png)
 
    *   Classification: coming soon.

### Step by step
- Collect data
- Using TFLite model to estimation yoga pose and get landmarks into csv file
- Define model to training with input as embedding vector from landmark in csv file
- Evaluate model
- Convert model -> TFLite format to deploy on Edge device 
- Testing
