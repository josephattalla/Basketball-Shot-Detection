# Basketball Shot Detection

This project is designed to detect and count basketball shots, distinguishing between successful and unsuccessful attempts. It utilizes an object detection model, specifically YOLO (You Only Look Once), to track basketballs and hoops in a video feed. The primary goal is to determine when a basketball enters the hoop area, indicating a potential shot attempt, and then assess whether the shot was made based on subsequent ball movement.

## Requirements

To use this project, you'll need:

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Ultralytics' YOLO object detection library

## Usage

### Running the Detector

Instantiate the `Shot_Detector` class with the necessary parameters:

- `source`: Path to the video source.
- `output_path`: Path to save the resulting video.
- `detection_fps`: Number of frames per second to process for object detection.
- `display_object_info`: Boolean flag to display information about detected objects.

Use the `.run()` method to run the detection algorithm, and it will return the makes and attempts detected.

### Example

```python
from shot_detector import Shot_Detector

detector = Shot_Detector(source="path/to/video.mp4", output_path="path/to/output", detection_fps=30, display_object_info=True)
makes, attempts = detector.run()
print(f"Successful shots: {makes}/{attempts}")
```

## Algorithm Details

The algorithm used to detect shots and makes was inspired by https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker
