# Object Detection with MediaPipe and OpenCV

This project demonstrates how to use MediaPipe's Object Detection model along with OpenCV for real-time object detection on images. The code loads an object detection model, processes an input image, and visualizes the detected objects by drawing bounding boxes and category labels.

## Prerequisites

Before running the code, ensure you have the following installed:

1. **Python 3.7 or later**  
2. **Dependencies**: Install the required libraries using pip:
   ```bash
   pip install mediapipe opencv-python-headless numpy
   ```

3. **MediaPipe Model**: Download the required `model.tflite` file for object detection and place it in the project directory. Update the `model_asset_path` in the code if necessary.

4. **Image File**: Ensure you have an image at `./train/images/IMG_0509.jpg` or update the file path in the script.

## Usage

### Code Structure
1. **Import Libraries**:
   - `mediapipe` for object detection.
   - `opencv-python` for image processing.
   - `numpy` for array manipulations.

2. **Object Detector Initialization**:
   The MediaPipe Object Detector is initialized with the `model.tflite` file and configured to process images.

3. **Image Preprocessing**:
   - Load the image using OpenCV.
   - Convert it from BGR to RGB format for compatibility with MediaPipe.
   - Ensure the image is in the correct format (uint8 with shape `(H, W, 3)`).

4. **Object Detection**:
   The image is passed to the detector, and the results are processed to draw bounding boxes and labels on the image.

5. **Results Visualization**:
   The processed image with detections is displayed using OpenCV.

### Running the Code
1. Clone or download this repository.
2. Place your object detection model (`model.tflite`) in the root directory.
3. Place your test image in the `./train/images/` directory or update the path in the script.
4. Run the script:
   ```bash
   python object_detection.py
   ```
5. View the output image in a pop-up window. Press any key to close it.

### Example Output
Detected objects will be highlighted with green bounding boxes, and their categories along with confidence scores will be displayed on the image.

## Error Handling

- If the image fails to load, a `ValueError` is raised.
- The script includes checks for valid image format and dtype. If these are not met, appropriate error messages are shown.
- Errors during MediaPipe image creation or object detection are handled and displayed.

## Dependencies

- Python
- MediaPipe
- OpenCV
- NumPy

To install the dependencies, run:
```bash
pip install mediapipe opencv-python-headless numpy
```

## Customization

1. **Change Input Image**: Replace the file path in `cv2.imread()` to point to a different image.
2. **Adjust Model**: Replace `model.tflite` with a different MediaPipe model if needed.
3. **Modify Confidence Threshold**: Change the `if confidence >= 0.50` line to adjust the detection threshold.
4. **Bounding Box Color and Style**: Edit the `cv2.rectangle` and `cv2.putText` parameters to customize the visualization.

## License

This project is open-source and free to use for personal and educational purposes. Check the respective licenses for MediaPipe and OpenCV if using in commercial projects.