import mediapipe as mp
import cv2
import numpy as np

# MediaPipe imports
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Initialize the object detector
options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='./model.tflite'),
    max_results=5,  # Adjust as needed
    running_mode=VisionRunningMode.IMAGE
)

def create_mp_image(image_data):
    """Create MediaPipe Image with error handling."""
    try:
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=image_data)
    except Exception as e:
        print(f"Error in MediaPipe image creation: {e}")
        raise

try:
    with ObjectDetector.create_from_options(options) as detector:
        # Load an image
        image = cv2.imread('./train/images/IMG_0509.jpg')
        if image is None:
            raise ValueError("Failed to load image. Check the file path.")

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Ensure the image is uint8
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

        # Validate the image
        if not isinstance(image_rgb, np.ndarray):
            raise TypeError("Image data is not a valid numpy array.")

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected image shape (H, W, 3), got {image_rgb.shape}.")

        if image_rgb.dtype != np.uint8:
            raise ValueError(f"Expected image dtype uint8, got {image_rgb.dtype}.")

        print(f"Image shape: {image_rgb.shape}, dtype: {image_rgb.dtype}")

        # Create MediaPipe image
        mp_image = create_mp_image(image_rgb)

        # Perform detection
        detection_results = detector.detect(mp_image)

        # Process and display results
        for detection in detection_results.detections:
            confidence = detection.categories[0].score
            if confidence >= 0.50:  # Filter by confidence
                bbox = detection.bounding_box
                category = detection.categories[0].category_name

                # Draw bounding box on the image
                start_point = (int(bbox.origin_x), int(bbox.origin_y))
                end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
                cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
                cv2.putText(image, f"{category}: {confidence:.2f}", start_point,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the results
        cv2.imshow('Object Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {e}")
