import numpy as np
import mediapipe as mp
import tensorflow as tf

print(f"NumPy version: {np.__version__}")
print(f"MediaPipe version: {mp.__version__}")
print(f"TensorFlow version: {tf.__version__}")

# Simple NumPy operation
arr = np.array([1, 2, 3], dtype=np.float32)
print(f"Array: {arr}, dtype: {arr.dtype}")
