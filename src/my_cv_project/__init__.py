# def main() -> None:
#     print("Hello from my-cv-project!")
import cv2
import torch

print(f"OpenCV Version: {cv2.__version__}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available (GPU): {torch.cuda.is_available()}")

# Create a blank black image with OpenCV
img = cv2.mat_from_array([[0, 0], [0, 0]]) if hasattr(cv2, 'mat_from_array') else None
print("Environment setup successful!")