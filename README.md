# LimeLight AI - Computer Vision Object Detection System

A sophisticated computer vision system designed for real-time detection, tracking, and coordinate transformation of colored objects (blue, red, and yellow) using advanced camera calibration and 3D coordinate mapping.

## 🎯 Overview

LimeLight AI is a Python-based computer vision pipeline that combines OpenCV image processing with precise camera calibration to detect colored objects and transform their pixel coordinates into real-world 3D coordinates. The system is particularly useful for robotics applications, automated systems, and any scenario requiring accurate spatial positioning of objects.

## ✨ Features

- **Multi-Color Detection**: Simultaneous detection of blue, red, and yellow objects
- **Camera Calibration**: Advanced camera matrix and distortion correction
- **3D Coordinate Transformation**: Convert pixel coordinates to real-world coordinates
- **Angular Calculations**: Precise angle measurements and discrete angle rounding
- **Dynamic Area Filtering**: Adaptive filtering based on object position in frame
- **Ellipse Fitting**: Robust contour analysis with ellipse fitting for angle detection
- **Real-time Processing**: Optimized pipeline for real-time applications

## 📁 Project Structure

```
LimeLight_AI/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── cv_code_blue.py             # Blue object detection
├── cv_code_red.py              # Red object detection  
├── cv_code_yellow.py           # Yellow object detection
├── new_blue_code.py            # Enhanced blue object detection
├── new_red_code.py             # Enhanced red object detection
└── new_yellow_code.py          # Enhanced yellow object detection
```

## 🔧 Technical Specifications

### Camera Calibration Parameters
- **Camera Matrix**: 3x3 intrinsic camera matrix with focal lengths and principal point
- **Distortion Coefficients**: 5-parameter distortion model for lens correction
- **Frame Dimensions**: 320x240 pixel resolution
- **Field of View**: 27° horizontal, 20.5° vertical (half-angles)

### Object Detection Parameters
- **Object Dimensions**: 8.89cm x 3.81cm (real-world size)
- **Dynamic Area Filtering**: Adaptive area thresholds based on vertical position
- **Color Ranges**: 
  - Blue: HSV [90-120, 100-255, 30-255]
  - Red: HSV [0-10, 100-255, 100-255] + [170-180, 100-255, 100-255]
  - Yellow: HSV [10-30, 100-255, 100-255]

### Coordinate System
- **World Coordinates**: 3D coordinate system with Z=0 plane for target objects
- **Camera Position**: Precisely calibrated 3D position in world coordinates
- **Coordinate Transformation**: Ray-casting from camera through pixel to world plane

## 🚀 Installation

### Prerequisites
```bash
pip install numpy opencv-python
```

### Dependencies
- **NumPy**: For numerical computations and array operations
- **OpenCV**: For computer vision and image processing
- **Math**: For trigonometric calculations

## 💻 Usage

### Basic Usage

```python
import cv2
from cv_code_blue import runPipeline

# Initialize camera or load image
cap = cv2.VideoCapture(0)  # Use camera
# or
image = cv2.imread('your_image.jpg')  # Load image

# Initialize robot/output array
llrobot = [0] * 11  # Initialize with appropriate size

# Process frame
ret, frame = cap.read()
if ret:
    result = runPipeline(frame, llrobot)
    # Process results...
```

### Key Functions

#### `runPipeline(image, llrobot)`
Main processing pipeline that:
- Converts image to HSV color space
- Applies color-specific masking
- Performs morphological operations
- Detects and analyzes contours
- Calculates angles and coordinates
- Updates the robot output array

#### `get_sample_position_python(llpython_output, color_param, ...)`
Transforms pixel coordinates to world coordinates using:
- Camera calibration parameters
- Ray-casting algorithms
- 3D coordinate transformation
- Angle processing and correction

#### `calculate_contour_world_coords(cx, cy, ...)`
Converts individual pixel coordinates to world coordinates with:
- Distortion correction
- Camera matrix transformation
- 3D ray intersection calculations

## 🎛️ Configuration

### Camera Calibration
Update the camera matrix and distortion coefficients in each file:

```python
camera_matrix = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
```

### Color Tuning
Adjust HSV color ranges for different lighting conditions:

```python
# Example for blue objects
lower_blue = np.array([90, 100, 30])
upper_blue = np.array([120, 255, 255])
```

## 🔄 Version Differences

### Standard vs Enhanced Versions
- **Standard versions** (`cv_code_*.py`): Core implementations with standard filtering
- **Enhanced versions** (`new_*_code.py`): Enhanced implementations with improved area filtering and refined parameters

### Key Improvements in Enhanced Versions
- Enhanced dynamic area filtering constants
- Improved morphological operations
- Better contour analysis algorithms
- More robust error handling

## 🤖 Applications

- **Robotics**: Object detection and manipulation
- **Automation**: Quality control and sorting systems
- **Augmented Reality**: Object tracking and overlay
- **Sports Analysis**: Ball tracking and trajectory analysis
- **Industrial Vision**: Part identification and positioning

## 🛠️ Troubleshooting

### Common Issues

1. **Poor Detection**: Adjust HSV color ranges for your lighting conditions
2. **Inaccurate Coordinates**: Recalibrate camera matrix and distortion coefficients
3. **False Positives**: Tune area filtering parameters
4. **Performance Issues**: Optimize image resolution and processing parameters

## 📈 Performance Optimization

- Use appropriate image resolution (320x240 recommended)
- Implement region of interest (ROI) processing
- Optimize morphological kernel sizes
- Consider multi-threading for real-time applications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **FTCRoboRangers2024-25** - Initial work and development

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- NumPy developers for numerical computing support
- Contributors to camera calibration algorithms

---

*For questions, issues, or contributions, please open an issue on the repository.*
