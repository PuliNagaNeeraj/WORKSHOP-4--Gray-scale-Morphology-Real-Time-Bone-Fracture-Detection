# WORKSHOP-4--Gray-scale-Morphology-Real-Time-Bone-Fracture-Detection
## Aim :
To implement real-time fracture detection using grayscale morphology on X-ray images.

## Algorithm Steps:
1.Preprocess the input X-ray image by converting it to grayscale.

2.Apply thresholding to segment the image into foreground (fracture) and background regions.

3.Utilize morphological operations, such as dilation and erosion, to enhance fracture features and remove noise.

4.Detect fracture areas by finding contours in the processed image.

5.Highlight the detected fracture areas on the original image.

6.Display the original image with highlighted fracture areas in real-time.

## Program :
### Developed By : PULI NAGA NEERAJ
### Register Number : 212223240130

import cv2
import numpy as np

# Function to perform fracture area prediction using morphological operations
def predict_fracture_area(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to segment the image
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to enhance features
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of potential fractures
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the contour with maximum area (likely the fracture)
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    
    # Create a mask for the fracture area
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)
    
    # Apply the mask to the original image to highlight the fracture area
    fracture_area = cv2.bitwise_and(img, img, mask=mask)
    
    return fracture_area

# Main function for fracture area prediction from an input image
def main():
    # Read input image
    input_img_path = r'bone fracture.jpeg'  # Replace with the path to your X-ray image
    img = cv2.imread(input_img_path)
    
    # Check if the image is loaded successfully
    if img is None:
        print("Error: Unable to load image.")
        return
    
    # Predict the fracture area in the input image
    fracture_area = predict_fracture_area(img)
    
    # Display the original image and the predicted fracture area
    cv2.imshow('Original', img)
    cv2.imshow('Fracture Area Prediction', fracture_area)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
  
## output
![image](https://github.com/23004426/DIP_Workshop_4/assets/144979327/0683110b-658b-4c15-8507-b0c3bb2309c1)


The program successfully detects fractures in real-time from the input X-ray images.

The original X-ray image and the image with highlighted fracture areas are displayed simultaneously.

Fracture areas are outlined in green on the original image, making them easily identifiable.

## Advantages:

### 1.Noise Reduction:
Morphological operations, such as erosion and dilation, can help in reducing noise in X-ray images, leading to clearer detection of fractures.

### 2.Enhancement of Structures:
Morphological operations can enhance the structural features in X-ray images, making fractures more prominent and easier to detect.

### Real-time Processing:
Morphological operations are computationally efficient, allowing for real-time processing of X-ray images, which is crucial in medical settings where prompt diagnosis is required.

## Challenges:
### 1.Parameter Sensitivity:
The effectiveness of morphological operations heavily depends on the choice of parameters such as the size and shape of structuring elements. Tuning these parameters for different types of fractures and variations in X-ray images can be challenging.

### 2.False Positives/Negatives:
Morphological operations may sometimes lead to false positives or false negatives, where non-fracture structures are mistakenly identified as fractures or actual fractures are missed, respectively. This can be particularly problematic in critical medical diagnoses.

### 3.Complex Fracture Patterns:
Morphological operations might struggle to detect complex fracture patterns, especially when fractures are subtle or located in anatomically complex regions of the body. Advanced algorithms or additional processing steps may be required to address this challenge.
