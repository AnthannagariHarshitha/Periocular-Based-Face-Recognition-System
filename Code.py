import cv2
import numpy as np
from mtcnn import MTCNN
from skimage import feature
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# Helper functions
def resize_image(image, new_size=(128, 128)):
    return cv2.resize(image, new_size)

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def histogram_equalization(image):
    return cv2.equalizeHist(image)

def canny_edge_detection(image):
    return cv2.Canny(image, 100, 200)

def gabor_filter(image, ksize=31, sigma=4.0, theta=0.0, lambd=10.0, gamma=0.5):
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma)
    return cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)

def compute_lbp(image):
    lbp = feature.local_binary_pattern(image, P=8, R=1, method='uniform')
    return lbp

def compute_hog(image):
    hog_descriptor, hog_image = feature.hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return hog_descriptor, hog_image

def normalize_image(image):
    return normalize(image.reshape(-1, 1)).reshape(image.shape)

# Non-Maximum Suppression (NMS) function
def non_max_suppression(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)  # Ensure yy2 and yy1 have the same shape
        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

# Face detection function using MTCNN
def detect_faces_mtcnn(image):
    detector = MTCNN()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces_mtcnn = detector.detect_faces(rgb_image)
    
    boxes = []
    for face in faces_mtcnn:
        x, y, w, h = face['box']
        boxes.append((x, y, x + w, y + h))
    
    # Apply NMS to filter out overlapping bounding boxes
    if len(boxes) > 0:
        boxes = non_max_suppression(np.array(boxes))
    else:
        boxes = np.empty((0, 4), int)  # Create an empty array with shape (0, 4) if no faces are detected
    return boxes

# Function to enhance the image for better face detection
def enhance_image(image):
    grayscale_image = convert_to_grayscale(image)
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    sharpened_image = cv2.addWeighted(grayscale_image, 1.5, blurred_image, -0.5, 0)
    equalized_image = histogram_equalization(sharpened_image)
    return equalized_image

# Main function to process an image
def main(image_path):
    # Load and preprocess image
    image = cv2.imread(image_path)
    enhanced_image = enhance_image(image)

    # Face detection using MTCNN
    faces_mtcnn = detect_faces_mtcnn(image)
    face_count_final = len(faces_mtcnn)

    # Print the total number of detected faces
    print(f"Total number of faces detected: {face_count_final}")

    # Draw rectangles around detected faces
    for (x, y, x2, y2) in faces_mtcnn:
        cv2.rectangle(image, (x, y), (x2, y2), (255, 0, 0), 2)

    # Additional image preprocessing and feature extraction
    resized_image = resize_image(image)
    grayscale_image = convert_to_grayscale(resized_image)
    equalized_image = histogram_equalization(grayscale_image)

    # Extract features
    edges = canny_edge_detection(equalized_image)
    gabor_response = gabor_filter(equalized_image)
    lbp_features = compute_lbp(equalized_image)
    hog_descriptor, hog_image = compute_hog(equalized_image)

    # Normalize image
    normalized_image = normalize_image(equalized_image)

    # Display results
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 4, 1), plt.title('Original Image'), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 4, 2), plt.title('Grayscale Image'), plt.imshow(grayscale_image, cmap='gray')
    plt.subplot(2, 4, 3), plt.title('Equalized Image'), plt.imshow(equalized_image, cmap='gray')
    plt.subplot(2, 4, 4), plt.title('Edges'), plt.imshow(edges, cmap='gray')
    plt.subplot(2, 4, 5), plt.title('Gabor Response'), plt.imshow(gabor_response, cmap='gray')
    plt.subplot(2, 4, 6), plt.title('LBP'), plt.imshow(lbp_features, cmap='gray')
    plt.subplot(2, 4, 7), plt.title('HOG'), plt.imshow(hog_image, cmap='gray')
    plt.subplot(2, 4, 8), plt.title('Detected Faces'), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

# Run the main function with the provided image path
image_path = 'crowd.jpg'  # Update with the path to your uploaded image
main(image_path)
