# Lab 5: Face Detection and Clustering using K-Means

## Table of Contents
- [Aim](#aim)
- [Methodology](#methodology)
- [Implementation Details](#implementation-details)
- [Key Findings](#key-findings)
- [Conclusions](#conclusions)
- [Technologies Used](#technologies-used)
- [Setup and Usage](#setup-and-usage)

---

## Aim

The primary objective of this lab is to implement a face detection and clustering system that:
1. Detects faces in group images using Haar Cascade Classifiers
2. Extracts color-based features (Hue and Saturation) from detected faces
3. Groups similar faces using K-Means clustering algorithm
4. Classifies a template face image into the identified clusters

---

## Methodology

### 1. Face Detection
- **Haar Cascade Classifier** is used to detect faces in the input image (`plaksha_Faculty.jpg`)
- The classifier uses pre-trained models from OpenCV to identify facial regions
- Detection parameters are tuned to optimize face detection accuracy:
  - Scale factor: 1.05
  - Minimum neighbors: 4
  - Face size range: 15x15 to 45x45 pixels

### 2. Feature Extraction
- Detected face regions are converted from RGB to HSV color space
- Two key features are extracted for each face:
  - **Hue**: Represents the color type (average value across the face region)
  - **Saturation**: Represents the color intensity (average value across the face region)
- These features form a 2D feature vector for each detected face

### 3. K-Means Clustering
- The extracted features are used to cluster faces into **3 groups**
- K-Means algorithm groups faces with similar Hue-Saturation characteristics
- Cluster centroids represent the average feature values for each group

### 4. Template Classification
- A template image (`Dr_Shashi_Tharoor.jpg`) is processed similarly
- Features are extracted and compared with existing clusters
- The template is assigned to the nearest cluster based on its Hue-Saturation values

---

## Implementation Details

### Step 1: Face Detection and Annotation
```python
# Load and convert image to RGB
img = cv.imread('plaksha_Faculty.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Detect faces using Haar Cascade
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces_rect = face_cascade.detectMultiScale(img, 1.05, 4, minSize=(15, 15), maxSize=(45,45))

# Draw rectangles around detected faces
for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
```

**Result**: Multiple faces detected and marked with red bounding boxes.

### Step 2: Feature Extraction
```python
# Convert to HSV color space
img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

# Extract Hue and Saturation for each face
hue_saturation = []
for (x, y, w, h) in faces_rect:
    face = img_hsv[y:y + h, x:x + w]
    hue = np.mean(face[:, :, 0])
    saturation = np.mean(face[:, :, 1])
    hue_saturation.append((hue, saturation))
```

### Step 3: K-Means Clustering
```python
# Perform K-Means clustering (k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(hue_saturation)

# Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
```

### Step 4: Visualization
The project includes multiple visualizations:
- **Face images plotted on Hue-Saturation space**: Shows each detected face at its corresponding feature coordinates
- **Scatter plot with cluster assignments**: Different colors represent different clusters
- **Centroid markers**: X-marks showing the center of each cluster
- **Template image classification**: Purple marker showing where the template image fits in the feature space

---

## Key Findings

### 1. Face Detection Accuracy
- The Haar Cascade classifier successfully detected faces in a group photograph
- Tuning the `scaleFactor`, `minNeighbors`, and size constraints was crucial for optimal detection
- Small faces (15x15 to 45x45 pixels) were effectively detected with the chosen parameters

### 2. Feature Distribution
- Hue and Saturation provide meaningful features for facial clustering
- Faces cluster based on lighting conditions, skin tones, and color characteristics
- The feature space shows clear separation between different groups

### 3. Clustering Results
- **K-Means with k=3** successfully grouped faces into distinct clusters
- Cluster 0 and Cluster 1 show different distributions in the Hue-Saturation space
- Centroids provide representative average values for each cluster
- The clustering helps identify faces with similar color characteristics

### 4. Template Classification
- The template image (Dr. Shashi Tharoor) was successfully classified into one of the existing clusters
- The classification is based on Euclidean distance to cluster centroids
- The template's position in the feature space indicates its similarity to other faces in the assigned cluster

---

## Conclusions

1. **Haar Cascade Effectiveness**: Haar Cascade classifiers remain an efficient method for real-time face detection, especially when computational resources are limited.

2. **Color-Based Features**: Hue and Saturation from HSV color space provide simple yet effective features for basic face grouping, particularly useful for identifying faces under similar lighting conditions.

3. **K-Means Performance**: K-Means clustering successfully groups faces with similar color characteristics, though the optimal number of clusters (k=3 in this case) may vary depending on the dataset.

4. **Limitations and Improvements**:
   - Color-based features are sensitive to lighting variations
   - More robust features (e.g., facial landmarks, deep learning embeddings) would improve clustering accuracy
   - The Haar Cascade may miss faces at certain angles or under poor lighting

5. **Real-World Applications**:
   - Photo organization and grouping
   - Quick face grouping for event management
   - Preprocessing step for more sophisticated face recognition systems

6. **Distance Metrics in Classification**:
   - Common distance metrics include **Euclidean**, **Manhattan**, and **Minkowski** distances
   - These metrics are essential for K-NN and clustering algorithms
   - Cross-validation helps evaluate model performance on unseen data and prevents overfitting

7. **Bias-Variance Tradeoff in KNN**:
   - Low K values lead to high variance (overfitting to training data)
   - High K values lead to high bias (oversimplified model)
   - Proper K selection balances this tradeoff

