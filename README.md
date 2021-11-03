# Late-Blight-Detection
This repository shows  Machine Learning and Deep Learning techniques to classify ten states of the late blight disease severity in potato fields.  The classification is performed using UAV's multispectral images, these images were acquired at International Potato Center (CIP) fields.

Classes Definition:

-Class 1: 0-10% Severity
-Class 2: 10-20% Severity
-Class 3: 20-30% Severity
-Class 4: 30-40% Severity 
-Class 5: 40-50% Severity
-Class 6: 50-60% Severity
-Class 7: 60-70% Severity
-Class 8: 70-80% Severity
-Class 9: 80-90% Severity
-Class 10: 90-100% Severity 
-Class 11: Soil and others

Input Data: 

- Multispectral Image (RGB, NIR, RED EDGE)
- Vegetation Indexes (NDVI and NDRE)
- Five Haralick textural features (Energy, Entropy, Correlation, Inverse Difference Moment, Inertia )

Output Data:
- Classified image in .tif format
