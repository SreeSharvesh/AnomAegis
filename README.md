# **AnomAegis - Network Traffic Anomaly Detection System**  

## **Project Overview**  
AnomAegis is an advanced **network anomaly detection system** designed to identify and mitigate **DDoS attacks** by analyzing network traffic patterns. With the increasing number of IoT devices and their vulnerability to cyber threats, attackers often exploit them to form botnets for **Distributed Denial of Service (DDoS) attacks**. This project aims to detect such attacks by employing **autoencoder-based deep learning models** and traditional **machine learning anomaly detection techniques**.  

In real-world scenarios, anomaly detection is only effective when performed in **streaming or near real-time environments**. To achieve this, we train an **autoencoder** exclusively on normal network traffic, allowing it to learn an optimal representation of legitimate behavior. When the model encounters an attack, its representation deviates significantly from normal traffic, enabling the detection of anomalies.  

## **Key Features**  
✔ **Autoencoder-Based Detection** – Learns representations of normal traffic and flags deviations as potential attacks.  
✔ **Machine Learning Models for Anomaly Detection** – Includes techniques like **Isolation Forest, Local Outlier Factor (LOF), and Support Vector Machines (SVM)**.  
✔ **Minimal Data Requirement** – Trains effectively with only 8000 samples of normal traffic, reducing the need for extensive datasets.  
✔ **Low-Latency & Efficient Detection** – Optimized for real-time or near real-time attack detection.  
✔ **Visualization & Analysis** – Utilizes **t-SNE** for visualizing the separability of normal and attack traffic in latent space.  
✔ **Scalable & Extensible** – The system can be extended to detect other network anomalies beyond DDoS attacks.  

## **Methodology**  

### **1. Autoencoder-Based Approach**  
- A **simple neural network** is designed with an **input and output layer of identical dimensions** to capture the structure of normal network traffic.  
- The autoencoder is trained only on **non-fraud (normal) cases**, allowing it to learn an optimal representation of legitimate traffic.  
- When presented with DDoS attack traffic, the autoencoder's reconstruction error is significantly higher, making it possible to **distinguish attacks from normal behavior**.  
- **Latent space representations** of normal and attack traffic are extracted to improve anomaly detection accuracy.  
- A **simple linear classifier** or a **multi-class neural network** (with softmax outputs) can be trained on the extracted representations.  

### **2. Machine Learning-Based Anomaly Detection Techniques**  

#### **a) Isolation Forest**  
- An ensemble learning method specifically designed for anomaly detection.  
- It isolates anomalies by randomly selecting a feature and a split value, making DDoS attacks easier to detect.  

#### **b) Density-Based Anomaly Detection (Local Outlier Factor - LOF)**  
- Assumes that normal data points exist in dense neighborhoods while anomalies (DDoS attacks) lie in sparse regions.  
- Uses **k-Nearest Neighbors (k-NN) and reachability distance** to measure deviation.  

#### **c) Clustering-Based Anomaly Detection (K-Means Clustering)**  
- Groups similar data points into clusters and marks outliers as anomalies.  
- Works on the assumption that **DDoS attacks do not conform to normal cluster patterns**.  

#### **d) Support Vector Machine (SVM) for Anomaly Detection**  
- Uses **One-Class SVM** to learn a boundary around normal data and classify points outside this boundary as anomalies.  
- Works well for identifying outliers when labeled data is unavailable.  

## **Results & Visualization**  
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)** is used to visualize high-dimensional latent representations.  
- The model effectively differentiates between **normal traffic and DDoS attack traffic**.  
- Simple classifiers trained on these representations show high accuracy in detecting attacks.  
