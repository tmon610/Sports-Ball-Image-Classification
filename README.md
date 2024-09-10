This repository contains the code and resources for a mini-project submission, comparing traditional and deep learning methods for sports ball image classification. The project explores and evaluates classical feature-based methods like SVM with SIFT/HoG and deep learning methods using CNN architectures (e.g., ResNet, EfficientNet, MobileNet) to classify sports balls from a dataset containing over 9000 images across 15 different sports categories.

The goal of this project is to compare the performance of traditional machine learning models and modern deep learning models in classifying sports balls. The comparison focuses on accuracy, robustness to image perturbations (noise, blur, occlusion, etc.), and model complexity.

We implemented two approaches:

Classical Approach: SVM with feature descriptors (SIFT, HoG).
Deep Learning Approach: Pre-trained Convolutional Neural Networks (ResNet, EfficientNet, MobileNet).
Dataset
The dataset used is the Sports Balls Multiclass Image Classification Dataset from Kaggle. It consists of over 9000 images from 15 sports categories, such as football, baseball, tennis, and basketball.

Training Data: 80% of the dataset
Test Data: 20% of the dataset
Validation Data: A subset of the training data for hyperparameter tuning.
Methods
Classical Approach (SVM)

Feature extraction using SIFT, HoG, ORB, and BRISK descriptors.
Bag of Visual Words (BoVW) to represent images as fixed-length vectors.
Multi-class classification using Support Vector Machines (SVM).
Deep Learning Approach

Transfer learning using pre-trained models like ResNet, EfficientNet, and MobileNet.
Customization of the final fully connected layers for the 15-class classification task.
Data augmentation techniques to improve robustness and generalization.

Results
The ResNet-26D model achieved the highest classification accuracy of 85%, outperforming other models in terms of robustness to image perturbations.
The SVM with SIFT features achieved a maximum accuracy of 30.7%, performing better than the baseline model.
Detailed performance metrics, accuracy plots, and robustness evaluations are provided in the results section of the code.

Conclusion
Deep learning models outperform classical methods in terms of accuracy and robustness, especially in noisy and occluded images. However, traditional models like SVM are computationally less expensive and can still provide valuable insights in simpler or less resource-intensive applications.

Future Work
Future work could focus on:

Real-time sports ball tracking in videos.
Further optimization of deep learning models for faster inference.
Extending the dataset to include more sports or similar classification tasks.
