
# Brain Tumor Classification using Deep Learning Techniques

Brain tumors are localized intracranial lesions, masses, or abnormal cell growths that occupy space within the skull. This project focuses on the classification of brain tumors using deep learning techniques.

## Dataset

The dataset consists of brain tumor images retrieved from Kaggle. It includes three types of tumors:
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor

The dataset contains a total of 5000 images, with 3000 samples used for training and 2000 examples for testing. The original images are of size 512 x 512 pixels, but they have been resized to 128 x 128 grayscale images for feature extraction and model training.

## Methodology

Two deep learning models have been implemented to classify the brain tumors:
1. **XceptionNet**:
   - XceptionNet is a deep convolutional neural network (CNN) architecture inspired by the Inception model.
   - It consists of 71 layers and separates spatial and depthwise convolutions.
   - Accuracy: 92%
   - Loss: 0.31

2. **EfficientNet with Transfer Learning**:
   - EfficientNet is a CNN architecture that uniformly scales all dimensions like depth, width, and resolution using compound scaling.
   - Transfer learning technique is employed where a pre-trained model is used as the starting point for model training.
   - Accuracy: 94%
   - Loss: 0.18

## Usage

1. **Preprocessing**:
   - Use preprocessing techniques to resize the images to 128 x 128 pixels and convert them to grayscale.

2. **Model Training**:
   - Train the XceptionNet and EfficientNet models on the preprocessed dataset.
   - Fine-tune the models if necessary.

3. **Evaluation**:
   - Evaluate the trained models on the test dataset to measure accuracy and loss.

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn

Install the required dependencies using `pip install -r requirements.txt`.

## Acknowledgments

- Kaggle for providing the brain tumor dataset.
- TensorFlow and Keras for deep learning frameworks.
