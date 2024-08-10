# Age-Prediction-using-Images.

## Problem statement
Age prediction as a CV task is useful for various real-world applications, such as age-restricted content filtering, personalised marketing targeting specific age demographics, enhancing security systems with age verification, and assisting in medical diagnostics and age-related research. The problem statement is to build a model to predict the age of a person, given the image of their face. (Only ImageNet Dataset pre-trained checkpoints can be used for transfer learning.)

## Overview of the dataset 

The dataset was divided into training and testing sets to facilitate model training and evaluation. The training set, consisting of 21,340 images, was used to train the model, while the testing set, comprising 1,950 images, was reserved for evaluating the model's performance on unseen data. Each image is 200x200 pixels containing the faces of different age group persons. The images in the dataset have labels like “image_xx.jpg”.  The dataset is diverse as it has images from various RGB to Grayscale. The quality of images in the dataset varies, reflecting real-world conditions where facial images may exhibit variations in lighting, resolution, and occlusions. 

## Solution

The instructions were given that we are allowed to use only the ImageNet dataset pre-trained checkpoints can be used for transfer learning. In my proposed solution for age prediction, the focus is on utilizing the TF EfficientNet-B4 model variant pre-trained on the JFT dataset (tf_efficientnet_b4.ns_jft_in1k) provided by the Timm library. Timm serves as a comprehensive repository of pre-trained models for computer vision tasks, offering a diverse range of architectures and variants. The TF EfficientNet-B4.ns_jft_in1k model, a convolutional neural network architecture trained on the JFT dataset without the noisy-student (ns) training strategy, is central to our approach. 

The TF EfficientNet-B4.ns_jft_in1k model boasts a compact architecture with 19.3 million parameters, offering efficient computation with 4.5 GMACs (Giga Multiply-Accumulates) and 49.5 million activations. Trained on images of size 380 x 380, it demonstrates remarkable performance in age prediction tasks. In the proposed solution, the TF EfficientNet-B4.ns_jft_in1k pre-trained model is utilized as the backbone architecture for age prediction. By fine-tuning this model on the age prediction dataset, I aim to leverage its robust feature extraction capabilities and transferable representations.

The code implementation revolves around data loading, preprocessing, model training, and evaluation using PyTorch and the Timm library. The dataset is preprocessed, and split into training and validation sets, and the model is trained using L1 loss and Adam optimizer with a learning rate scheduler. Visualization of the model's predictions on validation data is also conducted to assess performance.

Subsequently, the trained model is deployed for inference on the testing dataset, and predictions are generated in CSV format for submission. Through the utilization of the TF EfficientNet-B4.ns_jft_in1k model and the Timm library, the proposed solution aims to achieve state-of-the-art performance in age prediction while ensuring efficiency and scalability.


## References

[1] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
[2] Self-training with Noisy Student improves ImageNet classification
[3] https://huggingface.co/timm/tf_efficientnet_b4.ns_jft_in1k
