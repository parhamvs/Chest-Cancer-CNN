This project is a breast cancer classification project using deep learning. The project utilizes convolutional neural networks (CNNs) to classify mammogram images into four different categories: normal, benign, in situ carcinoma, and invasive carcinoma.

# Dataset
The dataset used for this project is publicly available on Kaggle (https://www.kaggle.com/uciml/breast-cancer-wisconsin-data). It contains mammogram images with labels indicating the category of the breast cancer.

# Preprocessing
The images in the dataset are preprocessed by rescaling their pixel values to the range of 0 to 1. The images are also resized to 224x224 pixels to be used as input for the CNN.

# Training
The CNN architecture used for this project consists of three convolutional layers with ReLU activation, followed by max pooling layers. The output of the convolutional layers is flattened and fed into two fully connected layers with ReLU activation, and a softmax activation output layer.

The model is compiled with Adam optimizer, categorical crossentropy loss function, and accuracy metric. The model is trained for 10 epochs on the training dataset.

# Evaluation
The trained model is evaluated on the test dataset, which contains mammogram images that were not seen by the model during training. The accuracy and loss metrics are calculated and plotted using the Matplotlib library.
