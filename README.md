# Classification of Dogs and Cats

Download the data from kaggle Dogs vs Cats Competition.
https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

Data Description

The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id. For each image in the test set, you should predict a probability that the image is a dog (1 = dog, 0 = cat).


Using Convolutonal Layers we train the model and test the model accuracy on test data.

First used simple layer of 2 convolutional network and got an accuracy of 78%

Next used comlex Convolutional networks and got an accuracy of 92%

Next using Pretrained model VGG16 got an accuracy of 98%

Next using Pretrained model RESNET50 got an accuracy of 98.9%

