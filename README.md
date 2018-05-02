# Classification of Dogs and Cats

Download the data from kaggle Dogs vs Cats Competition.
https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data


### Data Description
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id. For each image in the test set, you should predict a probability that the image is a dog (1 = dog, 0 = cat).

### Data Preprocessing:
First divided the dataset into the following sub folders:

train – This is the training set used for training the model. Consists of 11500 images of cats and 11500 images of dogs. 

test1 – Final test set accuracy to determine the accuracy of the model. Consists of 12500 images of both Cats and Dogs. 

valid – Validation set for checking the models accuracy. Consists of 1000 cats and 1000 dogs images.

sample – Small dataset on which testing can be done so that we can save a lot of time and check the convergence of model. Consists of 12 images of each.

### Sample Data

Cats

![samplecats](https://user-images.githubusercontent.com/19996897/39514811-e8224404-4e15-11e8-9440-637536201f39.PNG)

Dogs

![sampledogs](https://user-images.githubusercontent.com/19996897/39514812-e8502b8a-4e15-11e8-88c4-fd836c311f9d.PNG)


### Softwares and Packages used: 

Tensorflow, Python, Opencv, scikit learn.

### Training the model
Using Convolutonal Layers we train the model and test the model accuracy on test data.
First used simple layer of 2 convolutional network and got an accuracy of 78%.
Next used 8 layers Convolutional networks and got an accuracy of 92%.
Next using Pretrained model VGG16 got an accuracy of 98%.
Next using Pretrained model RESNET50 got an accuracy of 98.5%.

## For getting State of Art results for this problem, I followed the techniques presented in the fast.ai. These are the techniques I used for getting these results.

##### We're going to use a pre-trained model, that is, a model created by some one else to solve a different problem. Instead of building a model from scratch to solve a similar problem, we'll use a model trained on ImageNet (1.2 million images and 1000 classes) as a starting point. The model is a Convolutional Neural Network (CNN), a type of Neural Network that builds state-of-the-art models for computer vision. 

### Pre Trained Model used:

RESNET34

Architecture:

![resnet34](https://user-images.githubusercontent.com/19996897/39515061-98a13d76-4e16-11e8-8c78-e5bd9b2cba1d.jpg)

Then we start training the model by selecting a lower learning rate value.

### Here are the results :

These are the results after first training session:

[0.02893] – Training error  
0.987 – Validation accuracy

A few correct labels at random

![dogscatsrandomresults](https://user-images.githubusercontent.com/19996897/39515269-3621a8d8-4e17-11e8-81cf-9c096a1320b1.PNG)

A few incorrect labels at random

![incorrectresults](https://user-images.githubusercontent.com/19996897/39515265-359ca034-4e17-11e8-93e3-a44d3b114435.PNG)

Most correct cats
![catsresults1](https://user-images.githubusercontent.com/19996897/39515268-35f98358-4e17-11e8-9558-f073248efe2e.PNG)

Most correct dogs
![dogsresults1](https://user-images.githubusercontent.com/19996897/39515260-35213c6e-4e17-11e8-9153-98f9d6a16ac4.PNG)

Most incorrect cats
![incorrectcats](https://user-images.githubusercontent.com/19996897/39515261-35497e40-4e17-11e8-9085-252367704824.PNG)

Most incorrect dogs
![incorrectcats](https://user-images.githubusercontent.com/19996897/39515261-35497e40-4e17-11e8-9085-252367704824.PNG)

Most uncertain predictions
![uncertain](https://user-images.githubusercontent.com/19996897/39515267-35cd4086-4e17-11e8-8eb4-18ece8186f0c.PNG)
