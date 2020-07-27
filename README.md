# rock-paper-scissers-classification
image classification of rock paper scissor hands

In this post we are going to setup a simple CNN to be able to classify images of hands playing rock, paper, scissor game. This data-set will be loaded from tensorflow_datasets module

Use pip install tensorflow-datasets if you don’t have this module installed already.

UNDERSTANDING THE DATA-SET
The data-set contains images of people playing the rock, paper scissor games as shown in the picture below.It consist of 2,892 images having only train and test splits. Each image has a shape of [300, 300, 3] with 3 output classes(i.e rock, scissor, paper

LOADING THE DATA-SET
To load the data-set the first thing we will need to do is import the necessary libraries. We will then use the tfds.load() to load (downloads and then load on the first time)our data-set while setting with _info and as_supervised to True.

Let’s try to go over some parts of the code that might not be clear

While using the tfds.load() on line 8, setting with_info = True returns information about our data-set which is then stored in the variable we declared (i.e info).
as_supervised = True loads our data-set as a(image, label) tuple structure.
PREPROCESS THE DATA
Remember we have just train and test split we need to get our validation split. We will use 10% of the train data as our validation split.

Before feeding our data into the CNN it will have to go through some form of preprocessing.

Each pixels of the image in our data-set ranges from 0 to 255 which we will scale to between 0 and 1 with the help of a small function

We assign the available split to their respective variable.
A function was created to scale the pixel of the images
tf.cast ensures that our images are of type float32
Diving by 255. scales our pixel to between 0 and 1 in float format
Using the map() method on each data applies our preprocess function to each of our data
The above code Takes care of getting the validation data from the train data

The valid_train data was shuffled with a buffer_size of 1000
valid_data contains 10% of the train data
train_data contains the remaining 90%
Batch size of test data was set which is just the same as using the total number of test data as the batch size since the data-set is a small one
FORMATTING
A little bit of formatting is all that is left before we are good to go.

Each data was batched and by using prefetch(1) our data-set will always be one batch ahead
iter() goes through the data to separate image and label remember we set as_supervise = True
next() moves on to the next batch
The images were resized from (300 by 300) to (150 by 150)
train_images.shape returns TensorShape([32, 150, 150, 3]) The first item is our batch_size while the shape of the images in our data-set is [150,150,3] i.e our input_shape
DATA AUGMENTATION
We are dealing with small sets of data hence data augmentation will be of great help to us

imagedatagenerator has some parameters you can set like width shift, horizontal flip etc.
datagen goes through both the train images and train labels with a batch size of 32
NOTE: Don’t use data augmentation on both your validation and test data
The above code will create random images of different variation depending on the parameter that was set. for instance we set horizontal_flip as True this means that some images will be flipped horizontally

Data augmentation is a great way to allow your model train on variations of images especially when you have small a data-set

MODEL
After preprocessing and formatting we are now ready to build our model at last. We will definitely start by importing some necessary libraries before building our CNN

Our first layer consist of 32 number of filters each of size 3 (i.e 3 x 3) and our input_shape variable also goes to the first layer
The next layer consist of maxpooling with a pool size of 2
The same was repeated, with an increase in the number of filters 64,128
for the layers having (padding = same)each image will be padded
Kernel_initializer is use to randomness weight at the start of the training
activation function used was ‘relu’
flatten layer flattens the shape of our images before passing on to the Dense layer
We have just one Dense hidden layer with an hidden unit of 128
We expects our model to return a three output, so our output layer contains 3 hidden unit with a ‘softmax’ activation function
The above code will return a summary of our model where we can see the layers and total parameters both trainable and Non-trainable

We then need to compile our model and fit it.

Adam optimizer was used with a learning rate of 0.001
we are using ‘sparse_categorical_crossentropy’ has our loss function
Early stopping callback was also implemented to monitor our val_loss and to stop the model if there is no increase in val_loss for 10 epochs
model.fit() will start the training of our model
epochs was set to 200 but don’t worry since we are using early_stopping callback our model will stop was it starts to overfit
After some time the validation accuracy will be about 94%. Not bad at all!

Let’s see how both the loss and accuracy curves looks like by plotting them

The result shows the effect of early_stopping as it stopped us from overfitting.

There is a steady decrease in both the train and the validation loss curves and also an increase in both of their accuracy

You can then go on to test it on the test_data

The results shows a test accuracy of about 98% .

CONCLUSION

We were able to setup a CNN model and train it on images of hands playing the rock, paper, scissor game to attain a model accuracy of 98% .
