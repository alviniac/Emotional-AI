# Emotional-AI

In this project, I built an emotion recognition system that takes human face images as inputs and classifies them across discrete emotion categories. I trained a deep learning model using convolutional neural networks on Kaggle’s facial expression recognition data set (found here https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) and experimented with different model architectures.  The end result was a model that classifies Kaggle’s facial image data with 55% validation accuracy, which is short of the state of the art accuracy of 61%, though the winner of the Kaggle competition managed to achieve 71% accuracy.

<a href = "https://github.com/alviniac/Emotional-AI/blob/master/transfer%20learning%20model.ipynb">"transfer learning model.pynb”</a> –<p>
According to many deep learning experts, in practice, models are rarely trained from scratch.  It is recommended to first try transfer learning, which entails leveraging the power of models that have already been trained on extremely large datasets.  Although the model may be trained on unrelated data, oftentimes it may still work well.  I used pre-trained network weights from VGG-16 and VGG-19, trained on ImageNet, as fixed feature extractors.  I removed the final fully connected layer from these models, so that the end of the model outputs features rather than predictions.  The bottleneck features extracted from these models were then used as inputs to a simple two-layer model, which outputs the predictions.  Results weren’t good with a 38% accuracy – I believe this is because ImageNet data is too different from the facial image data.  Thus, it’s probably better to retrain the model from scratch.  <p>
Unfortunately, since VGG-16 and VGG-19 models have too many parameters, training would be too expensive, so I opted for creating my own CNN architectures for this problem.<p>
<a href = "https://github.com/alviniac/Emotional-AI/blob/master/trying%20different%20models.ipynb">“trying different models.ipynb”</a> –<p>
In this notebook I design my own CNN architectures for the problem.  I went with this standard CNN architecture as noted in this url (http://cs231n.github.io/convolutional-networks/#architectures):<br><center>
INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC<p></center>
Here are some things I found improved the performance of my model:<br>
•	Stacking two CONV - RELU layers before pooling layers instead of just one<br>
•	Increasing number of filters as we get deeper into the network<br>
•	Using two fully connected layers with relu activation function in the end after the convolutional + max pooling layers
<p>
<a href = "https://github.com/alviniac/Emotional-AI/blob/master/final%20model%20training.ipynb">“final model training”</a> –<p>
The best performing model from “trying different models.ipynb” is retrained here except with data augmentation implemented.  With data augmentation the accuracy of the model increased by 3%.  <p>
<a href = "https://github.com/alviniac/Emotional-AI/blob/master/final%20face%20recog%20test.ipynb">“final face recog test”</a>-<p> 
Finally, the model is loaded in this notebook and used to predict the emotion in an image with a face.  I wrote a function that can accept file paths or image urls to output the emotions and respective scores for that image.  

