# IMDb-Sentiment-Analysis
This is a Sentiment Analysis Model built using Machine Learning and Deep Learning to classify movie reviews from the IMDB dataset into "positive" and "negative" classes.
Models. I built and experimented with different models to compare their performance on the dataset - <br>

### Long Short-Term Memory Network:<br>
Recurrent Neural Networks are especially suited for sequential data (sequence of words in this case). Unlike the more common feed-forward neural networks, an RNN does not input an entire example in one go. Instead, it processes a sequence element-by-element, at each step incorporating new data with the information processed so far. This is quite similar to the way humans too process sentences - we read a sentence word-by-word in order, at each step processing a new word and incorporating it with the meaning of the words read so far.
LSTMs further improve upon these vanilla RNNs. Although theoretically RNNs are able to retain information over many time-steps ago, practically it becomes extremely difficult for simple RNNs to learn long-term dependencies, especially in extremely long sentences and paragraphs. LSTMs have been designed to have special mechanisms to allow past information to be reutilised at a later time. As a result, in practise, LSTMs are almost always preferable over RNNs.
Here, I built an LSTM model using Keras Sequential API. A summary of the model and its layers is given below. The model was trained with a batch size of 64, using the Adam Optimizer.
<br>
While tuning the hyper-parameters, a Dropout layer was introduced as measure of regularization to minimize the overfitting of the model on the training dataset. A separate validation set (taken from the training data) was used to check the performance of the model during this phase.
<br>
>Results: This model managed to achieve an accuracy of 76.62% when evaluated on the hidden test dataset.

### Convolutional Network:<br>
The idea of Convolutional Networks has been quite common in Computer Vision. The use of convolutional filters to extract features and information from pixels of an image allows the model to identify edges, colour gradients, and even specific features of the image like positions of eyes & nose (for face images). Apart from this, 1D Convolutional Neural Networks have also proven quite competitive with RNNs for NLP tasks. Given a sequential input, 1D CNNs are well able to recognize and extract local patterns in this sequence. Since the same input transformation is performed at every patch, a pattern learned at a certain position in the sequence can very easily later be recognized at a differnt position. Further, in comparison to RNNs, ConvNets in general are extremely cheap to train computationally - In the current project (built using Google Colaboratory with a GPU kernel), the LSTM model took more than 30 minutes to complete an epoch (during training) while the CNN model took hardly 9 seconds on average!<br>
I built the model using Keras Sequential API. A summary of the model and its layers is below.
<br>
>Results: This model was trained with a batch size of 64 using Adam Optimizer. The best model (weights and the architecture) was saved during this phase. This model achieved an accuracy of 89.7 % on the test dataset, a good increase over the LSTM model.
