# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import cog

import tensorflow.keras
import tensorflow as tf
import sys, time, os, warnings
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import re
from PIL import Image
import pickle
from PIL import Image
import time


from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from pickle import load






"""def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = preprocess_input(img)
    return img, image_path"""

image_model = tf.keras.applications.VGG16(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)



def calc_max_length(tensor):
    return max(len(t) for t in tensor)


BATCH_SIZE = 64
BUFFER_SIZE = 1000
max_length = 33
embedding_dim = 256
units = 512
vocab_size = 8329
features_shape = 512
attention_features_shape = 49
num_steps = 500


# CNN_Encoder
class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 49, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)
        self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)

    def call(self, x):
        #x= self.dropout(x)
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

# RNN_Decoder with Attention Mechanism
class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    self.fc1 = tf.keras.layers.Dense(self.units)

    self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)
    self.batchnormalization = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)

    self.fc2 = tf.keras.layers.Dense(vocab_size)

    # Implementing Attention Mechanism
    self.Uattn = tf.keras.layers.Dense(units)
    self.Wattn = tf.keras.layers.Dense(units)
    self.Vattn = tf.keras.layers.Dense(1)


  def call(self, x, features, hidden):

    # features(VGG-16 output) shape == (batch_size, 49, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, 64, hidden_size)
    # Attention Function
    '''e(ij) = f(s(t-1),h(j))'''
    ''' e(ij) = Vattn(T)*tanh(Uattn * h(j) + Wattn * s(t))'''
    score = self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, 49, 1)
    # you get 1 at the last axis because you are applying score to self.Vattn
    # Then find Probability using Softmax
    '''attention_weights(alpha(ij)) = softmax(e(ij))'''
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    # Give weights to the different pixels in the image
    ''' C(t) = Summation(j=1 to T) (attention_weights * VGG-16 features) '''
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)


    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # Adding Dropout and BatchNorm Layers
    x= self.dropout(x)
    x= self.batchnormalization(x)
    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


###########################################################################
class Predictor(cog.Predictor):
    def setup(self):
      """Load the model into memory to make running multiple predictions efficient"""
      train_captions = load(open('./captions.pkl', 'rb'))
      self.top_k = 5000
      self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.top_k,
		                                                oov_token="<unk>",
		                                                filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

      self.tokenizer.fit_on_texts(train_captions)
      train_seqs = self.tokenizer.texts_to_sequences(train_captions)
      self.tokenizer.word_index['<pad>'] = 0
      self.tokenizer.index_word[0] = '<pad>'

	    
      checkpoint_path = "./ckpt-4"

      #restoring the model
	  
      self.ckpt = tf.train.Checkpoint(encoder=encoder,decoder=decoder,optimizer = optimizer)
      self.ckpt.restore(checkpoint_path)
    
    @cog.input("input", type=cog.Path, help="Image to descripe")
    def predict(self, input):
      """Run a single prediction on the model"""
      # load the photo
        
      ##################################
      # load the photo
      image = load_img(input, target_size=(224, 224))
      # convert the image pixels to a numpy array
      image = img_to_array(image)
      # reshape data for the model
      image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
      # prepare the image for the VGG model
      image = preprocess_input(image)
      ############################
      hidden = decoder.reset_state(batch_size=1)
      temp_input = tf.expand_dims(image[0], 0)
      img_tensor_val = image_features_extract_model(temp_input)
       
      img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
      features = encoder(img_tensor_val)
      dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
      result = []
      
      for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        
        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(self.tokenizer.index_word[predicted_id])

        if self.tokenizer.index_word[predicted_id] == '<end>':
          break
        dec_input = tf.expand_dims([predicted_id], 0)
      ###########################    
    
      for i in result:
      	if i=="<unk>":
      		result.remove(i)
      	else: pass
      #Remove startseq and endseq
      result =' '.join(result).rsplit(' ', 1)[0]
      return result 

