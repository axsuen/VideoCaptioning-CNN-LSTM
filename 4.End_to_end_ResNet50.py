"""Trained on NSCC GPU Resource"""

import json
import numpy as np
import pandas as pd
import os
import functools,operator
import random
import matplotlib.pyplot as plt
import keras

# tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Embedding, 
    TimeDistributed, Dense, RepeatVector, 
    Activation, Flatten, Reshape, concatenate,  
    Dropout, BatchNormalization, Bidirectional, Masking)
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import add
from keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import AdditiveAttention, Attention
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import GRU
from keras.regularizers import l2


from tensorflow.keras.layers import GlobalAveragePooling2D

import cv2

# define file paths
datafile_path = '/mnt/data'
train_path = '/mnt/data/train/train'
test_path = '/mnt/data/test/test'
checkpoint_path = '/mnt/output'
save_model_path = '/mnt/output'
submission_path =  '/mnt/output'

with open(os.path.join(datafile_path, "training_annotation.json")) as file:
    training_annotation = json.load(file)

with open(os.path.join(datafile_path, "object1_object2.json")) as file:
    object1_object2 = json.load(file)
    
with open(os.path.join(datafile_path, "relationship.json")) as file:
    relationship = json.load(file)

def get_relationship(val):
    for key, value in relationship.items():
        if val == value:
            return key
    return "relationship key doesn't exist"

def get_object(val):
    for key, value in object1_object2.items():
        if val == value:
            return key
    return "object key doesn't exist"

def save_descriptions(training_annotation, filename):
  lines = list()
  for key in training_annotation.keys():
    item = training_annotation[key]
    object1 = get_object(item[0])
    relationship = get_relationship(item[1])
    object2 = get_object(item[2])
    lines.append(key + ' ' + object1 + ' ' + relationship + ' ' + object2)

  data = '\n'.join(lines)
  file = open(filename, 'w')
  file.write(data)
  file.close()

filename = os.path.join(datafile_path, 'train/descriptions.txt')
save_descriptions(training_annotation, filename)

file = open(filename, 'r')
text = file.read()

# process line by line
descriptions = {}
for line in text.split('\n'):
  tokens = line.split()
  image_id, image_desc = tokens[0], tokens[1:]
  desc = '<BOS> ' + ' '.join(image_desc) + ' <EOS>'
  descriptions[image_id] = desc


# tokenize description

output_length = 5

vectorizer = TextVectorization(standardize = None, output_sequence_length=output_length)
text_ds = tf.data.Dataset.from_tensor_slices([descriptions[i] for i in descriptions])
vectorizer.adapt(text_ds)
for i in descriptions:
  descriptions[i] = vectorizer([[descriptions[i]]])

voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))
index_word = dict(zip(range(len(voc)), voc))

np.random.seed(5242)

# Train test split
train_split = 0.85

descriptions_random = descriptions.copy()
descriptions_random = list(descriptions.items())
random.shuffle(descriptions_random)
training_list = descriptions_random[:int(len(descriptions) * train_split)]
validation_list = descriptions_random[int(len(descriptions) * train_split):]

training_dict = dict(training_list)
validation_dict = dict(validation_list)

# declare variables

"""
time_steps_encoder is the number of frames per video we will be using for training
num_encoder_tokens is the number of features from each frame
latent_dim is the number of hidden features for lstm
time_steps_decoder is the maximum length of each sentence
num_decoder_tokens is the final number of tokens in the softmax layer
"""

epochs = 30
batch_size = 20
time_steps_encoder = 30
num_encoder_tokens = 2048
latent_dim = 512
height = 224
width = 224
channel = 3
shape = (time_steps_encoder, height, width, channel)
time_steps_decoder = output_length
num_decoder_tokens = len(voc)

# cnn pre-trained models

def inceptionv3(shape = (299, 299, 3)):
  model = tf.keras.applications.InceptionV3(
      include_top=False,
      input_shape = shape,
      weights='imagenet'
  )

  # train last 5 layers
  for layer in model.layers[:-5]:
    layer.trainable = False

  output = GlobalAveragePooling2D()

  return Sequential([model, output])

def resnet50(shape = (244, 244, 3)):
  model = tf.keras.applications.ResNet50(
      include_top=False,
      input_shape = shape,
      weights='imagenet'
  )

  # train last 5 layers
  for layer in model.layers[:-10]:
    layer.trainable = False

  output = GlobalAveragePooling2D()

  return Sequential([model, output])

def mobilenetv2(shape = (244, 244, 3)):
  model = tf.keras.applications.MobileNetV2(
      include_top=False,
      input_shape = shape,
      weights='imagenet'
  )

  # train last 5 layers
  for layer in model.layers[:-5]:
    layer.trainable = False

  output = GlobalAveragePooling2D()

  return Sequential([model, output])

# load video input
def load_video(video_id, action):
  if action == 'train':
    img_lst = sorted(os.listdir(os.path.join(train_path, video_id)))
  elif action == 'test':
    img_lst = sorted(os.listdir(os.path.join(test_path, video_id)))
  else:
    print('Error: please indicate train or test')

  frame_lst = []
  for i in img_lst:
    if action == 'train':
      img_path = os.path.join(train_path, video_id, i)
    elif action == 'test':
      img_path = os.path.join(test_path, video_id, i)
    else:
      print('Error: please indicate train or test')
      
    img = cv2.imread(img_path)
    img = cv2.resize(img, (height, height))
    # img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.keras.applications.resnet.preprocess_input(img)
    # img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    frame_lst.append(img)
  
  return np.array(frame_lst)

def load_desc(video_id):
  desc = to_categorical(descriptions[video_id][0], num_decoder_tokens)
  return desc

# custom generator class

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=batch_size, frames = 30, height = height, width = width, channel = channel, num_decoder_tokens = num_decoder_tokens, output_length = output_length, shuffle=True):
        'Initialization'
        self.height = height
        self.width = width
        self.channel = channel
        self.batch_size = batch_size
        self.frames = frames
        self.list_IDs = list_IDs
        self.num_decoder_tokens = num_decoder_tokens
        self.output_length = output_length
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x1, x2, y = self.__data_generation(list_IDs_temp)

        return [x1, x2], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        x1 = np.empty((self.batch_size, self.frames, self.height, self.width, self.channel))
        x2 = np.empty((self.batch_size, self.output_length, self.num_decoder_tokens), dtype=int)
        y = np.empty((self.batch_size, self.output_length, self.num_decoder_tokens), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            x1[i,] = load_video(ID, 'train')
            x2[i,] = load_desc(ID)

            # Store class
            y[i,] = load_desc(ID)
        
        return x1, x2, y



dropout_rate = 0.5
weight_decay = 0.03


# set up encoder  
input = Input(shape = shape)
# covnet = inceptionv3(shape[1:])
covnet = resnet50(shape[1:])
# covnet = mobilenetv2(shape[1:])
convnet_timedistributed = TimeDistributed(covnet, input_shape = shape)
covnet_dense = TimeDistributed(Dense(1024, activation = 'relu'))
covnet_output = convnet_timedistributed(input)
covnet_dropout = TimeDistributed(Dropout(dropout_rate))
covnet_output = covnet_dropout(covnet_output)
covnet_output = covnet_dense(covnet_output)
encoder = CuDNNLSTM(latent_dim, kernel_regularizer = l2(weight_decay), bias_regularizer = l2(weight_decay), return_state=True, return_sequences=True, name='encoder_lstm')

_, state_h, state_c = encoder(covnet_output)
encoder_states = [state_h, state_c]

# set up decoder
decoder_inputs = Input(shape=(time_steps_decoder, num_decoder_tokens), name = 'decoder_inputs')
decoder_lstm = CuDNNLSTM(latent_dim, kernel_regularizer = l2(weight_decay), bias_regularizer = l2(weight_decay), return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_relu')
decoder_outputs = decoder_dense(decoder_outputs)


# Checkpoint
checkpoint_filepath = os.path.join(checkpoint_path, 'checkpoint_2.ckpt')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='auto',
    save_best_only=True)

# Early Stopping
earlystopping = EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True)

# Optimizer
opt = tf.keras.optimizers.Adamax(lr=0.001)

# lr_scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, verbose=1, mode="auto")

# model
model = Model([input, decoder_inputs], decoder_outputs)
model.compile(metrics=['accuracy'], optimizer=opt, loss='categorical_crossentropy')

# # load model checkpoint if needed
# model.load_weights(checkpoint_filepath)


partition = {'train': list(training_dict.keys()), 'validation': list(validation_dict.keys())}


training_generator = DataGenerator(partition['train'])
validation_generator = DataGenerator(partition['validation'])


try:
  model.fit(training_generator, validation_data = validation_generator, validation_steps=(len(validation_dict)//batch_size), epochs=epochs, callbacks=[lr_scheduler, earlystopping, model_checkpoint_callback])
except KeyboardInterrupt:
    print("\nInterrupt received, stopping")
finally:
    pass

if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

encoder_model_name = 'encoder_model_endtoend_2.h5'
decoder_model_name = 'decoder_model_weights_endtoend_2.h5'

# Saving encoder as in training
encoder_model = Model(input, encoder_states)

# Saving decoder states and dense layer 
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

encoder_model.save(os.path.join(save_model_path, encoder_model_name))
decoder_model.save_weights(os.path.join(save_model_path, decoder_model_name))


"""### Inference"""

# re-declare variables

epochs = 30
batch_size = 20
time_steps_encoder = 30
num_encoder_tokens = 2048
latent_dim = 512
height = 224
width = 224
channel = 3
shape = (time_steps_encoder, height, width, channel)

time_steps_decoder = output_length
num_decoder_tokens = len(voc)

encoder_model_name = 'encoder_model_endtoend_2.h5'
decoder_model_name = 'decoder_model_weights_endtoend_2.h5'

# inference encoder model
inf_encoder_model = load_model(os.path.join(save_model_path, encoder_model_name))

# inference decoder model
decoder_inputs = Input(shape=(None, num_decoder_tokens))

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_lstm = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')

decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

inf_decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
inf_decoder_model.load_weights(os.path.join(save_model_path, decoder_model_name))


"""### Exhaustive Search"""

def beam_search(video_id, action):
  node = 50
  all_candidates = []

  # initialise prediction with <BOS>
  encoder_input = load_video(video_id, action)
  encoder_input = np.expand_dims(encoder_input, axis=0)
  states_value = inf_encoder_model.predict(encoder_input)
  target_seq = np.zeros((1, 1, num_decoder_tokens))
  target_seq[0, 0, word_index['<BOS>']] = 1
  output_tokens, h, c = inf_decoder_model.predict([target_seq] + states_value)
  output_tokens = output_tokens.reshape((num_decoder_tokens))
  sampled_token_index = output_tokens.argsort()[-node:][::-1]
  states_value = [h, c]

  # predict for every node
  for i in sampled_token_index:
    if (index_word[i] == '<BOS>') or (index_word[i] == '<EOS>') or (index_word[i] not in object1_object2.keys()):
      continue
    
    else:
      path = []
      score = 0.0

      sampled_char = index_word[i]
      prob = output_tokens[i]
      score = score - np.log(prob)
      path.append(sampled_char)

      target_seq_2 = np.zeros((1, 1, num_decoder_tokens))
      target_seq_2[0, 0, i] = 1
      output_tokens_2, h_2, c_2 = inf_decoder_model.predict([target_seq_2] + states_value)
      output_tokens_2 = output_tokens_2.reshape((num_decoder_tokens))
      sampled_token_index_2 = output_tokens_2.argsort()[-node:][::-1]
      states_value_2 = [h_2, c_2]

      for i in sampled_token_index_2:
        if (index_word[i] == '<BOS>') or (index_word[i] == '<EOS>') or (index_word[i] not in relationship.keys()):
          continue
        
        else:
          path_2 = path.copy()
          score_2 = score.copy()

          sampled_char_2 = index_word[i]
          prob_2 = output_tokens_2[i]
          score_2 = score - np.log(prob_2)
          path_2.append(sampled_char_2)

          target_seq_3 = np.zeros((1, 1, num_decoder_tokens))
          target_seq_3[0, 0, i] = 1
          output_tokens_3, h_3, c_3 = inf_decoder_model.predict([target_seq_3] + states_value_2)
          output_tokens_3 = output_tokens_3.reshape((num_decoder_tokens))
          sampled_token_index_3 = output_tokens_3.argsort()[-node:][::-1]
          states_value_3 = [h_3, c_3]

          for i in sampled_token_index_3:
            if (index_word[i] == '<BOS>') or (index_word[i] == '<EOS>') or (index_word[i] not in object1_object2.keys()):
              continue
            
            else:
              path_3 = path_2.copy()
              score_3 = score_2.copy()

              sampled_char_3 = index_word[i]
              prob_3 = output_tokens_3[i]
              score_3 = score_2 - np.log(prob_3)
              path_3.append(sampled_char_3)
              
              candidate = [path_3, score_3]
              all_candidates.append(candidate)

  ordered = sorted(all_candidates, key=lambda tup:tup[1])
  return ordered

def get_top5_preds(ordered):

  top = 5

  preds = [lst[0] for lst in ordered]
  obj1 = [lst[0] for lst in preds]
  rel = [lst[1] for lst in preds]
  obj2 = [lst[2] for lst in preds]

  obj1_top5 = []
  rel_top5 = []
  obj2_top5 = []

  for i in obj1:
    if i not in obj1_top5:
      obj1_top5.append(i)
    if len(obj1_top5) == top:
      break

  for i in rel:
    if i not in rel_top5:
      rel_top5.append(i)
    if len(rel_top5) == top:
      break

  for i in obj2:
    if i not in obj2_top5:
      obj2_top5.append(i)
    if len(obj2_top5) == top:
      break

  dic_words = {'object1': ' '.join(obj1_top5), 'relationship':' '.join(rel_top5), 'object2:':' '.join(obj2_top5)}

  obj1_top5_idx = [str(object1_object2[word]) for word in obj1_top5]
  rel_top5_idx = [str(relationship[word]) for word in rel_top5]
  obj2_top5_idx = [str(object1_object2[word]) for word in obj2_top5]

  dic_idx = {'object1': ' '.join(obj1_top5_idx), 'relationship':' '.join(rel_top5_idx), 'object2':' '.join(obj2_top5_idx)}
  
  # for validation set format for comparison
  val_format = []
  val_format.append(object1_object2[obj1_top5[0]])
  val_format.append(relationship[rel_top5[0]])
  val_format.append(object1_object2[obj2_top5[0]])

  return dic_words, dic_idx, val_format

# test set prediction

test_videos = os.listdir(test_path)
test_videos = sorted([f for f in test_videos if f.startswith('000')])

df_pred = pd.DataFrame(columns = ['object1', 'relationship', 'object2'])
df_pred.index.names = ['ID']

count = 0

for video in test_videos:

  ordered = beam_search(video, 'test')
  words, idx, val_format = get_top5_preds(ordered)

  df_pred.loc[video] = idx

  count += 1
  print("{}/{} Video: {}".format(count, len(test_videos), video))
  print("{}\n".format(idx))

# convert to submission format
df_submission = pd.DataFrame(columns=['label'])
df_submission.index.names = ['ID']
id = 0
for i in df_pred.index:
  for j in df_pred.loc[str(i)]:
    df_submission.loc[id] = j
    id += 1
  
df_submission.to_csv(os.path.join(submission_path, 'submission.csv'))