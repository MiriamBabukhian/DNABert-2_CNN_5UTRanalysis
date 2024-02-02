#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: miriambabukhian
"""

import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from keras.optimizers import Adadelta
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


#Create CNN model
def create_cnn_model(input_shape, nClasses):
    cnnModel = Sequential()
    cnnModel.add(Conv1D(filters=64, kernel_size=12, input_shape=input_shape, activation='relu'))
    cnnModel.add(Conv1D(filters=64, kernel_size=12, activation='relu', padding = 'same'))
    cnnModel.add(MaxPooling1D(pool_size=4))
    cnnModel.add(Dropout(0.25))
    cnnModel.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
    cnnModel.add(Conv1D(filters=32, kernel_size=12, activation='relu',padding = 'same' ))
    cnnModel.add(MaxPooling1D(pool_size=3))
    cnnModel.add(Dropout(0.25))
    cnnModel.add(Flatten())
    cnnModel.add(Dense(200, activation='relu'))
    cnnModel.add(Dropout(0.25))
    cnnModel.add(Dense(50, activation='relu'))
    cnnModel.add(Dropout(0.25))
    cnnModel.add(Dense(units=nClasses, activation='sigmoid'))

    cnnModel.compile(loss='binary_crossentropy', metrics=["accuracy"], optimizer=Adadelta())
    
    return cnnModel 


def seq2code(seq):
    r"""Performs one-hot-encoding of one sequence
    Arguments: 
        sequence(:obj:`str`):
            one input sequence.
    Returns: 
        one-hot-encoded sequence as a np array
    
    """
    map = np.asarray([[0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0.5, 0.5, 0, 0],
                      [0.5, 0, 0.5, 0],
                      [0.5, 0, 0, 0.5],
                      [0, 0.5, 0.5, 0],
                      [0, 0.5, 0, 0.5],
                      [0, 0, 0.5, 0.5]])

    seq = seq.upper().replace('A', '\x01').replace('C', '\x02')
    seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')
    seq = seq.replace('M', '\x05').replace('R', '\x06').replace('W', '\x07')
    seq = seq.replace('S', '\x08').replace('Y', '\x09').replace('K', '\x0A')

    return map[np.fromstring(seq, np.int8) % 12] 



def generate_mutated_sequences(sequence):
    r"""Generate mutated sequences for ISM
    
    Arguments:
        sequence (:obj:`str`):
            input sequence.
    Returns:
      
      :obj:`List of mutated sequences
    """
    sequence_alts = []
    for index, ref in enumerate(sequence):
        possible_nucleotides = ['A', 'C', 'G', 'T']
        
        #make sure that the function does NOT select the original nucleotide 
        possible_nucleotides.remove(ref) if ref in possible_nucleotides else None
        
        #randomly sample the list of possible nucleotides and mutate the original sequence
        subs = random.choice(possible_nucleotides)
        mutated_sequence = sequence[:index] + subs + sequence[index + 1:]
        sequence_alts.append((index, mutated_sequence))
    
    return sequence_alts 



def in_silico_mutagenesis(model, X_test):  #make sure X_test is a list (use .flatten().tolist() when working with a numpy array)
    r"""Pass mutated sequences through the model 
    in evaluation mode
    
    Arguments:
        X_test (:obj:`List of input sequences').
        model (:obj:'our model')
    Returns:
      
       :obj: dictionary of mutation index as keys and predicted probabilities as values.
    """
    mutated_seqs = [] 
    for index, seq in enumerate(X_test):
        mutated_seqs.append(generate_mutated_sequences(seq))
    ordered_seqs = list(map(list, zip(*mutated_seqs)))

    #perform one hot encoding of the mutated sequences and pass them through the model
    all_predictions = []
    for group_index, group in enumerate(ordered_seqs): 
        sequences = [seq for _, seq in group]
    
        encodings = []
        for seq in sequences:
            encodings.append(seq2code(seq))
        encodings = np.array(encodings)
        
        #check if the encoding worked
        all_zeros_rows = np.all(encodings == 0, axis=(1, 2))

        if np.any(all_zeros_rows):
            print("There are rows with all zeros.") 
        else:
            print("There are no rows with all zeros.") 
        
        #predict output
        predictions = model.predict(encodings)
        all_predictions.append((group_index, predictions))
     
    all_predictions_dict = {index: predictions for index, predictions in all_predictions} 
    
    return all_predictions_dict 




def plot_ism_predictions(model, X_test, wild_type_predictions):
    
    r""" Plots predicted probabilities for mutated sequences against 
  predicted probabilities for wild type.
  
  Arguments: 
      X_test (:obj:`List of input sequences').
      model (:obj:'our model')
      wild_type_predictions(:obj:'np array of predictions')

    Returns:
     :obj: np array of predictions 
    
    """
    mutate_preds = in_silico_mutagenesis(model, X_test)
    
    num_sequences = wild_type_predictions.shape[0]

    for sequence_index in range(num_sequences):
        
        plt.figure(figsize=(8, 5))
    
    # Wild type sequence
        wild_type_probs = wild_type_predictions[sequence_index]
        plt.scatter([0], wild_type_probs, color='red')
    
    # Mutated sequences
        mutated_indices = list(mutate_preds.keys())
        mutated_probs = [mutate_preds[idx][sequence_index] for idx in mutated_indices]
        plt.scatter(mutated_indices, mutated_probs, label='Mutated', color='black') 
        
        #insert axhline
        plt.axhline(y=wild_type_probs, color='red', linestyle='--', label='Wild Type Prediction')

    
    # Add labels and a title
        plt.xlabel('Mutation Index')
        plt.ylabel('Prediction Probability')
        plt.title(f'Sequence {sequence_index + 1} Predictions')
        plt.ylim(0, 1)
    
    # Show the legend
        plt.legend()
    
    # Show the plot
        plt.show()
        
    return mutate_preds 



def one_hot_to_sequence(one_hot_sequences, index_to_nucleotide):
    r""" decodes one hot encoding
  
  Arguments: 
      one_hot_sequences (:obj:`List of one-hot-encoded sequences').
      index_to_nucleotide(:obj:`Vocabulary of nucleotides as keys and positions as values')

    Returns:
     :obj: 'list of decoded sequences'
    
    """  
    sequences = []
    for one_hot_seq in one_hot_sequences:
        sequence = ''.join(index_to_nucleotide[idx] for idx in one_hot_seq.argmax(axis=1))
        sequences.append(sequence)
    return sequences 




def calculate_gc_percentage(sequence):
    r"""Calculates GC percentage of a sequence. 
    Arguments:
        sequence (:obj:`str`):
            single input sequence.
    Returns: 
        GC Percentage of the sequence.
    """
    
    gc_count = 0

    for base in sequence:
        if base.upper() in ['G', 'C']:
            gc_count += 1

    total_bases = len(sequence)
    gc_percentage = (gc_count / total_bases) * 100

    return gc_percentage 


