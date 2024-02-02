#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: miriambabukhian
"""

from functions_cnn import create_cnn_model, seq2code, plot_ism_predictions, one_hot_to_sequence, calculate_gc_percentage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow import keras
import scipy


data = pd.read_csv("traindata_cnn.csv")

#exclude sequences for mutagenesis
brain_utrs_ids =  ["ENST00000404241", "ENST00000463675", "ENST00000084798", "ENST00000409851", "ENST00000320717",
                  "ENST00000372677", "ENST00000355994", "ENST00000303177", "ENST00000425191", "ENST00000326695"]
mask = pd.notna(data['ensembl_transcript_id']) & data['ensembl_transcript_id'].str.contains('|'.join(brain_utrs_ids), na=False)
brainutrs_df = data[mask]
print(brainutrs_df)
excluded_data = data[~mask]
data = excluded_data

Feats = np.array(data["CUTseq"])[:, np.newaxis]
ClassLabels = np.array(data["label"])[:, np.newaxis]

#exclude utr sequences for gc-pred correlation, the utr sequences are randomly picked
gc_seqs = Feats[300:340].flatten().tolist()
gc_labels = ClassLabels[300:340]
feats_excluded = np.concatenate((Feats[:300], Feats[340:]))
Feats = feats_excluded
labels_excluded = np.concatenate((ClassLabels[:300], ClassLabels[340:]))
ClassLabels = labels_excluded


# encode data with onehot encoding
encodings = []
for seq in range(len(Feats)):
    encodings.append(seq2code(Feats[seq][0]))

encodings= np.array(encodings)

# Check if any rows have all zeros (check if one hot encoding worked)
all_zeros_rows = np.all(encodings == 0, axis=(1, 2))

if np.any(all_zeros_rows):
    print("There are rows with all zeros.")
else:
    print("There are no rows with all zeros.")
    
    
# testtrain split
X_train, X_test, y_train, y_test = train_test_split(encodings, ClassLabels,
                                                    test_size=0.2, random_state=42, stratify= ClassLabels)

# run cnn model on train data
cnn_model = create_cnn_model(input_shape = (150, 4), nClasses = 1)
cnn_fit = cnn_model.fit(X_train, y_train.ravel(), validation_data=(X_test, y_test), epochs = 100)


# predict on X test
cnn_pred = cnn_model.predict(X_test)

#plot accuracy and loss curves
training_loss = cnn_fit.history['loss']
validation_loss = cnn_fit.history['val_loss']
training_accuracy = cnn_fit.history['accuracy']
validation_accuracy = cnn_fit.history['val_accuracy']

#accuracy curve over epochs
plt.plot(cnn_fit.history['accuracy'])
plt.plot(cnn_fit.history['val_accuracy'])
plt.title('CNN accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#loss curve over epochs
plt.plot(cnn_fit.history['loss'])
plt.plot(cnn_fit.history['val_loss'])
plt.title('CNN loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# ROC & AUC
fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, cnn_pred)
auc_nn = auc(fpr_nn, tpr_nn)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_nn, tpr_nn, label='Neural network (area = {:.3f})'.format(auc_nn))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')

plt.show()

#confusion matrix
y_pred = np.where(cnn_pred > 0.5, 1, 0)
print(y_pred[:5])

cm = confusion_matrix(y_test, y_pred, normalize='pred')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= {'random': 0, 'utr': 1})
disp.plot()
plt.show()

#In silico mutagenesis
mut_seqs = np.array(brainutrs_df["CUTseq"])[:, np.newaxis]
mut_labels = np.array(brainutrs_df["label"])[:, np.newaxis]

#encode wildtype sequences
wildtype_encodings = []
for seq in range(len(mut_seqs)):
    wildtype_encodings.append(seq2code(mut_seqs[seq][0]))

#predict on wild type 
wd_encodings= np.array(wildtype_encodings)
wildtype_preds = cnn_model.predict(wd_encodings)

# predict on mutated seqs
preds_mutate = plot_ism_predictions(cnn_model, mut_seqs.flatten().tolist(), wildtype_preds)

#to revert one hot encoding for interpretation
index_to_nucleotide = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
reverted_sequences = one_hot_to_sequence(mut_seqs, index_to_nucleotide)

#calculate GC content on previously selected sequences
gc_content = []
for seq in gc_seqs:
  gc_content.append(calculate_gc_percentage(seq))
  
gc_encodings = []
for seq in gc_seqs:
    gc_encodings.append(seq2code(seq))

gc_encodings= np.array(gc_encodings)

gc_preds = cnn_model.predict(gc_encodings)

# Create a scatter plot
plt.scatter(gc_content, gc_preds, color='blue', alpha=0.5)

# Set axis labels and title
plt.xlabel('GC Content')
plt.ylabel('Predicted Probability')
plt.title('GC Content vs. Predicted Probability-CNN')

# Show the plot
plt.show()

#calculate distance correlation
gc_arr = np.array(gc_content)
def calculate_dist_corr(gc_cont, predicted_proba):
  dist_corr = scipy.spatial.distance.correlation(gc_cont,predicted_proba)
  return dist_corr
calculate_dist_corr(gc_arr,gc_preds.flatten())










