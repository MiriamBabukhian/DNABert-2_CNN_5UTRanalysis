#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: miriambabukhian
"""
from utils_DNABert2 import FivePrimeSeqs, train, validation, plot_ism_predictions, predict_wild_type, calculate_gc_percentage
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import torch 
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AdamW, get_linear_schedule_with_warmup, set_seed
import seaborn as sns
import scipy

data = pd.read_csv("traindata_cnn.csv") 

#exclude sequences for mutagenesis
brain_utrs_ids = ["ENST00000404241", "ENST00000463675", "ENST00000084798", "ENST00000409851", "ENST00000320717",
                  "ENST00000372677", "ENST00000355994", "ENST00000303177", "ENST00000425191", "ENST00000326695"]
mask = pd.notna(data['ensembl_transcript_id']) & data['ensembl_transcript_id'].str.contains('|'.join(brain_utrs_ids), na=False)
brainutrs_df = data[mask]
print(brainutrs_df)
excluded_data = data[~mask]
data = excluded_data

#create feats and labels arrays
Feats = np.array(data["CUTseq"])[:, np.newaxis]
ClassLabels = np.array(data["label"])[:, np.newaxis]
labels_ids = {'random': 0, '5UTR':1}

#exclude sequences for gc-pred correlation, the sequences are randomly picked
gc_seqs = Feats[300:340].flatten().tolist()
gc_labels = ClassLabels[300:340]
feats_excluded = np.concatenate((Feats[:300], Feats[340:]))
Feats = feats_excluded
labels_excluded = np.concatenate((ClassLabels[:300], ClassLabels[340:]))
ClassLabels = labels_excluded

# define parameters
set_seed(123)
batch_size = 32
epochs = 3

#select device 
#for MacBook M1 and M2 run: 
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') 
print('Using device:', device)  

#For CUDA devices: 
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print('Using device:', device)

# define model config
n_labels = len(labels_ids)
config = AutoConfig.from_pretrained("zhihan1996/DNABERT-2-117M", num_labels = n_labels, output_attentions = True)

#load tokenizer
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")

#define model
model = AutoModelForSequenceClassification.from_pretrained("zhihan1996/DNABERT-2-117M", config = config)
model.to(device)
print('Model loaded to `%s`'%device)

#test train split
X_train, X_test, y_train, y_test = train_test_split(Feats, ClassLabels,
                                                    test_size=0.2, random_state=42, stratify= ClassLabels)


# load train data,  create Pytorch dataset
train_dataset = FivePrimeSeqs(X_train.flatten().tolist(), tokenizer, y_train)

# move data into DataLoader (send training data in batches)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print('Created `train_dataloader` with %d batches!'%len(train_dataloader))


# validation step (same as training but with test data)

test_dataset = FivePrimeSeqs(X_test.flatten().tolist(), tokenizer, y_test)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print('Created `test_dataloader` with %d batches!'%len(test_dataloader))



# define optimizer
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5,
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                  )

# Total number of training steps is number of batches * number of epochs.
# `train_dataloader` contains batched data so `len(train_dataloader)` gives the number of batches.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value 
                                            num_training_steps = total_steps)


# send data to train and eval

#store loss and accuracy for plotting
all_loss = {'train_loss':[], 'val_loss':[]}
all_acc = {'train_acc':[], 'val_acc':[]}

# Loop through each epoch.
print('Epoch')
for epoch in tqdm(range(epochs)):
  print()
  print('Training on batches...')
  # Perform one full pass over the training set.
  train_labels, train_predict, train_loss = train(train_dataloader, model, optimizer, scheduler, device)
  train_acc = accuracy_score(train_labels, train_predict)

  # Get prediction form model on validation data.
  print('Validation on batches...')
  valid_labels, valid_predict, val_proba, val_loss, attention = validation(test_dataloader, model, device)
  val_acc = accuracy_score(valid_labels, valid_predict)

  # Print loss and accuracy values 
  print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
  print()

  all_loss['train_loss'].append(train_loss)
  all_loss['val_loss'].append(val_loss)
  all_acc['train_acc'].append(train_acc)
  all_acc['val_acc'].append(val_acc)
  

#plot accuracy curve over epochs
plt.plot(range(1, len(all_acc['train_acc']) + 1), all_acc['train_acc'])
plt.plot(range(1, len(all_acc['val_acc']) + 1), all_acc['val_acc'])
plt.ticklabel_format(style='plain', axis='x', useOffset=False)
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('DNABert-2 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#plot loss curve over epochs
plt.plot(range(1, len(all_loss['train_loss']) + 1), all_loss['train_loss'])
plt.plot(range(1, len(all_loss['val_loss']) + 1), all_loss['val_loss'])
plt.ticklabel_format(style='plain', axis='x', useOffset=False)
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('DNABert-2 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#store values for confusion matrix
true_labels, predictions_labels, prediction_proba, avg_epoch_loss, attention = validation(test_dataloader, model, device)

# Create the evaluation report.
evaluation_report = classification_report(true_labels, predictions_labels, labels=list(labels_ids.values()), target_names=list(labels_ids.keys()))
# Show the evaluation report.
print(evaluation_report)

#plot confusion matrix
cm = confusion_matrix(true_labels, predictions_labels, normalize='pred')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= {'random': 0, '5UTR': 1})
disp.plot(cmap='Blues')
disp.ax_.set_title('Normalized Confusion Matrix-DNABert-2')
disp.ax_.grid(False)
plt.show()
    
#select predicted probabilities for ROC-AUC
preds = prediction_proba[:, 1:2] 
fpr, tpr, thresholds = roc_curve(true_labels, preds)
roc_auc = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='DNABert (area = {:.3f})'.format(roc_auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')

#plt.savefig('cnn_roc.pdf', format='pdf')
plt.show()


#In Silico Mutagensis 
mut_seqs = np.array(brainutrs_df["CUTseq"])[:, np.newaxis]
mut_labels = np.array(brainutrs_df["label"])[:, np.newaxis]

wd_encodings, wd_preds, wd_att, mut_preds = plot_ism_predictions(tokenizer, mut_seqs.flatten().tolist(), mut_labels, model, device)

#plot Heatmap of the attention scores, visualization has to be done per sequence and per head.
seq = 2
attention_head = 3
s = sns.heatmap(wd_att[seq, attention_head, :25, :25], cmap = "viridis", annot=False)
s.set(title = f'Attention - Sequence {seq + 1}, Head {attention_head}')
plt.show()
    

#tokenizer vocabulary 
vocabulary = tokenizer.get_vocab()
target_token_id = 188
for i, (token, token_id) in enumerate(vocabulary.items()):
    if token_id == target_token_id:
        print(f"Found the target token '{token}' with ID {target_token_id} in the vocabulary.")
        break  # Stop the loop if the target token is found


#Calculate GC-content of previously selected sequences
gc_content = []
for seq in gc_seqs:
  gc_content.append(calculate_gc_percentage(seq))
  
gc_encs, gc_preds, gc_att = predict_wild_type(tokenizer, gc_seqs, gc_labels, model, device)

# Create a scatter plot
plt.scatter(gc_content, gc_preds, color='blue', alpha=0.5)

# Set axis labels and title
plt.xlabel('GC Content')
plt.ylabel('Predicted Probability')
plt.title('GC Content vs. Predicted Probability- DNABert-2')

# Show the plot
plt.show()

#calculate distance correlation
gc_arr = np.array(gc_content)
def calculate_dist_corr(gc_cont, predicted_proba):
  dist_corr = scipy.spatial.distance.correlation(gc_cont,predicted_proba)
  return dist_corr
calculate_dist_corr(gc_arr,gc_preds.flatten())



    
    
    
    
    

