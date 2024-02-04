#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: miriambabukhian
"""

from tqdm import tqdm
import torch 
from torch import nn 
import random
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class FivePrimeSeqs(Dataset):
  r"""PyTorch Dataset class for loading data.
    
  Arguments:
    inputs_ids (:obj:`str`):
        input sequences.
    
    use_tokenizer (:obj:`transformers.tokenization_?`):
        Transformer type tokenizer used to process raw sequences into numbers.
        
    labels_ids (:obj):
        store sequences labels 

   Returns:  
      Vocabulary of tokenized input sequence and respective labels
  """
  def __init__(self, input_ids, tokenizer, labels): 
      
      self.inputs = tokenizer.batch_encode_plus(input_ids,  add_special_tokens= True, 
                                                max_length=512, truncation = False, 
                                                padding = True, return_attention_mask=True, return_tensors = "pt")
   # Add labels
      self.n_examples = len(labels)
      self.inputs.update({'labels':torch.tensor(labels)})
      print('Finished!\n') 
   
      return 
   
  def __len__(self): 
      
      return self.n_examples
     
 
  def __getitem__(self, item): 
      
      return{key: self.inputs[key][item] for key in self.inputs.keys()} 
  
    
  
    
  
def train(dataloader, model, optimizer, scheduler, device): 
  r"""
  Train model on a single pass through the dataloader.

  Arguments:

      dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
          Parsed data into batches of tensors.

      model (:obj:'our model')

      optimizer_ (:obj:`transformers.optimization.AdamW`):
          Optimizer used for training.

      scheduler_ (:obj:`torch.optim.lr_scheduler.LambdaLR`):
          PyTorch scheduler.

      device_ (:obj:`torch.device`):
          Device used to load data before model training.

  Returns:

      :obj:`List of [True Labels, Predicted
        Labels, Average Loss].
  """

 # Tracking variables.
  predictions_labels = []
  true_labels = []
 # Total loss for this epoch.
  total_loss = 0

  model.train()

  for batch in tqdm(dataloader, total=len(dataloader)):

   true_labels += batch['labels'].flatten().tolist()
   
   # move batch to device
   batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
   
   model.zero_grad() 
 
   outputs = model(**batch)
   
   loss, logits = outputs[:2] 
   
   total_loss += loss.item()

   # Perform a backward pass to calculate the gradients.
   loss.backward() 

   # Clip the norm of the gradients to 1.0.
   # This is to help prevent the "exploding gradients" problem.
   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

   # Update parameters 
  
   optimizer.step() 

   # Update the learning rate.
  
   scheduler.step() 

   # Move logits and labels to CPU
   logits = logits.detach().cpu().numpy()
   
    # Convert these logits to list of predicted labels values.
   predictions_labels += logits.argmax(axis=-1).flatten().tolist()

  avg_epoch_loss = total_loss / len(dataloader)
  
  # Return all true labels and prediction for future evaluations.
  return true_labels, predictions_labels, avg_epoch_loss  
  


def validation(dataloader, model, device_):
  r"""Validation function to evaluate model performance on a 
  separate set of data.
  
  Arguments:
    dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
          Parsed data into batches of tensors.
    device_ (:obj:`torch.device`):
          Device used to load data before model training.
          
    model (:obj:'our model')
  Returns:
    
    :obj:`List of [True Labels, Predicted
        Labels, Predicted Probabilities, Average Loss, Attention Scores]
  """
  
  # Tracking variables
  predictions_labels = []
  true_labels = []
  probabilities = []
 #total loss for this epoch.
  total_loss = 0

  model.eval()

  for batch in tqdm(dataloader, total=len(dataloader)):

   true_labels += batch['labels'].flatten().tolist()

   # move batch to device
   batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

   with torch.no_grad():  
        
        outputs = model(**batch) 
        
        #print attention weights for last layer
        attentions = outputs.attentions[-1] 

        loss, logits = outputs[:2] 
        
        # Move logits, labels and attention to CPU
        logits = logits.detach().cpu().numpy() 
        attentions = attentions.detach().cpu().numpy()
 
        total_loss += loss.item() 
        
        #get prediction probabilities by passing the model's logits through a Softmax function
        probabilities += nn.functional.softmax(torch.tensor(logits), dim=-1) 
        prediction_proba = torch.stack(probabilities) 
        prediction_proba = prediction_proba.numpy() 

        predict_content = logits.argmax(axis=-1).flatten().tolist()

        predictions_labels += predict_content 

  # Calculate the average loss over the training data.
  avg_epoch_loss = total_loss / len(dataloader)
  
  # Return all true labels and predictiond for future evaluations.
  return true_labels, predictions_labels, prediction_proba, avg_epoch_loss, attentions 




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




def in_silico_mutagenesis(tokenizer, inputs, labels, model, device_):  #make sure X_test is a list (use .flatten().tolist() when working with a numpy array)
    r"""Pass mutated sequences through the model 
    in evaluation mode
    
    Arguments:
        use_tokenizer (:obj:`transformers.tokenization_?`):
            Transformer type tokenizer used to process raw sequences into numbers.
            
        inputs (:obj:`str`):
            input sequences.

        Labels: 
            List of labels
             
       model (:obj:'our model')
       device_ (:obj:`torch.device`):
             Device used to load data before model training.
    Returns:
      
      :obj: dictionary of mutation index as keys and predicted probabilities as values.
    """

    mutated_seqs = [] 
    for index, seq in enumerate(inputs):
        mutated_seqs.append(generate_mutated_sequences(seq))
    ordered_seqs = list(map(list, zip(*mutated_seqs))) 
    
    all_predictions = []

    for group_index, group in enumerate(ordered_seqs): 
        sequences = [seq for _, seq in group]
        
    #    print(f"Group {group_index + 1} sequences: {sequences}") #line for debugging
        
        encodings = FivePrimeSeqs(sequences, tokenizer, labels)
    
     #   print(f"Encoding for Group {group_index + 1}: {encodings}") #line for debugging
        
        #start validation 
    
        probabilities = []
        model.eval()
    
        batch = {k:v.type(torch.long).to(device_) for k,v in encodings.inputs.items()}
    
        with torch.no_grad():  
            outputs = model(**batch) 
            
            loss, logits = outputs[:2] 
              
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy() 
            
            #get prediction probabilities by passing the model's logits through a Softmax function
            probabilities += nn.functional.softmax(torch.tensor(logits), dim=-1) 
            
            #convert probabilities to a numpy array 
            prediction_proba = torch.stack(probabilities) 
            prediction_proba = prediction_proba.numpy()  
            prediction_prob = prediction_proba[:, 1:2] 
            
            all_predictions.append((group_index, prediction_prob))
            
    all_predictions_dict = {index: predictions for index, predictions in all_predictions}     
         
 
    return all_predictions_dict 




def predict_wild_type(tokenizer, inputs, labels, model, device_):
    
    r"""Get predictions for the wild type sequences
    
    Arguments:
        use_tokenizer (:obj:`transformers.tokenization_?`):
            Transformer type tokenizer used to process raw sequences into numbers.
            
        inputs (:obj:`str`):
            input sequences.
       Labels: 
            List of labels
       model (:obj:'our model')
       device_ (:obj:`torch.device`):
             Device used to load data before model training.
    Returns:
      
      :obj: 'List of [wild type data, predicted probabilities, attention scores]
    """
    
    wd_data = FivePrimeSeqs(inputs, tokenizer, labels)
    
    probabilities = []

    model.eval()
    
    batch = {k:v.type(torch.long).to(device_) for k,v in wd_data.inputs.items()}
    
    with torch.no_grad():
        outputs = model(**batch) 
        
        
        attention = outputs.attentions[-1] 
        loss, logits = outputs[:2] 
          
        logits = logits.detach().cpu().numpy() 
        attentions = attention.detach().cpu().numpy() 
        
        #get prediction probabilities by passing the model's logits through a Softmax function
        probabilities += nn.functional.softmax(torch.tensor(logits), dim=-1) 
        
        #convert probabilities to a numpy array 
        prediction_proba = torch.stack(probabilities) 
        prediction_proba = prediction_proba.numpy()  
        prediction_prob = prediction_proba[:, 1:2]
        
        
    return wd_data, prediction_prob, attentions 


     
def plot_ism_predictions(tokenizer, inputs, labels, model, device_):
    r"""Plots predicted probabilities for mutated sequences against 
    predicted probabilities for wild type.
   
   Arguments:
       use_tokenizer (:obj:`transformers.tokenization_?`):
           Transformer type tokenizer used to process raw sequences into numbers.
           
       inputs (:obj:`str`):
           input sequences.
   
      dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
            Parsed data into batches of tensors.
            
      model (:obj:'our model')
      device_ (:obj:`torch.device`):
            Device used to load data before model training.
   Returns:
     
     :obj: 'List of [wild type encodings, wild type data, 
     wild type attention scores, predicted probabilities for mutated sequences]
   """
   
    mutate_preds = in_silico_mutagenesis(tokenizer, inputs, labels, model, device_)
    
    wd_encodings, wd_preds, wd_attentions = predict_wild_type(tokenizer, inputs, labels, model, device_) 
 
    num_sequences = wd_preds.shape[0]

    for sequence_index in range(num_sequences):
        
        plt.figure(figsize=(8, 5))
    
    # Wild type sequence
        wild_type_probs = wd_preds[sequence_index]
        plt.scatter([0], wild_type_probs, color='red')
    
    # Mutated sequences
        mutated_indices = list(mutate_preds.keys())
        mutated_probs = [mutate_preds[idx][sequence_index] for idx in mutated_indices]
        plt.scatter(mutated_indices, mutated_probs, label='Mutated', color='black') 
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
        
    return wd_encodings, wd_preds, wd_attentions, mutate_preds  

    

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


