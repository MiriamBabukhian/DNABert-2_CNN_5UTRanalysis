# DNABert-2_CNN_5UTRanalysis

Contains the code used in the Major Research Project: "Unveiling Cryptic Regulatory Elements in 5’UTRs with DNABert-2: A Comparative Analysis with CNN Models" at Utrecht University.

##Project Overview
The code retrieves RNA/DNA sequences, performs differential expression analysis for tissue-specific transcripts and passes the obtaines sequences through two models: DNABert-2 and a CNN. To interpret the model's decisions, in silico mutagensis is performed for both models, as well as Attention Scores Visualization for DNABert-2. 

##Code Structure 

The repository is divided over 3 folders: 

##5UTR_Transcript_Analysis: 
contains the data preprocessing steps, including retrieval of DNA/RNA sequences, Differential Expression analysis and PCA
##DNABert2_training: 
contains main script and functions of DNABert-2, including in silico mutagensis and interpretation
##CNN_training: 
contains main script and functions of the CNN,including in silico mutagensis

##Reusability: 
The code was written with reusability in mind. Most functions can potentailly be reused for other purposes. 
