source("utils_TA.R")

#import the libraries needed
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("rtracklayer")
BiocManager::install("Biostrings")
BiocManager::install("BSgenome.Hsapiens.UCSC.hg38", force = TRUE)
BiocManager::install("biomaRt")
library(dplyr)
library(tidyverse)
library(Rsamtools)

#import the datasets 
gtfdata <- rtracklayer::import("data/flair_filter_transcripts.gtf")
df=as.data.frame(gtfdata)
qfilt <- read.delim("data/quantification_flair_filter.counts.txt", header = T, sep = "\t")
transcriptToTranscriptID<-read.delim("data/TranscriptToTranscriptID", header = T, sep = "\t")
qfilt <- left_join(qfilt, transcriptToTranscriptID, by = "transcript")
ensemblgtf <- rtracklayer::import("data/Homo_sapiens.GRCh38.108.gtf")
ensembldf = as.data.frame(ensemblgtf)

#remove version number from our df 
df$transcript_id <- sub("\\.\\d+.*", "", df$transcript_id) 

#only keep ENSEMBL annotated transcripts 
df <- df %>% filter(type == "transcript")
df <- df[grep("ENST", df$transcript_id), ]
qfilt <- qfilt[grep("ENST", qfilt$transcript), ]

#left join ENSEMBL GTF and our dataframe to get info about the 5' UTR
#ensembldf$transcript_idv <- paste(ensembldf$transcript_id, ensembldf$transcript_version, sep = ".")
mergedf = left_join(df, ensembldf, by = "transcript_id")

# get DNA sequences 
seqs5pr = mergedf %>%
  filter(type.y == "five_prime_utr") %>%
  rename(ensembl_transcript_id = transcript_id) 
seqs5pr = get_5utr_sequence(seqs5pr, "ensembl_transcript_id")

# get RNA sequences 
seqs5pr = loop_rna_seq(seqs5pr, "5utr", nrow(mydata))

#cut rna seqs 
downsampled_df = cut_dna_sequence(downsampled_df, "RNAseq")
