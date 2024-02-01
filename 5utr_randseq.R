source("functions_TA.R")
library(dplyr)
BiocManager::install("Biostrings")
BiocManager::install("BSgenome.Hsapiens.UCSC.hg38", force = TRUE)
BiocManager::install("biomaRt")
library(dplyr)
library(tidyverse)

#find median of the 5'UTR sequences 
median(nchar(seqs5pr$RNAseq))

#cut sequences to 150 nucleotides 
utrdf <- seqs5pr[nchar(seqs5pr$RNAseq) >= 150, ]
utrdf <- loop_truncseq(utrdf, "RNAseq", nrow(data), max_length = 150)

#code all utrs as 1 
utrdf$label <- 1

#load random sequences, adjusted for GC-content
ran_test <- read.csv("data/random_test.csv")
ran_train <- read.csv("/data/random_train.csv")

ran_test$id <- 0
ran_train$id <- 0

#create df for random elements vs 5utrs

randomdf <- selectRandomRows(ran_train, "seq", n = 9025)
randomdf <- loop_rna_seq(randomdf, "seq", nrow(mydata))

randomdf <- loop_truncseq(randomdf, "RNAseq", nrow(data), max_length = 150)

randomutr <- full_join(utrdf, randomdf)

write.csv(randomutr, "data/random-vs-utr.csv", row.names = FALSE)







