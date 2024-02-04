source("utils_TA.R")
library(dplyr)
library(Seurat)

#import metadata 
samples <- read.csv2("data/sample_type2.csv")

#check if there are duplicates and remove them
samples <- samples %>% rename(sample_id = transcript)
samples <- samples[!duplicated(samples$sample_id), ]
qfilt <- qfilt[, !names(qfilt) %in% c("GTEX.WY7C.0008.SM.3NZB5_ctrl.1")]

#turn elements of the first column into row names 
rownames(samples) <- samples[,1]
samples[,1] <- NULL
rownames(qfilt) <- qfilt[,1]
qfilt[,1] <- NULL

#match sample information to quantification data 
temp_qfilt <- match(rownames(samples), colnames(qfilt))
qfilt <- qfilt[ ,temp_qfilt, drop = FALSE]

#create Seurat Object
seurat_obj <- CreateSeuratObject(counts = qfilt, meta.data = samples)
seurat_obj <- NormalizeData(seurat_obj)
seurat_obj <- ScaleData(seurat_obj) 

#set identity 
seurat_obj <- SetIdent(seurat_obj, value = 'origin')

#Run FindMarkers 
de_res <- FindMarkers(
  object = seurat_obj,
  ident.1 = 'Brain',
  ident.2 = 'Other',
  test.use = 'LR',
  min.pct = 0.05,
  logfc.threshold = 0
)

#select p values lower than 0.05
sigpvalue <- de_res[de_res[, 5] < 0.05, ]

#select transcripts with a log2fold change >1 or < -0.3
siglogfold <- sigpvalue[sigpvalue$avg_log2FC > 1 | sigpvalue$avg_log2FC < -0.3, ]

#make sure the transcript ids are in the same format as the sequence file & exclude rows that are present in qfilt but not df 
siglogfold$ensembl_transcript_id <- row.names(siglogfold)
siglogfold$ensembl_transcript_id <- sub("\\.\\d+-.*", "", siglogfold$ensembl_transcript_id)
siglogfold <- siglogfold[siglogfold$ensembl_transcript_id %in% seqs5pr$ensembl_transcript_id, ]

#code the label 
siglogfold$transcript_label <- ifelse(siglogfold$avg_log2FC > 1, '1', 
                                      ifelse(siglogfold$avg_log2FC < -0.3, '0', NA))

#make sure the transcript ids are in the same format as the sequence file 
siglogfold$ensembl_transcript_id <- row.names(siglogfold)
siglogfold$ensembl_transcript_id <- sub("\\.\\d+-.*", "", siglogfold$ensembl_transcript_id)


#exclude rows that are present in the quantification file but not in the dataframe 
siglogfold <- siglogfold[siglogfold$ensembl_transcript_id %in% seqs5pr$ensembl_transcript_id, ]

labelled_transcript <- left_join(siglogfold, seqs5pr, by = "ensembl_transcript_id")

# plot barplot to show brain-specific transcripts vs transcripts from other tissues
counts <- table(siglogfold$transcript_label)
barplot(counts, names.arg = c("Brain", "Others"), col = c("red", "blue"), 
                main = "Tissue-Specific Transcripts", ylab = "Count", xlab = "Tissue", 
                ylim = c(0, max(counts) + 50))

# downsample 

# Find the class with the larger number of observations
class1_count <- sum(labelled_transcript$transcript_label == 1)
class2_count <- sum(labelled_transcript$transcript_label == 0)

if (class1_count > class2_count) {
  majority_class <- 1
  minority_class <- 0
  minority_count <- class2_count
} else {
  majority_class <- 0
  minority_class <- 1
  minority_count <- class1_count
}

# Randomly sample rows from the majority class to match the minority class count
majority_indices <- which(labelled_transcript$transcript_label == majority_class)
downsampled_indices <- sample(majority_indices, minority_count)

# Create the downsampled dataset
downsampled_df <- labelled_transcript[c(downsampled_indices, which(labelled_transcript$transcript_label == minority_class)), ]
downsampled_df <- downsampled_df[nchar(downsampled_df$RNAseq) >= 30, ]

# Apply the function to a whole dataframe
downsampled_df = loop_cutseq(downsampled_df, "RNAseq", nrow(data), batch_size = 30)
newdata <- splitter(downsampled_df, "CUTseq")
newdata <- newdata[, c("transcript_label", "sequence_split")]

write.csv(newdata, "data//traindata_DNABert-DNA.csv", row.names=FALSE)


# do a very ugly but effective thing, keep brain transcripts to 30 ncls long - ONLY run if sequences have to be of different lengths 
df_brain <- newdata %>% filter(transcript_label == 1)

# non brain transcripts have to be 40 nucleotides long 
df_others <- downsampled_df %>% filter(transcript_label == 0)
df_others <- loop_cutseq(df_others, "RNAseq", nrow(data), batch_size = 40)
df_others <- remove_empty(df_others, which = c("rows", "cols"), cutoff = 1, quiet = TRUE)
df_others <-  splitter(df_others, "CUTseq")
df_others <- df_others[, c("transcript_label", "sequence_split")]

#downsample brain df, randomly remove last ~900 rows 
df_brain <- df_brain[-c(1100:1908), ]
final_df <- full_join(df_brain, df_others)

write.csv(final_df, "data//validcnn_data.csv", row.names=FALSE)



