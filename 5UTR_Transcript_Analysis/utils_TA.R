BiocManager::install("BSgenome.Hsapiens.UCSC.hg38", force = TRUE)
BiocManager::install("biomaRt")
library(Biostrings)

# get DNA sequences 
mart <- useMart('ensembl',
                dataset = 'hsapiens_gene_ensembl',
                host = 'https://useast.ensembl.org')

get_5utr_sequence <- function(data, column_name) {
  cds_seq <- getSequence(id=data[[column_name]], 
                         type="ensembl_transcript_id", 
                         seqType="5utr", 
                         mart=mart)
  return(cds_seq)
}

#get RNA sequences 
get_rna_sequence <- function(mydata, column_name, row) { 
  dna_seq <- DNAString(mydata[[row, column_name]])
  rna_seq <- RNAString(dna_seq)
  return(rna_seq)
}

#loop through getRNA function

loop_rna_seq <- function(mydata, column_name, row) {
  for (i in (1:nrow(mydata))) {
    print (i)
    mydata[i, "RNAseq"] <- toString(get_rna_sequence(mydata, column_name, i))
  }
  return (mydata)
}


# Function to cut DNA sequence into batches of n nucleotides
cut_dna_sequence <- function(data, column_name, row, batch_size) {
  num_batches <- nchar(data[[row, column_name]]) %/% batch_size
  cut_sequence <- substr(data[[row, column_name]], start = 1, stop = num_batches*batch_size)
  return(cut_sequence)
}

#loop through cutDNA function
loop_cutseq <- function(data, column_name, row, batch_size) {
  for (i in (1:nrow(data))) {
    data[i, "CUTseq"] <- cut_dna_sequence(data, column_name, i, batch_size)
  }
  return(data)
}

   
# functions to cut DNA sequence into pieces of 30 and maintain other columns

split_string_in_chunks <- function(string, chunk_size = 30) {
  string <- strsplit(string, "")[[1]]
  string <- unlist(lapply(split(string, ceiling(seq_along(string)/chunk_size)), FUN = function(x) paste(x, collapse="")))
  string
} 

splitter <- function(data, seq_column) {
  splits <- lapply(data[[seq_column]], split_string_in_chunks)
  data_expanded <- data[rep(1:nrow(data), times = lengths(splits)),]
  data_expanded$sequence_split <- unlist(splits)
  rownames(data_expanded) <- NULL
  data_expanded
} 


#select random elements of any df

selectRandomRows <- function(data, column_name, condition, n) {
  # Check if n is greater than the number of rows in the DataFrame
  if (n > nrow(data)) {
    stop("n is greater than the number of rows in the DataFrame.")
  }
  
  #filter dataframe based on condition 
  filtered_data <- data[data[[column_name]] == condition, ]
  random_indices <- sample(nrow(filtered_data), n)
  selected_rows <- data[random_indices, ]
  
  return(selected_rows)
} 

# get coding sequences 

get_cds <- function(data, column_name) {
  cds_seq <- getSequence(id=data[[column_name]], 
                         type="ensembl_transcript_id", 
                         seqType="coding", 
                         mart=mart)
  return(cds_seq)
}

#function to truncate the sequences at the chosen length of n nucleotides

truncate_sequences <- function(data, column_name, row, max_length) {  
  # Truncate the sequence at the maximum length
  truncated_sequence <- substr(data[[row, column_name]], start = 1, stop = max_length)
  
  return(truncated_sequence)
}

loop_truncseq <- function(data, column_name, row, max_length) {
  for (i in (1:nrow(data))) {
    data[i, "CUTseq"] <- truncate_sequences(data, column_name, i, max_length)
  }
  return(data)
}




