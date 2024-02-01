library(ggplot2)
library(RColorBrewer)

#Principal Component Analysis
samples_all = read.csv2("/Users/miriambabukhian/Downloads/data_pca.csv")
samples_all = samples_all[,-c(2,3)]
samples_all$origin <- gsub("\\s+", "", samples_all$origin)

#prepare datasets (using tpm counts)
rownames(qfilt) <- qfilt$transcript
qfilt$transcript <- NULL
qfiltemp <- match(samples_all$sample, colnames(qfilt))
qfilt <- qfilt[, qfiltemp, drop = FALSE]

#exclude transcripts expressed below 5 tpm 
qfilt <- qfilt[ rowSums(qfilt >= 5) >= 3, ]

#prepare pca dataframe
genes.pca <- prcomp(t(qfilt), center = TRUE, scale. = TRUE)
percentVar <- round(100 * apply(genes.pca$x, 2, var)/sum(apply(genes.pca$x, 2, var)),1)
pcs <- as.data.frame(genes.pca$x[, c(1:6)])
pcs$sample_id <- rownames(pcs)
rownames(pcs) <- NULL
head(pcs)

#plot the pca 
pcs <- merge(pcs, samples_all, by = "sample_id")
ggplot(pcs, aes(PC1, PC2)) +
  geom_point(aes(fill=origin),colour="grey20",pch=21, size=5) +
  theme_classic(base_size =14) +
  xlab(paste0("PC1: ",percentVar[1],"% variance")) +
  ylab(paste0("PC2: ",percentVar[2],"% variance")) + 
  guides(fill = guide_legend(override.aes = list(size=6))) +
  scale_fill_manual(values = c("#ffc0cb","#8d5b96","#7776b1","#9773ba","#b873ba","#c893c9",
                               "#ff69b4","#d4a910","#c4625d","#bc3c28","#815375",
                               "#0072b5", "#1f854e","#e18726"))

