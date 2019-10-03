#install.packages("GGally")

library(GGally)
library(readr)
library(ggplot2)

work_Dir <- "D:/Work&Study/Sahil - Letter position entropy/Feedback_RNN/Simple_RNN/HU_500.LR_0005.EMB_20/Result/"
check_Epoch = 1000
result <- read_delim(paste(work_Dir, "Result.txt", sep=""), "\t", escape_double = FALSE, locale = locale(encoding = "UTF-8"), trim_ws = TRUE)
result <- subset(result, (result$Epoch == check_Epoch & result$Accuracy_Pronunciation == "True"))[,c("MeanRT", "Cosine_Similarity", "Mean_Squared_Error", "Euclidean_Distance", "Cross_Entropy", "Hidden_Cosine_Similarity", "Hidden_Mean_Squared_Error", "Hidden_Euclidean_Distance", "Hidden_Cross_Entropy")]
colnames(result) <- c("MeanRT", "Phonology_CS", "Phonology_MSE", "Phonology_ED", "Phonology_CE", "Hidden_CS", "Hidden_MSE", "Hidden_ED", "Hidden_CE")

ggpairs(result,
        title = "Original"
        )
ggsave(paste(work_Dir, "Result_Scatter_Original.png", sep=""), width = 16, height = 16)

ggpairs(log(result),
        title = "LN"
)
ggsave(paste(work_Dir, "Result_Scatter_LN.png", sep=""), width = 16, height = 16)

ggpairs(sqrt(result),
        title = "SQRT"
)
ggsave(paste(work_Dir, "Result_Scatter_SQRT.png", sep=""), width = 16, height = 16)

ggpairs(sign(result) * abs(result)^(1/3),
        title = "CubeRoot"
)
ggsave(paste(work_Dir, "Result_Scatter_CubeRoot.png", sep=""), width = 16, height = 16)
