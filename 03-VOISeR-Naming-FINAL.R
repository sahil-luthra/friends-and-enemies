rm(list=ls()) #clear all variables
library(ggplot2); library(plyr); library(dplyr); library(car); library(reshape); library(lme4); library(cowplot); library(stringi); library(scales); library(ggrepel)
load("~/friends-and-enemies/PostScript02.RData")
setwd("~/friends-and-enemies")

# Initial setup of VOISeR data ---- 
## VOISeR was trained on the ELP_groupdata.csv file, which was generated using the following code
# nmgData.correct.groupItem <- ddply(nmgData.correct, .(Word), summarise, latency=mean(latency))
# nmgData.correct.groupItem <- merge(lexicon, nmgData.correct.groupItem, by = 'Word')
# accuracy.groupItem <- ddply(nmgData, .(Word), summarise, accuracy=mean(accuracy))
# nmgData.correct.groupItem <- merge(nmgData.correct.groupItem, accuracy.groupItem, by = 'Word')
# groupData <- subset(nmgData.correct.groupItem, select = c("Word","Pron_NoStress","Length","Log_Freq_HAL","latency","accuracy"))
# colnames(groupData) <- c("Ortho","Phono","Length","LogFreq","meanRT","accuracy")
# write.csv(groupData, file = "ELP_groupData.csv")

# We then train VOISeR and read in the output data here
VOISeR <- read.csv('VOISeR-data-FrequencyWeighted-FINAL.csv', header = TRUE, sep = ',', na.strings = "#N/A")

#Select correct trials from epoch 10000
VOISeR$Epoch <- as.factor(VOISeR$Epoch)
VOISeR.correct <- droplevels(subset(VOISeR, Accuracy_Pronunciation == "TRUE"))
accuracy.VOISeR <- 100 * nrow(VOISeR.correct)/nrow(VOISeR)
accuracy.VOISeR



VOISeR.trained <- droplevels(subset(VOISeR, Epoch == "10000"))
VOISeR.trained.correct <- droplevels(subset(VOISeR.trained, Accuracy_Pronunciation == "TRUE"))
accuracy.VOISeR <- 100 * nrow(VOISeR.trained.correct)/nrow(VOISeR.trained)
accuracy.VOISeR

#Confirm that VOISeR was trained on words in degree proportional to logFreq
cor.test(VOISeR.trained.correct$Probability, VOISeR.trained.correct$Trained_Count)



#Compute the LN of subject RTs and of all model parameters
VOISeR.trained.correct.LN <- VOISeR.trained.correct %>%
  mutate(MeanRT = log(MeanRT)) %>%
  mutate(Cosine_Similarity = log(Cosine_Similarity)) %>%
  mutate(Mean_Squared_Error = log(Mean_Squared_Error)) %>%
  mutate(Euclidean_Distance = log(Euclidean_Distance)) %>%
  mutate(Cross_Entropy = log(Cross_Entropy)) %>%
  mutate(Hidden_Cosine_Similarity = log(Hidden_Cosine_Similarity)) %>%
  mutate(Hidden_Mean_Squared_Error = log(Hidden_Mean_Squared_Error)) %>%
  mutate(Hidden_Euclidean_Distance = log(Hidden_Euclidean_Distance)) %>%
  mutate(Hidden_Cross_Entropy = log(Hidden_Cross_Entropy))

#Create data frames for each word length
VOISeR.trained.correct.LN <- VOISeR.trained.correct.LN[which(VOISeR.trained.correct.LN$Ortho %in% lexicon$Word), ]
colnames(VOISeR.trained.correct.LN)[2]<-"Word"
VOISeR.trained.correct.LN <- merge(lexicon, VOISeR.trained.correct.LN[c("Epoch","Word","Phono","MeanRT","Trained_Count","Cosine_Similarity","Mean_Squared_Error","Euclidean_Distance","Cross_Entropy","Exported_Pronunciation","Accuracy_Pronunciation")], by = 'Word')


VOISeR.trained.correct.LN.3letter <- VOISeR.trained.correct.LN[which(VOISeR.trained.correct.LN$Word %in% lexicon.3letter$Word), ]
colnames(VOISeR.trained.correct.LN.3letter)[2]<-"Word"
VOISeR.trained.correct.LN.3letter <- merge(lexicon.3letter, VOISeR.trained.correct.LN.3letter[c("Epoch","Word","Phono","MeanRT","Trained_Count","Cosine_Similarity","Mean_Squared_Error","Euclidean_Distance","Cross_Entropy","Exported_Pronunciation","Accuracy_Pronunciation")], by = 'Word')

VOISeR.trained.correct.LN.4letter <- VOISeR.trained.correct.LN[which(VOISeR.trained.correct.LN$Word %in% lexicon.4letter$Word), ]
colnames(VOISeR.trained.correct.LN.4letter)[2]<-"Word"
VOISeR.trained.correct.LN.4letter <- merge(lexicon.4letter, VOISeR.trained.correct.LN.4letter[c("Epoch","Word","Phono","MeanRT","Trained_Count","Cosine_Similarity","Mean_Squared_Error","Euclidean_Distance","Cross_Entropy","Exported_Pronunciation","Accuracy_Pronunciation")], by = 'Word')

VOISeR.trained.correct.LN.5letter <- VOISeR.trained.correct.LN[which(VOISeR.trained.correct.LN$Word %in% lexicon.5letter$Word), ]
colnames(VOISeR.trained.correct.LN.5letter)[2]<-"Word"
VOISeR.trained.correct.LN.5letter <- merge(lexicon.5letter, VOISeR.trained.correct.LN.5letter[c("Epoch","Word","Phono","MeanRT","Trained_Count","Cosine_Similarity","Mean_Squared_Error","Euclidean_Distance","Cross_Entropy","Exported_Pronunciation","Accuracy_Pronunciation")], by = 'Word')

VOISeR.trained.correct.LN.6letter <- VOISeR.trained.correct.LN[which(VOISeR.trained.correct.LN$Word %in% lexicon.6letter$Word), ]
colnames(VOISeR.trained.correct.LN.6letter)[2]<-"Word"
VOISeR.trained.correct.LN.6letter <- merge(lexicon.6letter, VOISeR.trained.correct.LN.6letter[c("Epoch","Word","Phono","MeanRT","Trained_Count","Cosine_Similarity","Mean_Squared_Error","Euclidean_Distance","Cross_Entropy","Exported_Pronunciation","Accuracy_Pronunciation")], by = 'Word')

VOISeR.trained.correct.LN.7letter <- VOISeR.trained.correct.LN[which(VOISeR.trained.correct.LN$Word %in% lexicon.7letter$Word), ]
colnames(VOISeR.trained.correct.LN.7letter)[2]<-"Word"
VOISeR.trained.correct.LN.7letter <- merge(lexicon.7letter, VOISeR.trained.correct.LN.7letter[c("Epoch","Word","Phono","MeanRT","Trained_Count","Cosine_Similarity","Mean_Squared_Error","Euclidean_Distance","Cross_Entropy","Exported_Pronunciation","Accuracy_Pronunciation")], by = 'Word')

VOISeR.trained.correct.LN.8letter <- VOISeR.trained.correct.LN[which(VOISeR.trained.correct.LN$Word %in% lexicon.8letter$Word), ]
colnames(VOISeR.trained.correct.LN.8letter)[2]<-"Word"
VOISeR.trained.correct.LN.8letter <- merge(lexicon.8letter, VOISeR.trained.correct.LN.8letter[c("Epoch","Word","Phono","MeanRT","Trained_Count","Cosine_Similarity","Mean_Squared_Error","Euclidean_Distance","Cross_Entropy","Exported_Pronunciation","Accuracy_Pronunciation")], by = 'Word')

#Visualize correlations between model parameters and subject RT
my.pairscor(VOISeR.trained.correct.LN[c(6,8:11)])
cor.test(VOISeR.trained.correct.LN$MeanRT, VOISeR.trained.correct.LN$Cross_Entropy)
#Correlation with cross-entropy is highest

# Do a few other simple tests on VOISeR
# Factor out word length, word frequency, and neighborhood size before doing RT-CE correlation
pcor.test(VOISeR.trained.correct.LN$MeanRT, VOISeR.trained.correct.LN$Cross_Entropy, 
          cbind(VOISeR.trained.correct.LN$Length, VOISeR.trained.correct.LN$Log_Freq_HAL, 
                VOISeR.trained.correct.LN$Ortho_N))

# Test for word length effect
ggplot(VOISeR.trained.correct.LN, aes(Length, Cross_Entropy)) + 
  geom_smooth(method=lm, color = "black", alpha = 0.15) + geom_point() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = "transparent", color = NA)) + 
  labs(x="Word Length", y = "ln(CE)") 
cor.test(VOISeR.trained.correct.LN$Length, VOISeR.trained.correct.LN$Cross_Entropy)

# Test for word frequency effect
ggplot(VOISeR.trained.correct.LN, aes(Log_Freq_HAL, Cross_Entropy)) + 
  geom_smooth(method=lm, color = "black", alpha = 0.15) + geom_point() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = "transparent", color = NA)) + 
  labs(x="Word Frequency", y = "ln(CE)") 
cor.test(VOISeR.trained.correct.LN$Log_Freq_HAL, VOISeR.trained.correct.LN$Cross_Entropy)

# Test for orthographic neighbor effect
ggplot(VOISeR.trained.correct.LN, aes(Ortho_N, Cross_Entropy)) + 
  geom_smooth(method=lm, color = "black", alpha = 0.15) + geom_point() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = "transparent", color = NA)) + 
  labs(x="Word Frequency", y = "ln(CE)") 
cor.test(VOISeR.trained.correct.LN$Ortho_N, VOISeR.trained.correct.LN$Cross_Entropy)


# Cross entropy regressions with Enemies ####
#3-letter words
VOISeR.CE.3letter.Freq.Enemies <- lm(Cross_Entropy ~ Log_Freq_HAL + Enemies1 + Enemies2 + Enemies3, data = subset(VOISeR.trained.correct.LN.3letter, Ortho_N>0))
summary(VOISeR.CE.3letter.Freq.Enemies)[["coefficients"]]

#4-letter words
VOISeR.CE.4letter.Freq.Enemies <- lm(Cross_Entropy ~ Log_Freq_HAL + Enemies1 + Enemies2 + Enemies3 + Enemies4, data = subset(VOISeR.trained.correct.LN.4letter, Ortho_N>0))
summary(VOISeR.CE.4letter.Freq.Enemies)[["coefficients"]]

#5-letter words
VOISeR.CE.5letter.Freq.Enemies <- lm(Cross_Entropy ~ Log_Freq_HAL + Enemies1 + Enemies2 + Enemies3 + Enemies4 + Enemies5, data = subset(VOISeR.trained.correct.LN.5letter, Ortho_N>0))
summary(VOISeR.CE.5letter.Freq.Enemies)[["coefficients"]]

#6-letter words
VOISeR.CE.6letter.Freq.Enemies <- lm(Cross_Entropy ~ Log_Freq_HAL + Enemies1 + Enemies2 + Enemies3 + Enemies4 + Enemies5 + Enemies6, data = subset(VOISeR.trained.correct.LN.6letter, Ortho_N>0))
summary(VOISeR.CE.6letter.Freq.Enemies)[["coefficients"]]

#7-letter words
VOISeR.CE.7letter.Freq.Enemies <- lm(Cross_Entropy ~ Log_Freq_HAL + Enemies1 + Enemies2 + Enemies3 + Enemies4 + Enemies5 + Enemies6 + Enemies7, data = subset(VOISeR.trained.correct.LN.7letter, Ortho_N>0))
summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]]

#8-letter words
VOISeR.CE.8letter.Freq.Enemies <- lm(Cross_Entropy ~ Log_Freq_HAL + Enemies1 + Enemies2 + Enemies3 + Enemies4 + Enemies5 + Enemies6 + Enemies7 + Enemies8 , data = subset(VOISeR.trained.correct.LN.8letter, Ortho_N>0))
summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]]

#Unadjusted b estimates
VOISeR.Coef.Freq.Enemies.3letter <- cbind(
  as.numeric(VOISeR.CE.3letter.Freq.Enemies$coefficients[3]),
  as.numeric(VOISeR.CE.3letter.Freq.Enemies$coefficients[4]),
  as.numeric(VOISeR.CE.3letter.Freq.Enemies$coefficients[5])
)

VOISeR.Coef.Freq.Enemies.4letter <- cbind(
  as.numeric(VOISeR.CE.4letter.Freq.Enemies$coefficients[3]),
  as.numeric(VOISeR.CE.4letter.Freq.Enemies$coefficients[4]),
  as.numeric(VOISeR.CE.4letter.Freq.Enemies$coefficients[5]),
  as.numeric(VOISeR.CE.4letter.Freq.Enemies$coefficients[6])
)

VOISeR.Coef.Freq.Enemies.5letter <- cbind(
  as.numeric(VOISeR.CE.5letter.Freq.Enemies$coefficients[3]),
  as.numeric(VOISeR.CE.5letter.Freq.Enemies$coefficients[4]),
  as.numeric(VOISeR.CE.5letter.Freq.Enemies$coefficients[5]),
  as.numeric(VOISeR.CE.5letter.Freq.Enemies$coefficients[6]),
  as.numeric(VOISeR.CE.5letter.Freq.Enemies$coefficients[7])
)

VOISeR.Coef.Freq.Enemies.6letter <- cbind(
  as.numeric(VOISeR.CE.6letter.Freq.Enemies$coefficients[3]),
  as.numeric(VOISeR.CE.6letter.Freq.Enemies$coefficients[4]),
  as.numeric(VOISeR.CE.6letter.Freq.Enemies$coefficients[5]),
  as.numeric(VOISeR.CE.6letter.Freq.Enemies$coefficients[6]),
  as.numeric(VOISeR.CE.6letter.Freq.Enemies$coefficients[7]),
  as.numeric(VOISeR.CE.6letter.Freq.Enemies$coefficients[8])
)

VOISeR.Coef.Freq.Enemies.7letter <- cbind(
  as.numeric(VOISeR.CE.7letter.Freq.Enemies$coefficients[3]),
  as.numeric(VOISeR.CE.7letter.Freq.Enemies$coefficients[4]),
  as.numeric(VOISeR.CE.7letter.Freq.Enemies$coefficients[5]),
  as.numeric(VOISeR.CE.7letter.Freq.Enemies$coefficients[6]),
  as.numeric(VOISeR.CE.7letter.Freq.Enemies$coefficients[7]),
  as.numeric(VOISeR.CE.7letter.Freq.Enemies$coefficients[8]), 
  as.numeric(VOISeR.CE.7letter.Freq.Enemies$coefficients[9])
)

VOISeR.Coef.Freq.Enemies.8letter <- cbind(
  as.numeric(VOISeR.CE.8letter.Freq.Enemies$coefficients[3]),
  as.numeric(VOISeR.CE.8letter.Freq.Enemies$coefficients[4]),
  as.numeric(VOISeR.CE.8letter.Freq.Enemies$coefficients[5]),
  as.numeric(VOISeR.CE.8letter.Freq.Enemies$coefficients[6]),
  as.numeric(VOISeR.CE.8letter.Freq.Enemies$coefficients[7]),
  as.numeric(VOISeR.CE.8letter.Freq.Enemies$coefficients[8]),
  as.numeric(VOISeR.CE.8letter.Freq.Enemies$coefficients[9]),
  as.numeric(VOISeR.CE.8letter.Freq.Enemies$coefficients[10])
)

VOISeR.Coef.Freq.Enemies.allLetter <- as.vector(cbind(VOISeR.Coef.Freq.Enemies.3letter,VOISeR.Coef.Freq.Enemies.4letter,VOISeR.Coef.Freq.Enemies.5letter,VOISeR.Coef.Freq.Enemies.6letter,VOISeR.Coef.Freq.Enemies.7letter,VOISeR.Coef.Freq.Enemies.8letter))

metrics <- cbind(metrics, VOISeR.Coef.Freq.Enemies.allLetter)


#Adjusted b estimates
scalFac <- nrow(subset(VOISeR.trained.correct.LN.3letter, Ortho_N>0)) / (
  1/((summary(VOISeR.CE.3letter.Freq.Enemies)[["coefficients"]][3,2])^2) + 
    1/((summary(VOISeR.CE.3letter.Freq.Enemies)[["coefficients"]][4,2])^2) + 
    1/((summary(VOISeR.CE.3letter.Freq.Enemies)[["coefficients"]][5,2])^2)
)
VOISeR.CoefAdj.Freq.Enemies.3letter <- cbind(
  summary(VOISeR.CE.3letter.Freq.Enemies)[["coefficients"]][3,1] * scalFac/((summary(VOISeR.CE.3letter.Freq.Enemies)[["coefficients"]][3,2])^2),
  summary(VOISeR.CE.3letter.Freq.Enemies)[["coefficients"]][4,1] * scalFac/((summary(VOISeR.CE.3letter.Freq.Enemies)[["coefficients"]][4,2])^2), 
  summary(VOISeR.CE.3letter.Freq.Enemies)[["coefficients"]][5,1] * scalFac/((summary(VOISeR.CE.3letter.Freq.Enemies)[["coefficients"]][5,2])^2)
)

scalFac <- nrow(subset(VOISeR.trained.correct.LN.4letter, Ortho_N>0)) / (
  1/((summary(VOISeR.CE.4letter.Freq.Enemies)[["coefficients"]][3,2])^2) + 
    1/((summary(VOISeR.CE.4letter.Freq.Enemies)[["coefficients"]][4,2])^2) + 
    1/((summary(VOISeR.CE.4letter.Freq.Enemies)[["coefficients"]][5,2])^2) +
    1/((summary(VOISeR.CE.4letter.Freq.Enemies)[["coefficients"]][6,2])^2)
)
VOISeR.CoefAdj.Freq.Enemies.4letter <- cbind(
  summary(VOISeR.CE.4letter.Freq.Enemies)[["coefficients"]][3,1] * scalFac/((summary(VOISeR.CE.4letter.Freq.Enemies)[["coefficients"]][3,2])^2),
  summary(VOISeR.CE.4letter.Freq.Enemies)[["coefficients"]][4,1] * scalFac/((summary(VOISeR.CE.4letter.Freq.Enemies)[["coefficients"]][4,2])^2), 
  summary(VOISeR.CE.4letter.Freq.Enemies)[["coefficients"]][5,1] * scalFac/((summary(VOISeR.CE.4letter.Freq.Enemies)[["coefficients"]][5,2])^2),
  summary(VOISeR.CE.4letter.Freq.Enemies)[["coefficients"]][6,1] * scalFac/((summary(VOISeR.CE.4letter.Freq.Enemies)[["coefficients"]][6,2])^2)
)

scalFac <- nrow(subset(VOISeR.trained.correct.LN.5letter, Ortho_N>0)) / (
  1/((summary(VOISeR.CE.5letter.Freq.Enemies)[["coefficients"]][3,2])^2) + 
    1/((summary(VOISeR.CE.5letter.Freq.Enemies)[["coefficients"]][4,2])^2) + 
    1/((summary(VOISeR.CE.5letter.Freq.Enemies)[["coefficients"]][5,2])^2) +
    1/((summary(VOISeR.CE.5letter.Freq.Enemies)[["coefficients"]][6,2])^2) + 
    1/((summary(VOISeR.CE.5letter.Freq.Enemies)[["coefficients"]][7,2])^2) 
)
VOISeR.CoefAdj.Freq.Enemies.5letter <- cbind(
  summary(VOISeR.CE.5letter.Freq.Enemies)[["coefficients"]][3,1] * scalFac/((summary(VOISeR.CE.5letter.Freq.Enemies)[["coefficients"]][3,2])^2),
  summary(VOISeR.CE.5letter.Freq.Enemies)[["coefficients"]][4,1] * scalFac/((summary(VOISeR.CE.5letter.Freq.Enemies)[["coefficients"]][4,2])^2), 
  summary(VOISeR.CE.5letter.Freq.Enemies)[["coefficients"]][5,1] * scalFac/((summary(VOISeR.CE.5letter.Freq.Enemies)[["coefficients"]][5,2])^2),
  summary(VOISeR.CE.5letter.Freq.Enemies)[["coefficients"]][6,1] * scalFac/((summary(VOISeR.CE.5letter.Freq.Enemies)[["coefficients"]][6,2])^2),
  summary(VOISeR.CE.5letter.Freq.Enemies)[["coefficients"]][7,1] * scalFac/((summary(VOISeR.CE.5letter.Freq.Enemies)[["coefficients"]][7,2])^2)
)

scalFac <- nrow(subset(VOISeR.trained.correct.LN.6letter, Ortho_N>0)) / (
  1/((summary(VOISeR.CE.6letter.Freq.Enemies)[["coefficients"]][3,2])^2) + 
    1/((summary(VOISeR.CE.6letter.Freq.Enemies)[["coefficients"]][4,2])^2) + 
    1/((summary(VOISeR.CE.6letter.Freq.Enemies)[["coefficients"]][5,2])^2) +
    1/((summary(VOISeR.CE.6letter.Freq.Enemies)[["coefficients"]][6,2])^2) + 
    1/((summary(VOISeR.CE.6letter.Freq.Enemies)[["coefficients"]][7,2])^2) +
    1/((summary(VOISeR.CE.6letter.Freq.Enemies)[["coefficients"]][8,2])^2)
)
VOISeR.CoefAdj.Freq.Enemies.6letter <- cbind(
  summary(VOISeR.CE.6letter.Freq.Enemies)[["coefficients"]][3,1] * scalFac/((summary(VOISeR.CE.6letter.Freq.Enemies)[["coefficients"]][3,2])^2),
  summary(VOISeR.CE.6letter.Freq.Enemies)[["coefficients"]][4,1] * scalFac/((summary(VOISeR.CE.6letter.Freq.Enemies)[["coefficients"]][4,2])^2), 
  summary(VOISeR.CE.6letter.Freq.Enemies)[["coefficients"]][5,1] * scalFac/((summary(VOISeR.CE.6letter.Freq.Enemies)[["coefficients"]][5,2])^2),
  summary(VOISeR.CE.6letter.Freq.Enemies)[["coefficients"]][6,1] * scalFac/((summary(VOISeR.CE.6letter.Freq.Enemies)[["coefficients"]][6,2])^2),
  summary(VOISeR.CE.6letter.Freq.Enemies)[["coefficients"]][7,1] * scalFac/((summary(VOISeR.CE.6letter.Freq.Enemies)[["coefficients"]][7,2])^2),
  summary(VOISeR.CE.6letter.Freq.Enemies)[["coefficients"]][8,1] * scalFac/((summary(VOISeR.CE.6letter.Freq.Enemies)[["coefficients"]][8,2])^2)
)

scalFac <- nrow(subset(VOISeR.trained.correct.LN.7letter, Ortho_N>0)) / (
  1/((summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][3,2])^2) + 
    1/((summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][4,2])^2) + 
    1/((summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][5,2])^2) +
    1/((summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][6,2])^2) + 
    1/((summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][7,2])^2) +
    1/((summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][8,2])^2) + 
    1/((summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][9,2])^2)
)
VOISeR.CoefAdj.Freq.Enemies.7letter <- cbind(
  summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][3,1] * scalFac/((summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][3,2])^2),
  summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][4,1] * scalFac/((summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][4,2])^2), 
  summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][5,1] * scalFac/((summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][5,2])^2),
  summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][6,1] * scalFac/((summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][6,2])^2),
  summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][7,1] * scalFac/((summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][7,2])^2),
  summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][8,1] * scalFac/((summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][8,2])^2),
  summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][9,1] * scalFac/((summary(VOISeR.CE.7letter.Freq.Enemies)[["coefficients"]][9,2])^2)
)

scalFac <- nrow(subset(VOISeR.trained.correct.LN.8letter, Ortho_N>0)) / (
  1/((summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][3,2])^2) + 
    1/((summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][4,2])^2) + 
    1/((summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][5,2])^2) +
    1/((summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][6,2])^2) + 
    1/((summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][7,2])^2) +
    1/((summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][8,2])^2) + 
    1/((summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][9,2])^2) + 
    1/((summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][10,2])^2)
)
VOISeR.CoefAdj.Freq.Enemies.8letter <- cbind(
  summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][3,1] * scalFac/((summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][3,2])^2),
  summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][4,1] * scalFac/((summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][4,2])^2), 
  summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][5,1] * scalFac/((summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][5,2])^2),
  summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][6,1] * scalFac/((summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][6,2])^2),
  summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][7,1] * scalFac/((summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][7,2])^2),
  summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][8,1] * scalFac/((summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][8,2])^2),
  summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][9,1] * scalFac/((summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][9,2])^2),
  summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][10,1] * scalFac/((summary(VOISeR.CE.8letter.Freq.Enemies)[["coefficients"]][10,2])^2)
)

VOISeR.CoefAdj.Freq.Enemies.allLetter <- as.vector(cbind(VOISeR.CoefAdj.Freq.Enemies.3letter,VOISeR.CoefAdj.Freq.Enemies.4letter,VOISeR.CoefAdj.Freq.Enemies.5letter,VOISeR.CoefAdj.Freq.Enemies.6letter,VOISeR.CoefAdj.Freq.Enemies.7letter,VOISeR.CoefAdj.Freq.Enemies.8letter))

metrics <- cbind(metrics, VOISeR.CoefAdj.Freq.Enemies.allLetter)

#Plots 
ggplot(metrics, aes(entropy.allLetter, VOISeR.CoefAdj.Freq.Enemies.allLetter)) + 
  geom_smooth(method=lm, color = "black", alpha = 0.15) + geom_point(color = metrics$Colors) + 
  geom_text_repel(box.padding = 1, aes(label=Labels), color = metrics$Colors, 
                  size = 6, force = 3, max.iter = 5000) + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = "transparent", color = NA)) + 
  labs(x="Entropy", y = "Influence of Enemies \n on VOISeR Latency (b estimate)")

cor.test(entropy.allLetter,VOISeR.CoefAdj.Freq.Enemies.allLetter)
cor.test(entropy.allLetter[-c(1,4,8,13,19,26)], VOISeR.CoefAdj.Freq.Enemies.allLetter[-c(1,4,8,13,19,26)]) #look at this without first-position values


ggplot(metrics, aes(nmgCoefAdj.Freq.Enemies.allLetter, VOISeR.CoefAdj.Freq.Enemies.allLetter)) + 
  geom_smooth(method=lm, color = "black", alpha = 0.15) + geom_point(color = metrics$Colors) + 
  geom_text_repel(box.padding = 1, aes(label=Labels), color = metrics$Colors, 
                  size = 6, force = 3, max.iter = 5000) + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = "transparent", color = NA)) + 
  labs(x="b estimates from \n human subjects data", y = "b estimates from \n VOISeR model") 

cor.test(nmgCoefAdj.Freq.Enemies.allLetter,VOISeR.CoefAdj.Freq.Enemies.allLetter)
cor.test(nmgCoefAdj.Freq.Enemies.allLetter[-c(1,4,8,13,19,26)], VOISeR.CoefAdj.Freq.Enemies.allLetter[-c(1,4,8,13,19,26)]) #look at this without first-position values

save.image("~/friends-and-enemies/PostScript03.RData")  
