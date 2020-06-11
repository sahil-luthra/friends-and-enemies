# Setup VOISER data ----
rm(list=ls()) #clear all variables
library(ggplot2); library(plyr); library(dplyr); library(car); library(reshape); library(lme4); library(cowplot); library(stringi); library(scales); library(ggrepel); library(Rmisc); library(afex)
setwd("~/friends-and-enemies/VOISeR-Consistency-Regularity/")
theme_set(theme_classic(base_size=20))

# Load in correct file
VOISeR <- read.csv('E_10K.Summary.csv', header = TRUE, sep = ',', na.strings = "#N/A")

#Accuracy
VOISeR$Epoch <- as.factor(VOISeR$Epoch)
VOISeR.correct <- droplevels(subset(VOISeR, Accuracy_Pronunciation == "TRUE"))
accuracy.VOISeR <- 100 * nrow(VOISeR.correct)/nrow(VOISeR)
accuracy.VOISeR

# Log transform metrics
VOISeR.correct.LN <- VOISeR.correct %>%
  mutate(MeanRT = log(MeanRT)) %>%
  mutate(Cosine_Similarity = log(Cosine_Similarity)) %>%
  mutate(Mean_Squared_Error = log(Mean_Squared_Error)) %>%
  mutate(Euclidean_Distance = log(Euclidean_Distance)) %>%
  mutate(Cross_Entropy = log(Cross_Entropy)) %>%
  mutate(Hidden_Cosine_Similarity = log(Hidden_Cosine_Similarity)) %>%
  mutate(Hidden_Mean_Squared_Error = log(Hidden_Mean_Squared_Error)) %>%
  mutate(Hidden_Euclidean_Distance = log(Hidden_Euclidean_Distance)) %>%
  mutate(Hidden_Cross_Entropy = log(Hidden_Cross_Entropy))

# Check how well lnCE approximates human RT
cor.test(VOISeR.correct.LN$MeanRT, VOISeR.correct.LN$Cross_Entropy)

# Test for Consistency Effects in Word Recognition (Taraban & McClelland, 1987) ----
# Set up data frame
words <- read.csv('VOISeR.Words-Taraban-McClelland.10K-epoch.csv', header = TRUE, sep = ',', na.strings = "#N/A")
words$lnCE <- log(words$Cross_Entropy)

# Note if word is consistent, inconsistent or exception
wordStim <- read.csv('stimuli.Words-Taraban-McClelland.csv', header = TRUE, sep = ',', na.strings = "#N/A")
words <- merge(words, wordStim, by = 'Ortho')
words$Accuracy <- words$Accuracy_Pronunciation

words$type = factor(words$type, levels(words$type)[c(1,3,2)])
levels(words$type)

words.Correct <- droplevels(subset(words, Accuracy == TRUE))

# Summarize accuracy results
word.ACCbyType <- summarySE(words, measurevar = "Accuracy", groupvars = c("type"))
word.ACCbyType

# Plot accuracy results
ggplot(word.ACCbyType, aes(type, Accuracy)) + geom_bar(position = position_dodge(), stat = "identity", 
                                                       fill = "#B4B4B4", width = 0.5) + 
  geom_errorbar(aes(ymax=Accuracy+se, ymin=Accuracy-se), width = 0.2) + 
  coord_cartesian(ylim = c(0,1)) + labs(title = "VOISeR Accuracy (10K Epochs)", subtitle = "Stimuli from Taraban and McClelland (1987)") +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5))

# Model accuracy results
model.word.Acc <- mixed(Accuracy ~ type + (1|Ortho),
                         data=words,family="binomial",method="LRT",expand_re = TRUE,
                         control = glmerControl(optimizer="bobyqa",calc.derivs = FALSE, optCtrl = list(maxfun = 1500000)))
model.word.Acc #omnibus motivates pairwise comparisons

words.ce <- droplevels(subset(words, type != "inconsistent"))
model.word.Acc.ce <- mixed(Accuracy ~ type + (1|Ortho),
                        data=words.ce,family="binomial",method="LRT",expand_re = TRUE,
                        control = glmerControl(optimizer="bobyqa",calc.derivs = FALSE, optCtrl = list(maxfun = 1500000)))
model.word.Acc.ce

words.ie <- droplevels(subset(words, type != "consistent"))
model.word.Acc.ie <- mixed(Accuracy ~ type + (1|Ortho),
                           data=words.ie,family="binomial",method="LRT",expand_re = TRUE,
                           control = glmerControl(optimizer="bobyqa",calc.derivs = FALSE, optCtrl = list(maxfun = 1500000)))
model.word.Acc.ie

# Summarize RT results
word.RTbyType <- summarySE(words.Correct, measurevar = "lnCE", groupvars = c("type"))
word.RTbyType

# Plot RT results
ggplot(word.RTbyType, aes(type, 15+lnCE)) + geom_bar(position = position_dodge(), stat = "identity", 
                                                     fill = "#B4B4B4", width = 0.5) + 
  geom_errorbar(aes(ymax=15+lnCE+se, ymin=15+lnCE-se), width = 0.2, position = position_dodge(0.5)) + 
  labs(title = "VOISeR Latencies (10K Epochs)", subtitle = "Stimuli from Taraban and McClelland (1987)") +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5))

# Model RT results
model.words.rt <- lm(lnCE ~ type,
                              data=words.Correct)
summary(model.words.rt) #gives us two of three pairwise comparisons

# recode to see third comparison
words.Correct$type = factor(words.Correct$type, levels(words.Correct$type)[c(2,1,3)])
model.words.rt <- lm(lnCE ~ type,
                     data=words.Correct)
summary(model.words.rt)

# Consistency Effects in Nonword Recognition (Glushko, 1979) ----
# Set up data frame
nonwords <- read.csv('VOISeR.Nonwords-Glushko.10K-epoch.csv', header = TRUE, sep = ',', na.strings = "#N/A")
nonwords$Accuracy <- nonwords$Accuracy_Pronunciation

# Determine RTs for correct trials
nonwords.correct <- droplevels(subset(nonwords, Accuracy == 1))
nonwords.correct$lnCE <- log(nonwords.correct$Cross_Entropy)

# Determine accuracy
nonwords <- subset(nonwords, select = c("Ortho", "Accuracy"))
nonwords$Accuracy <- sapply(nonwords$Ortho, function(x) 
   sum(nonwords$Accuracy[nonwords$Ortho == x])) 
nonwords <- distinct(nonwords)

# Note if nonword is consistent or inconsistent
nonword.stim <- read.csv('nonword_pronunciations_ELPcode.csv', header = FALSE, sep = ',', na.strings = "#N/A")
nonword.stim <- distinct(subset(nonword.stim, select = -c(V2)))
colnames(nonword.stim) <- c("Ortho", "type")
nonwords <- merge(nonwords, nonword.stim, by = 'Ortho')
nonwords.correct <- merge(nonwords.correct, nonword.stim, by = 'Ortho')

# Summarize accuracy results
nonword.ACCbyType <- summarySE(nonwords, measurevar = "Accuracy", groupvars = c("type"))
nonword.ACCbyType

# Plot accuracy results
ggplot(nonword.ACCbyType, aes(type, Accuracy)) + geom_bar(position = position_dodge(), stat = "identity", 
                                                       fill = "#B4B4B4", width = 0.5) + 
  geom_errorbar(aes(ymax=Accuracy+se, ymin=Accuracy-se), width = 0.2) + 
  coord_cartesian(ylim = c(0,1)) + labs(title = "VOISeR Accuracy (10K Epochs)", subtitle = "Stimuli from Glushko (1979)") +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5))

# Model accuracy results
model.nonword.Acc <- mixed(Accuracy ~ type + (1|Ortho),
                        data=nonwords,family="binomial",method="LRT",expand_re = TRUE,
                        control = glmerControl(optimizer="bobyqa",calc.derivs = FALSE, optCtrl = list(maxfun = 1500000)))
model.nonword.Acc 

# Summarize RT results
nonword.RTbyType <- summarySE(nonwords.correct, measurevar = "lnCE", groupvars = c("type"))
nonword.RTbyType

# Plot RT results
ggplot(nonword.RTbyType, aes(type, 15 + lnCE)) + 
  geom_bar(position = position_dodge(), stat = "identity",  fill = "#B4B4B4", width = 0.5) + 
  geom_errorbar(aes(ymax=15 + lnCE+se, ymin=15 + lnCE-se), width = 0.2, position = position_dodge(0.5)) + 
  labs(title = "VOISeR Latencies (10K Epochs)", subtitle = "Stimuli from Glushko (1979)") +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5))

# Model RT results
model.nonwords.rt <- lm(lnCE ~ type,
                     data=nonwords.correct)
summary(model.nonwords.rt) 

save.image("~/friends-and-enemies/VOISeR-Consistency-Regularity/VOISeR-Consistency-Regularity.RData")  

