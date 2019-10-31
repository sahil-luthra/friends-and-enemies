rm(list=ls()) #clear all variables
library(ggplot2); library(plyr); library(dplyr); library(car); library(reshape); library(cowplot); library(lme4); library(stringi); library(scales); library(ggrepel)
load("~/friends-and-enemies/PostScript01.RData")  
setwd("~/friends-and-enemies")

# Initial setup of naming data -------------------------
nmgData <- read.csv('nmg_data.csv', header = TRUE, sep = ',', na.strings = "#N/A",
                    colClasses = c("NULL","factor","factor",NA,"factor",NA,"factor","factor"))
nmgData$Word <- nmgData$item

# Limit analysis to correct trials  
nmgData$accuracy[nmgData$pronun==1] <- 1    
nmgData$accuracy[nmgData$pronun!=1] <- 0    
nmgData$accuracy[nmgData$rspnsDur<0] <- 0
nmgData.correct <- droplevels(subset(nmgData, pronun == 1 & rspnsDur>0))
nmgData.correct$logLatency <- log(nmgData.correct$latency)
nmgData.correct <- merge(lexicon[1:2], nmgData.correct, by = 'Word')
accuracy.nmg <- 100 * nrow(nmgData.correct)/nrow(nmgData)
accuracy.nmg
nmgData.correct <- droplevels(subset(nmgData.correct, Length >=3 & Length <= 8))

# Create data subsets for each word length of interest 
nmgData.correct.3letter <- droplevels(subset(nmgData.correct, Length == 3))
nmgData.correct.4letter <- droplevels(subset(nmgData.correct, Length == 4))
nmgData.correct.5letter <- droplevels(subset(nmgData.correct, Length == 5))
nmgData.correct.6letter <- droplevels(subset(nmgData.correct, Length == 6))
nmgData.correct.7letter <- droplevels(subset(nmgData.correct, Length == 7))
nmgData.correct.8letter <- droplevels(subset(nmgData.correct, Length == 8))

# Then merge performance data with lexical metrics
nmgData.correct.3letter <- merge(lexicon.3letter, nmgData.correct.3letter[c("Word","subject","latency","accuracy")], by = 'Word')
nmgData.correct.4letter <- merge(lexicon.4letter, nmgData.correct.4letter[c("Word","subject","latency","accuracy")], by = 'Word')
nmgData.correct.5letter <- merge(lexicon.5letter, nmgData.correct.5letter[c("Word","subject","latency","accuracy")], by = 'Word')
nmgData.correct.6letter <- merge(lexicon.6letter, nmgData.correct.6letter[c("Word","subject","latency","accuracy")], by = 'Word')
nmgData.correct.7letter <- merge(lexicon.7letter, nmgData.correct.7letter[c("Word","subject","latency","accuracy")], by = 'Word')
nmgData.correct.8letter <- merge(lexicon.8letter, nmgData.correct.8letter[c("Word","subject","latency","accuracy")], by = 'Word')

# Regression analyses for Enemies at each Position -----

#3-letter words  
nmgModel.RTs.3letter.Freq.Enemies <- lmer(log(latency) ~ Log_Freq_HAL + Enemies1 + Enemies2 + Enemies3 + (1|subject) + (1|Word), data = subset(nmgData.correct.3letter, Ortho_N>0))
summary(nmgModel.RTs.3letter.Freq.Enemies)

#4-letter words  
nmgModel.RTs.4letter.Freq.Enemies <- lmer(log(latency) ~ Log_Freq_HAL + Enemies1 + Enemies2 + Enemies3 + Enemies4 + (1|subject) + (1|Word), data = subset(nmgData.correct.4letter, Ortho_N>0))
summary(nmgModel.RTs.4letter.Freq.Enemies)

# 5-letter words
nmgModel.RTs.5letter.Freq.Enemies <- lmer(log(latency) ~ Log_Freq_HAL + Enemies1 + Enemies2 + Enemies3 + Enemies4 + Enemies5 + (1|subject) + (1|Word), data = subset(nmgData.correct.5letter, Ortho_N>0))
summary(nmgModel.RTs.5letter.Freq.Enemies)

# 6-letter words
nmgModel.RTs.6letter.Freq.Enemies <- lmer(log(latency) ~ Log_Freq_HAL + Enemies1 + Enemies2 + Enemies3 + Enemies4 + Enemies5 + Enemies6 + (1|subject) + (1|Word), data = subset(nmgData.correct.6letter, Ortho_N>0))
summary(nmgModel.RTs.6letter.Freq.Enemies)

# 7-letter words
nmgModel.RTs.7letter.Freq.Enemies <- lmer(log(latency) ~ Log_Freq_HAL + Enemies1 + Enemies2 + Enemies3 + Enemies4 + Enemies5 + Enemies6 + Enemies7 + (1|subject) + (1|Word), data = subset(nmgData.correct.7letter, Ortho_N>0))
summary(nmgModel.RTs.7letter.Freq.Enemies)

#8-letter words
nmgModel.RTs.8letter.Freq.Enemies <- lmer(log(latency) ~ Log_Freq_HAL + Enemies1 + Enemies2 + Enemies3 + Enemies4 + Enemies5 + Enemies6 + Enemies7 + Enemies8 + (1|subject) + (1|Word), data = subset(nmgData.correct.8letter, Ortho_N>0))
summary(nmgModel.RTs.8letter.Freq.Enemies)

# Unadjusted b weights
nmgCoef.Freq.Enemies.3letter <- cbind(
  summary(nmgModel.RTs.3letter.Freq.Enemies)[["coefficients"]][3,1],
  summary(nmgModel.RTs.3letter.Freq.Enemies)[["coefficients"]][4,1], 
  summary(nmgModel.RTs.3letter.Freq.Enemies)[["coefficients"]][5,1]
)

nmgCoef.Freq.Enemies.4letter <- cbind(
  summary(nmgModel.RTs.4letter.Freq.Enemies)[["coefficients"]][3,1],
  summary(nmgModel.RTs.4letter.Freq.Enemies)[["coefficients"]][4,1], 
  summary(nmgModel.RTs.4letter.Freq.Enemies)[["coefficients"]][5,1],
  summary(nmgModel.RTs.4letter.Freq.Enemies)[["coefficients"]][6,1]
)

nmgCoef.Freq.Enemies.5letter <- cbind(
  summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][3,1],
  summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][4,1],
  summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][5,1],
  summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][6,1],
  summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][7,1]
)

nmgCoef.Freq.Enemies.6letter <- cbind(
  summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][3,1],
  summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][4,1],
  summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][5,1],
  summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][6,1],
  summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][7,1],
  summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][8,1]
)

nmgCoef.Freq.Enemies.7letter <- cbind(
  summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][3,1],
  summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][4,1],
  summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][5,1],
  summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][6,1],
  summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][7,1],
  summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][8,1],
  summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][9,1]
)

nmgCoef.Freq.Enemies.8letter <- cbind(
  summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][3,1],
  summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][4,1],
  summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][5,1],
  summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][6,1],
  summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][7,1],
  summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][8,1],
  summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][9,1],
  summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][10,1]
)

nmgCoef.Freq.Enemies.allLetter <- as.vector(cbind(nmgCoef.Freq.Enemies.3letter,nmgCoef.Freq.Enemies.4letter,nmgCoef.Freq.Enemies.5letter,nmgCoef.Freq.Enemies.6letter,nmgCoef.Freq.Enemies.7letter, nmgCoef.Freq.Enemies.8letter))
metrics <- cbind.data.frame(metrics, nmgCoef.Freq.Enemies.allLetter)

#Adjusted b estimates
scalFac <- nrow(subset(nmgData.correct.3letter, Ortho_N>0)) / (
  1/((summary(nmgModel.RTs.3letter.Freq.Enemies)[["coefficients"]][3,2])^2) + 
    1/((summary(nmgModel.RTs.3letter.Freq.Enemies)[["coefficients"]][4,2])^2) + 
    1/((summary(nmgModel.RTs.3letter.Freq.Enemies)[["coefficients"]][5,2])^2)
)
nmgCoefAdj.Freq.Enemies.3letter <- cbind(
  summary(nmgModel.RTs.3letter.Freq.Enemies)[["coefficients"]][3,1] * scalFac/((summary(nmgModel.RTs.3letter.Freq.Enemies)[["coefficients"]][3,2])^2),
  summary(nmgModel.RTs.3letter.Freq.Enemies)[["coefficients"]][4,1] * scalFac/((summary(nmgModel.RTs.3letter.Freq.Enemies)[["coefficients"]][4,2])^2), 
  summary(nmgModel.RTs.3letter.Freq.Enemies)[["coefficients"]][5,1] * scalFac/((summary(nmgModel.RTs.3letter.Freq.Enemies)[["coefficients"]][5,2])^2)
)

scalFac <- nrow(subset(nmgData.correct.4letter, Ortho_N>0)) / (
  1/((summary(nmgModel.RTs.4letter.Freq.Enemies)[["coefficients"]][3,2])^2) + 
    1/((summary(nmgModel.RTs.4letter.Freq.Enemies)[["coefficients"]][4,2])^2) + 
    1/((summary(nmgModel.RTs.4letter.Freq.Enemies)[["coefficients"]][5,2])^2) +
    1/((summary(nmgModel.RTs.4letter.Freq.Enemies)[["coefficients"]][6,2])^2)
)
nmgCoefAdj.Freq.Enemies.4letter <- cbind(
  summary(nmgModel.RTs.4letter.Freq.Enemies)[["coefficients"]][3,1] * scalFac/((summary(nmgModel.RTs.4letter.Freq.Enemies)[["coefficients"]][3,2])^2),
  summary(nmgModel.RTs.4letter.Freq.Enemies)[["coefficients"]][4,1] * scalFac/((summary(nmgModel.RTs.4letter.Freq.Enemies)[["coefficients"]][4,2])^2), 
  summary(nmgModel.RTs.4letter.Freq.Enemies)[["coefficients"]][5,1] * scalFac/((summary(nmgModel.RTs.4letter.Freq.Enemies)[["coefficients"]][5,2])^2),
  summary(nmgModel.RTs.4letter.Freq.Enemies)[["coefficients"]][6,1] * scalFac/((summary(nmgModel.RTs.4letter.Freq.Enemies)[["coefficients"]][6,2])^2)
)

scalFac <- nrow(subset(nmgData.correct.5letter, Ortho_N>0)) / (
  1/((summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][3,2])^2) + 
    1/((summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][4,2])^2) + 
    1/((summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][5,2])^2) +
    1/((summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][6,2])^2) + 
    1/((summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][7,2])^2) 
)
nmgCoefAdj.Freq.Enemies.5letter <- cbind(
  summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][3,1] * scalFac/((summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][3,2])^2),
  summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][4,1] * scalFac/((summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][4,2])^2), 
  summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][5,1] * scalFac/((summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][5,2])^2),
  summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][6,1] * scalFac/((summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][6,2])^2),
  summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][7,1] * scalFac/((summary(nmgModel.RTs.5letter.Freq.Enemies)[["coefficients"]][7,2])^2)
)

scalFac <- nrow(subset(nmgData.correct.6letter, Ortho_N>0)) / (
  1/((summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][3,2])^2) + 
    1/((summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][4,2])^2) + 
    1/((summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][5,2])^2) +
    1/((summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][6,2])^2) + 
    1/((summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][7,2])^2) +
    1/((summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][8,2])^2)
)
nmgCoefAdj.Freq.Enemies.6letter <- cbind(
  summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][3,1] * scalFac/((summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][3,2])^2),
  summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][4,1] * scalFac/((summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][4,2])^2), 
  summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][5,1] * scalFac/((summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][5,2])^2),
  summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][6,1] * scalFac/((summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][6,2])^2),
  summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][7,1] * scalFac/((summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][7,2])^2),
  summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][8,1] * scalFac/((summary(nmgModel.RTs.6letter.Freq.Enemies)[["coefficients"]][8,2])^2)
)

scalFac <- nrow(subset(nmgData.correct.7letter, Ortho_N>0)) / (
  1/((summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][3,2])^2) + 
    1/((summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][4,2])^2) + 
    1/((summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][5,2])^2) +
    1/((summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][6,2])^2) + 
    1/((summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][7,2])^2) +
    1/((summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][8,2])^2) + 
    1/((summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][9,2])^2)
)
nmgCoefAdj.Freq.Enemies.7letter <- cbind(
  summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][3,1] * scalFac/((summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][3,2])^2),
  summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][4,1] * scalFac/((summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][4,2])^2), 
  summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][5,1] * scalFac/((summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][5,2])^2),
  summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][6,1] * scalFac/((summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][6,2])^2),
  summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][7,1] * scalFac/((summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][7,2])^2),
  summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][8,1] * scalFac/((summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][8,2])^2),
  summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][9,1] * scalFac/((summary(nmgModel.RTs.7letter.Freq.Enemies)[["coefficients"]][9,2])^2)
)

scalFac <- nrow(subset(nmgData.correct.8letter, Ortho_N>0)) / (
  1/((summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][3,2])^2) + 
    1/((summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][4,2])^2) + 
    1/((summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][5,2])^2) +
    1/((summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][6,2])^2) + 
    1/((summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][7,2])^2) +
    1/((summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][8,2])^2) + 
    1/((summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][9,2])^2) + 
    1/((summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][10,2])^2)
)
nmgCoefAdj.Freq.Enemies.8letter <- cbind(
  summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][3,1] * scalFac/((summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][3,2])^2),
  summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][4,1] * scalFac/((summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][4,2])^2), 
  summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][5,1] * scalFac/((summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][5,2])^2),
  summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][6,1] * scalFac/((summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][6,2])^2),
  summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][7,1] * scalFac/((summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][7,2])^2),
  summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][8,1] * scalFac/((summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][8,2])^2),
  summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][9,1] * scalFac/((summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][9,2])^2),
  summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][10,1] * scalFac/((summary(nmgModel.RTs.8letter.Freq.Enemies)[["coefficients"]][10,2])^2)
)

nmgCoefAdj.Freq.Enemies.allLetter <- as.vector(cbind(nmgCoefAdj.Freq.Enemies.3letter,nmgCoefAdj.Freq.Enemies.4letter,nmgCoefAdj.Freq.Enemies.5letter,nmgCoefAdj.Freq.Enemies.6letter,nmgCoefAdj.Freq.Enemies.7letter,nmgCoefAdj.Freq.Enemies.8letter))

metrics <- cbind.data.frame(metrics, nmgCoefAdj.Freq.Enemies.allLetter)


#Compare adjusted b estimates to entropy
theme_set(theme_bw(base_size=20))

ggplot(metrics, aes(entropy.allLetter, nmgCoefAdj.Freq.Enemies.allLetter)) + 
  geom_smooth(method=lm, color = "black", alpha = 0.15) + geom_point(color = metrics$Colors) + 
  geom_text_repel(box.padding = 1, aes(label=Labels), color = metrics$Colors, 
                   size = 6, force = 3, max.iter = 5000) + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = "transparent", color = NA)) + 
  labs(x="Entropy", y = "Influence of Enemies \n on Naming Latency (b estimate)") 
cor.test(entropy.allLetter, nmgCoefAdj.Freq.Enemies.allLetter)

#we look at this without any first-position values

ggplot(metrics[-c(1,4,8,13,19,26), ], aes(entropy.allLetter, nmgCoefAdj.Freq.Enemies.allLetter)) + 
  geom_smooth(method=lm, color = "black", alpha = 0.15) + geom_point(color = metrics[-c(1,4,8,13,19,26), ]$Colors) + 
  geom_text_repel(box.padding = 1, aes(label=Labels), color = metrics[-c(1,4,8,13,19,26), ]$Colors, 
                  size = 6, force = 3, max.iter = 5000) + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = "transparent", color = NA)) + 
  labs(x="Entropy", y = "Influence of Enemies \n on Naming Latency (b estimate)") 
cor.test(entropy.allLetter[-c(1,4,8,13,19,26)], nmgCoefAdj.Freq.Enemies.allLetter[-c(1,4,8,13,19,26)])

save.image("~/friends-and-enemies/PostScript02.RData")  
