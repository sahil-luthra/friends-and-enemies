rm(list=ls()) #clear all variables
library(ggplot2); library(plyr); library(dplyr); library(car); library(reshape); library(lme4); library(cowplot); library(stringi); library(scales); library(ggrepel)
load("~/friends-and-enemies/PostScript03.RData")
setwd("~/friends-and-enemies")

# Initial setup of data file ----
lexDecData <- read.csv('ld_data.csv', sep = ',', na.strings = "#N/A")
colnames(lexDecData) <- c("filename","Trial","ItemID","Type","Accuracy","RT","Word")
lexDecData$Subject <- gsub("DATA", lexDecData$filename, ignore.case = TRUE, replacement = "")
lexDecData$Subject <- gsub(".LDT", lexDecData$Subject, ignore.case = TRUE, replacement = "")
lexDecData <- droplevels(subset(lexDecData, Type == 1 | Type == 0))

lexDecData.correct <- droplevels(subset(lexDecData, Accuracy == 1))
lexDecData.correct$RT <- as.numeric(as.character(lexDecData.correct$RT))
lexDecData.correct <- subset(lexDecData.correct, RT > 0)
lexDecData.correct.word <- droplevels(subset(lexDecData.correct, Type == 1))
lexDecData.correct.word <- lexDecData.correct.word[c(8,2:7)]

lexDecData.correct.word$Subject <- as.factor(lexDecData.correct.word$Subject)
lexDecData.correct.word$Trial <- as.factor(lexDecData.correct.word$Trial)
lexDecData.correct.word$ItemID <- as.factor(lexDecData.correct.word$ItemID)
lexDecData.correct.word$Type <- as.factor(lexDecData.correct.word$Type)
lexDecData.correct.word$Accuracy <- as.factor(lexDecData.correct.word$Accuracy)
lexDecData.correct.word$RT <- as.numeric(as.character(lexDecData.correct.word$RT))
lexDecData.correct.word$Word <- as.factor(lexDecData.correct.word$Word)

ggplot(lexDecData.correct.word, aes(RT)) + geom_density()

lexDecData.correct.word$logRT <- log(lexDecData.correct.word$RT)

lexDecData.correct.word <- lexDecData.correct.word[which(lexDecData.correct.word$Word %in% lexicon$Word), ]

lexDecData.correct.word.3to8letter <- merge(lexicon, lexDecData.correct.word, by = 'Word')
lexDecData.correct.word.3to8letter <- droplevels(subset(lexDecData.correct.word.3to8letter, Length >= 3))
lexDecData.correct.word.3to8letter <- droplevels(subset(lexDecData.correct.word.3to8letter, Length <= 8))

lexDecData.correct.word.3letter <- lexDecData.correct.word[which(lexDecData.correct.word$Word %in% lexicon.3letter$Word), ]
lexDecData.correct.word.3letter <- merge(lexicon.3letter, lexDecData.correct.word.3letter, by = 'Word')

lexDecData.correct.word.4letter <- lexDecData.correct.word[which(lexDecData.correct.word$Word %in% lexicon.4letter$Word), ]
lexDecData.correct.word.4letter <- merge(lexicon.4letter, lexDecData.correct.word.4letter, by = 'Word')

lexDecData.correct.word.5letter <- lexDecData.correct.word[which(lexDecData.correct.word$Word %in% lexicon.5letter$Word), ]
lexDecData.correct.word.5letter <- merge(lexicon.5letter, lexDecData.correct.word.5letter, by = 'Word')

lexDecData.correct.word.6letter <- lexDecData.correct.word[which(lexDecData.correct.word$Word %in% lexicon.6letter$Word), ]
lexDecData.correct.word.6letter <- merge(lexicon.6letter, lexDecData.correct.word.6letter, by = 'Word')

lexDecData.correct.word.7letter <- lexDecData.correct.word[which(lexDecData.correct.word$Word %in% lexicon.7letter$Word), ]
lexDecData.correct.word.7letter <- merge(lexicon.7letter, lexDecData.correct.word.7letter, by = 'Word')

lexDecData.correct.word.8letter <- lexDecData.correct.word[which(lexDecData.correct.word$Word %in% lexicon.8letter$Word), ]
lexDecData.correct.word.8letter <- merge(lexicon.8letter, lexDecData.correct.word.8letter, by = 'Word')

# Baseline Models
# lexDecModel.RTs.3letter.Freq.OrthoN <- lmer(log(RT) ~ Log_Freq_HAL + Ortho_N + (1|Subject) + (1|Word), data = subset(lexDecData.correct.word.3letter, Ortho_N>0))
# lexDecModel.RTs.4letter.Freq.OrthoN <- lmer(log(RT) ~ Log_Freq_HAL + Ortho_N + (1|Subject) + (1|Word), data = subset(lexDecData.correct.word.4letter, Ortho_N>0))
# lexDecModel.RTs.5letter.Freq.OrthoN <- lmer(log(RT) ~ Log_Freq_HAL + Ortho_N + (1|Subject) + (1|Word), data = subset(lexDecData.correct.word.5letter, Ortho_N>0))
# lexDecModel.RTs.6letter.Freq.OrthoN <- lmer(log(RT) ~ Log_Freq_HAL + Ortho_N + (1|Subject) + (1|Word), data = subset(lexDecData.correct.word.6letter, Ortho_N>0))
# lexDecModel.RTs.7letter.Freq.OrthoN <- lmer(log(RT) ~ Log_Freq_HAL + Ortho_N + (1|Subject) + (1|Word), data = subset(lexDecData.correct.word.7letter, Ortho_N>0))
# lexDecModel.RTs.8letter.Freq.OrthoN <- lmer(log(RT) ~ Log_Freq_HAL + Ortho_N + (1|Subject) + (1|Word), data = subset(lexDecData.correct.word.8letter, Ortho_N>0))


# Regression analyses for Number of Enemies at each Position -----
#3-letter words  
lexDecModel.RTs.3letter.Freq.Enemies <- lmer(log(RT) ~ Log_Freq_HAL + Enemies1 + Enemies2 + Enemies3 + (1|Subject) + (1|Word), data =  subset(lexDecData.correct.word.3letter, Ortho_N>0))
# anova(lexDecModel.RTs.3letter.Freq.OrthoN, lexDecModel.RTs.3letter.Freq.Enemies)
summary(lexDecModel.RTs.3letter.Freq.Enemies)

#4-letter words  
lexDecModel.RTs.4letter.Freq.Enemies <- lmer(log(RT) ~ Log_Freq_HAL + Enemies1 + Enemies2 + Enemies3 + Enemies4 + (1|Subject) + (1|Word), data =  subset(lexDecData.correct.word.4letter, Ortho_N>0))
# anova(lexDecModel.RTs.4letter.Freq.OrthoN, lexDecModel.RTs.4letter.Freq.Enemies)
summary(lexDecModel.RTs.4letter.Freq.Enemies)

# 5-letter words
lexDecModel.RTs.5letter.Freq.Enemies <- lmer(log(RT) ~ Log_Freq_HAL + Enemies1 + Enemies2 + Enemies3 + Enemies4 + Enemies5 + (1|Subject) + (1|Word), data =  subset(lexDecData.correct.word.5letter, Ortho_N>0))
# anova(lexDecModel.RTs.5letter.Freq.OrthoN, lexDecModel.RTs.5letter.Freq.Enemies)
summary(lexDecModel.RTs.5letter.Freq.Enemies)

# 6-letter words
lexDecModel.RTs.6letter.Freq.Enemies <- lmer(log(RT) ~ Log_Freq_HAL + Enemies1 + Enemies2 + Enemies3 + Enemies4 + Enemies5 + Enemies6 + (1|Subject) + (1|Word), data =  subset(lexDecData.correct.word.6letter, Ortho_N>0))
# anova(lexDecModel.RTs.6letter.Freq.OrthoN, lexDecModel.RTs.6letter.Freq.Enemies)
summary(lexDecModel.RTs.6letter.Freq.Enemies)

# 7-letter words
lexDecModel.RTs.7letter.Freq.Enemies <- lmer(log(RT) ~ Log_Freq_HAL + Enemies1 + Enemies2 + Enemies3 + Enemies4 + Enemies5 + Enemies6 + Enemies7 + (1|Subject) + (1|Word), data =  subset(lexDecData.correct.word.7letter, Ortho_N>0))
# anova(lexDecModel.RTs.7letter.Freq.OrthoN, lexDecModel.RTs.7letter.Freq.Enemies)
summary(lexDecModel.RTs.7letter.Freq.Enemies)

# 8-letter words
lexDecModel.RTs.8letter.Freq.Enemies <- lmer(log(RT) ~ Log_Freq_HAL + Enemies1 + Enemies2 + Enemies3 + Enemies4 + Enemies5 + Enemies6 + Enemies7 + Enemies8 + (1|Subject) + (1|Word), data =  subset(lexDecData.correct.word.8letter, Ortho_N>0))
# anova(lexDecModel.RTs.8letter.Freq.OrthoN, lexDecModel.RTs.8letter.Freq.Enemies)
summary(lexDecModel.RTs.8letter.Freq.Enemies)

# Compare regression b weights to entropy  ----
# Unadjusted b weights
lexDecCoef.Freq.Enemies.3letter <- cbind(
  summary(lexDecModel.RTs.3letter.Freq.Enemies)[["coefficients"]][3,1],
  summary(lexDecModel.RTs.3letter.Freq.Enemies)[["coefficients"]][4,1], 
  summary(lexDecModel.RTs.3letter.Freq.Enemies)[["coefficients"]][5,1]
)

lexDecCoef.Freq.Enemies.4letter <- cbind(
  summary(lexDecModel.RTs.4letter.Freq.Enemies)[["coefficients"]][3,1],
  summary(lexDecModel.RTs.4letter.Freq.Enemies)[["coefficients"]][4,1], 
  summary(lexDecModel.RTs.4letter.Freq.Enemies)[["coefficients"]][5,1],
  summary(lexDecModel.RTs.4letter.Freq.Enemies)[["coefficients"]][6,1]
)

lexDecCoef.Freq.Enemies.5letter <- cbind(
  summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][3,1],
  summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][4,1],
  summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][5,1],
  summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][6,1],
  summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][7,1]
)

lexDecCoef.Freq.Enemies.6letter <- cbind(
  summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][3,1],
  summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][4,1],
  summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][5,1],
  summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][6,1],
  summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][7,1],
  summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][8,1]
)

lexDecCoef.Freq.Enemies.7letter <- cbind(
  summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][3,1],
  summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][4,1],
  summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][5,1],
  summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][6,1],
  summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][7,1],
  summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][8,1],
  summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][9,1]
)

lexDecCoef.Freq.Enemies.8letter <- cbind(
  summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][3,1],
  summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][4,1],
  summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][5,1],
  summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][6,1],
  summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][7,1],
  summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][8,1],
  summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][9,1],
  summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][10,1]
)

lexDecCoef.Freq.Enemies.allLetter <- as.vector(cbind(lexDecCoef.Freq.Enemies.3letter,lexDecCoef.Freq.Enemies.4letter,lexDecCoef.Freq.Enemies.5letter,lexDecCoef.Freq.Enemies.6letter,lexDecCoef.Freq.Enemies.7letter, lexDecCoef.Freq.Enemies.8letter))

metrics <- cbind.data.frame(metrics, lexDecCoef.Freq.Enemies.allLetter)

# Adjusted b weights
scalFac <- nrow(subset(lexDecData.correct.word.3letter, Ortho_N>0)) / (
  1/((summary(lexDecModel.RTs.3letter.Freq.Enemies)[["coefficients"]][3,2])^2) + 
    1/((summary(lexDecModel.RTs.3letter.Freq.Enemies)[["coefficients"]][4,2])^2) + 
    1/((summary(lexDecModel.RTs.3letter.Freq.Enemies)[["coefficients"]][5,2])^2)
)
lexDecCoefAdj.Freq.Enemies.3letter <- cbind(
  summary(lexDecModel.RTs.3letter.Freq.Enemies)[["coefficients"]][3,1] * scalFac/((summary(lexDecModel.RTs.3letter.Freq.Enemies)[["coefficients"]][3,2])^2),
  summary(lexDecModel.RTs.3letter.Freq.Enemies)[["coefficients"]][4,1] * scalFac/((summary(lexDecModel.RTs.3letter.Freq.Enemies)[["coefficients"]][4,2])^2), 
  summary(lexDecModel.RTs.3letter.Freq.Enemies)[["coefficients"]][5,1] * scalFac/((summary(lexDecModel.RTs.3letter.Freq.Enemies)[["coefficients"]][5,2])^2)
)

scalFac <- nrow(subset(lexDecData.correct.word.4letter, Ortho_N>0)) / (
  1/((summary(lexDecModel.RTs.4letter.Freq.Enemies)[["coefficients"]][3,2])^2) + 
    1/((summary(lexDecModel.RTs.4letter.Freq.Enemies)[["coefficients"]][4,2])^2) + 
    1/((summary(lexDecModel.RTs.4letter.Freq.Enemies)[["coefficients"]][5,2])^2) +
    1/((summary(lexDecModel.RTs.4letter.Freq.Enemies)[["coefficients"]][6,2])^2)
)
lexDecCoefAdj.Freq.Enemies.4letter <- cbind(
  summary(lexDecModel.RTs.4letter.Freq.Enemies)[["coefficients"]][3,1] * scalFac/((summary(lexDecModel.RTs.4letter.Freq.Enemies)[["coefficients"]][3,2])^2),
  summary(lexDecModel.RTs.4letter.Freq.Enemies)[["coefficients"]][4,1] * scalFac/((summary(lexDecModel.RTs.4letter.Freq.Enemies)[["coefficients"]][4,2])^2), 
  summary(lexDecModel.RTs.4letter.Freq.Enemies)[["coefficients"]][5,1] * scalFac/((summary(lexDecModel.RTs.4letter.Freq.Enemies)[["coefficients"]][5,2])^2),
  summary(lexDecModel.RTs.4letter.Freq.Enemies)[["coefficients"]][6,1] * scalFac/((summary(lexDecModel.RTs.4letter.Freq.Enemies)[["coefficients"]][6,2])^2)
)

scalFac <- nrow(subset(lexDecData.correct.word.5letter, Ortho_N>0)) / (
  1/((summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][3,2])^2) + 
    1/((summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][4,2])^2) + 
    1/((summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][5,2])^2) +
    1/((summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][6,2])^2) + 
    1/((summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][7,2])^2) 
)
lexDecCoefAdj.Freq.Enemies.5letter <- cbind(
  summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][3,1] * scalFac/((summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][3,2])^2),
  summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][4,1] * scalFac/((summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][4,2])^2), 
  summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][5,1] * scalFac/((summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][5,2])^2),
  summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][6,1] * scalFac/((summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][6,2])^2),
  summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][7,1] * scalFac/((summary(lexDecModel.RTs.5letter.Freq.Enemies)[["coefficients"]][7,2])^2)
)

scalFac <- nrow(subset(lexDecData.correct.word.6letter, Ortho_N>0)) / (
  1/((summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][3,2])^2) + 
    1/((summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][4,2])^2) + 
    1/((summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][5,2])^2) +
    1/((summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][6,2])^2) + 
    1/((summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][7,2])^2) +
    1/((summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][8,2])^2)
)
lexDecCoefAdj.Freq.Enemies.6letter <- cbind(
  summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][3,1] * scalFac/((summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][3,2])^2),
  summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][4,1] * scalFac/((summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][4,2])^2), 
  summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][5,1] * scalFac/((summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][5,2])^2),
  summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][6,1] * scalFac/((summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][6,2])^2),
  summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][7,1] * scalFac/((summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][7,2])^2),
  summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][8,1] * scalFac/((summary(lexDecModel.RTs.6letter.Freq.Enemies)[["coefficients"]][8,2])^2)
)

scalFac <- nrow(subset(lexDecData.correct.word.7letter, Ortho_N>0)) / (
  1/((summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][3,2])^2) + 
    1/((summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][4,2])^2) + 
    1/((summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][5,2])^2) +
    1/((summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][6,2])^2) + 
    1/((summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][7,2])^2) +
    1/((summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][8,2])^2) + 
    1/((summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][9,2])^2)
)
lexDecCoefAdj.Freq.Enemies.7letter <- cbind(
  summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][3,1] * scalFac/((summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][3,2])^2),
  summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][4,1] * scalFac/((summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][4,2])^2), 
  summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][5,1] * scalFac/((summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][5,2])^2),
  summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][6,1] * scalFac/((summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][6,2])^2),
  summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][7,1] * scalFac/((summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][7,2])^2),
  summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][8,1] * scalFac/((summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][8,2])^2),
  summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][9,1] * scalFac/((summary(lexDecModel.RTs.7letter.Freq.Enemies)[["coefficients"]][9,2])^2)
)

scalFac <- nrow(subset(lexDecData.correct.word.8letter, Ortho_N>0)) / (
  1/((summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][3,2])^2) + 
    1/((summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][4,2])^2) + 
    1/((summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][5,2])^2) +
    1/((summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][6,2])^2) + 
    1/((summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][7,2])^2) +
    1/((summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][8,2])^2) + 
    1/((summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][9,2])^2) + 
    1/((summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][10,2])^2)
)
lexDecCoefAdj.Freq.Enemies.8letter <- cbind(
  summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][3,1] * scalFac/((summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][3,2])^2),
  summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][4,1] * scalFac/((summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][4,2])^2), 
  summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][5,1] * scalFac/((summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][5,2])^2),
  summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][6,1] * scalFac/((summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][6,2])^2),
  summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][7,1] * scalFac/((summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][7,2])^2),
  summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][8,1] * scalFac/((summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][8,2])^2),
  summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][9,1] * scalFac/((summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][9,2])^2),
  summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][10,1] * scalFac/((summary(lexDecModel.RTs.8letter.Freq.Enemies)[["coefficients"]][10,2])^2)
)

lexDecCoefAdj.Freq.Enemies.allLetter <- as.vector(cbind(lexDecCoefAdj.Freq.Enemies.3letter,lexDecCoefAdj.Freq.Enemies.4letter,lexDecCoefAdj.Freq.Enemies.5letter,lexDecCoefAdj.Freq.Enemies.6letter,lexDecCoefAdj.Freq.Enemies.7letter, lexDecCoefAdj.Freq.Enemies.8letter))

metrics <- cbind.data.frame(metrics, lexDecCoefAdj.Freq.Enemies.allLetter)

# Plots

ggplot(metrics, aes(entropy.allLetter, lexDecCoefAdj.Freq.Enemies.allLetter)) + 
  geom_smooth(method=lm, color = "black", alpha = 0.15) + geom_point(color = metrics$Colors) + 
  geom_text_repel(box.padding = 1, aes(label=Labels), color = metrics$Colors, 
                  size = 6, force = 3, max.iter = 5000) + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = "transparent", color = NA)) + 
  labs(x="Entropy", y = "Influence of Enemies \n on Lexical Decision Latency (b estimate)")
cor.test(entropy.allLetter, lexDecCoefAdj.Freq.Enemies.allLetter)
cor.test(entropy.allLetter[-c(1,4,8,13,19,26)], lexDecCoefAdj.Freq.Enemies.allLetter[-c(1,4,8,13,19,26)])

save.image("~/friends-and-enemies/PostScript04.RData")  
