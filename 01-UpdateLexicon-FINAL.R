# In this script, we load the ELP lexicon and compute additional lexical metrics.
# Setup: Load packages, functions --------------------------------
rm(list=ls()) #clear all variables
setwd("~/friends-and-enemies") #set working directory
library(ggplot2); library(plyr); library(dplyr); library(car); library(reshape); library(lme4); library(cowplot); library(stringi); library(scales); library(ggrepel)
theme_set(theme_classic(base_size = 36))

summarySE <- function(data=NULL, measurevar, groupvars=NULL, na.rm=FALSE, conf.interval=.95, .drop=TRUE) {
  library(plyr)
  # New version of length which can handle NA's: if na.rm==T, don't count them
  length2 <- function (x, na.rm=FALSE) {
    if (na.rm) sum(!is.na(x))
    else       length(x)
  }
  # This does the summary. For each group's data frame, return a vector with
  # N, mean, and sd
  datac <- ddply(data, groupvars, .drop=.drop,
                 .fun = function(xx, col) {
                   c(N    = length2(xx[[col]], na.rm=na.rm),
                     mean = mean   (xx[[col]], na.rm=na.rm),
                     sd   = sd     (xx[[col]], na.rm=na.rm)
                   )
                 },
                 measurevar
  )
  # Rename the "mean" column    
  datac <- rename(datac, c("mean" = measurevar))
  datac$se <- datac$sd / sqrt(datac$N)  # Calculate standard error of the mean
  # Confidence interval multiplier for standard error
  # Calculate t-statistic for confidence interval: 
  # e.g., if conf.interval is .95, use .975 (above/below), and use df=N-1
  ciMult <- qt(conf.interval/2 + .5, datac$N-1)
  datac$ci <- datac$se * ciMult
  return(datac)
}

corr_eqn <- function(x,y, digits = 3) {
  corr_coef <- round(cor(x, y), digits = digits)
  paste("italic(r) == ", corr_coef)
}

my.pairscor <- function (data, classes = "", hist = TRUE, smooth = TRUE, cex.points = .5, col.hist = "darkgrey", col.points = "blue", rsize=1, psize=1) {
  ##	print(paste("lclass: ", lclass,"\n\n,",classes))
  ##	super.sym<-trellis.par.get("superpose.symbol")
  #	super.sym<-c("black","red","blue","magenta","lightorange")
  ##	col.points <- super.sym$col[unique(unclass(classes))][unclass(classes)]
  #	col.points <- super.sym[unique(unclass(classes))]
  panel.hist <- function(x, ...) {
    usr <- par("usr")
    on.exit(par(usr))
    par(usr = c(usr[1:2], 0, 1.5))
    h <- hist(x, plot = FALSE)
    breaks <- h$breaks
    nB <- length(breaks)
    y <- h$counts
    y <- y/max(y)
    rect(breaks[-nB], 0, breaks[-1], y, ...)
  }
  pairscor.lower <- function(x, y, ...) {
    usr <- par("usr")
    on.exit(par(usr))
    par(usr = c(0, 1, 0, 1))
    m = cor.test(x, y)
    r = round(m$estimate, 2)
    rr = round((m$estimate^2), 2)
    p = round(m$p.value, 4)
    rtxt = paste("r", r, ",", rr)
    ptxt = paste("p=", p)
    options(warn = -1)
    m2 = cor.test(x, y, method = "spearman")
    r2 = round(m2$estimate, 2)
    rr2 = round((m2$estimate^2), 2)
    p2 = round(m2$p.value, 4)
    rtxt2 = paste("r=", r2,  ",", rr2)
    ptxt2 = paste("p =", p2)
    options(warn = 0)
    tcol="black"
    if(p<= .1) tcol = "green"
    if(p <= .05) tcol = "red"
    text(0.5, 0.8, rtxt, col = tcol, cex=rsize)
    text(0.5, 0.6, ptxt, col = tcol, cex=psize)
    lines(c(0.2, 0.8), c(0.5, 0.5))
    tcol="black"
    if(p2<= .1) tcol = "green"
    if(p2 < .05) tcol = "red"        
    text(0.5, 0.4, rtxt2, col = tcol, cex=rsize)
    text(0.5, 0.2, ptxt2, col = tcol, cex=psize)
  }
  panel.smooth2 = function(x, y, col = par("col"), bg = NA, 
                           pch = par("pch"), cex = 1, span = 2/3, iter = 3, ...) {
    #points(x, y, pch = pch, col = col, cex = cex)
    points(x, y, pch = pch, col = col, cex = cex)
    ok <- is.finite(x) & is.finite(y)
    if (any(ok)) 
      lines(stats::lowess(x[ok], y[ok], f = span, iter = iter), 
            col = "red", ...)
  }
  #    panel.smooth2 = function(x, y, col = par("col"), bg = NA, 
  #        pch = par("pch"), cex = 1, span = 2/3, iter = 3, ...) {
  #        points(x, y, pch = pch, col = col, bg = bgs, cex = cex)
  #        ok <- is.finite(x) & is.finite(y)
  #        if (any(ok)) 
  #            lines(stats::lowess(x[ok], y[ok], f = span, iter = iter), 
  #                col = "black", ...)
  #    }
  #    panel.smooth2 = function(x, y, col = par("col"), bg = NA, 
  #        pch = par("pch"), cex = 1, span = 2/3, iter = 3, ...) {
  #        points(x, y, pch = pch, col = col, bg = bg, cex = cex)
  #        ok <- is.finite(x) & is.finite(y)
  #        if (any(ok)) 
  #            lines(stats::lowess(x[ok], y[ok], f = span, iter = iter), 
  #                col = "black", ...)
  #    }
  if (hist == TRUE) {
    if (smooth == TRUE) {
      #        	if(lclass > 0){
      pairs(data, diag.panel = panel.hist, lower.panel = pairscor.lower, 
            upper.panel = panel.smooth2, col =col.points, 
            cex = cex.points)
    }
    #              	else{
    #                		pairs(data, diag.panel = panel.hist, lower.panel = pairscor.lower, 
    #                	upper.panel = panel.smooth2, col = col.points, 
    #                	cex = cex.points)
    #                }
    #      }
    #        else {
    #            pairs(data, diag.panel = panel.hist, lower.panel = pairscor.lower)
    #        }
  }
  else {
    if (smooth == TRUE) {
      pairs(data, lower.panel = pairscor.lower, upper.panel = panel.smooth2, 
            col = 		, cex = cex.points)
    }
    else {
      pairs(data, lower.panel = pairscor.lower)
    }
  }
}

alphabet <- c("a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z")
vowels = c("!","0","#","Y","L","a","A","@","e","E","i","I","o","O","u","U","V")

probByLet <- function(df, position) {
  prob <- sapply(alphabet, function(x) length(which(df[position]==x)) /
                   nrow(df)) 
  prob[prob==0] <- 0.000001
  prob <- as.data.frame(prob)
  prob
}

entropy <- function(df, position) {
  prob <- sapply(alphabet, function(x) length(which(df[position]==x)) /
                   nrow(df)) 
  prob[prob==0] <- 0.000001
  round(-sum(prob*log(prob,2)),2)
}

entropyFW <- function(df,position) { 
  prob <- sapply(alphabet, function(x) sum(df$Log_Freq_HAL[df[position]==x])) /
    sum(sapply(alphabet, function(x) sum(df$Log_Freq_HAL[df[position]==x])))
  prob[prob==0] <- 0.000001
  round(-sum(prob*log(prob,2)),2)
}


# Setup: Load data -------------------------
lexicon <- read.csv('lexicon.csv', header = TRUE, sep = ',', na.strings = "#N/A")

#Format lexicon
lexicon$Length <- as.numeric(as.character(lexicon$Length)); lexicon$Freq_KF <- as.numeric(as.character(lexicon$Freq_KF)); lexicon$Freq_HAL <- as.numeric(as.character(lexicon$Freq_HAL)); lexicon$Log_Freq_HAL <- as.numeric(as.character(lexicon$Log_Freq_HAL)); lexicon$Log_Freq_HAL <- as.numeric(as.character(lexicon$Log_Freq_HAL)); lexicon$SUBTLWF <- as.numeric(as.character(lexicon$SUBTLWF)); lexicon$LgSUBTLWF <- as.numeric(as.character(lexicon$LgSUBTLWF)); lexicon$SUBTLCD <- as.numeric(as.character(lexicon$SUBTLCD)); lexicon$LgSUBTLCD <- as.numeric(as.character(lexicon$LgSUBTLCD));
lexicon$Ortho_N <- as.numeric(as.character(lexicon$Ortho_N)); lexicon$Phono_N <- as.numeric(as.character(lexicon$Phono_N)); lexicon$OG_N <- as.numeric(as.character(lexicon$OG_N)); lexicon$Freq_N <- as.numeric(as.character(lexicon$Freq_N)); lexicon$Freq_N_P <- as.numeric(as.character(lexicon$Freq_N_P)); lexicon$Freq_N_PH <- as.numeric(as.character(lexicon$Freq_N_PH)); lexicon$Freq_N_OG <- as.numeric(as.character(lexicon$Freq_N_OG)); lexicon$Freq_N_OGH <- as.numeric(as.character(lexicon$Freq_N_OGH)); lexicon$Freq_Greater <- as.numeric(as.character(lexicon$Freq_Greater)); 
lexicon$Freq_G_Mean <- as.numeric(as.character(lexicon$Freq_G_Mean)); lexicon$Freq_Less <- as.numeric(as.character(lexicon$Freq_Less)); lexicon$Freq_L_Mean <- as.numeric(as.character(lexicon$Freq_L_Mean)); lexicon$Freq_Rel <- as.numeric(as.character(lexicon$Freq_Rel)); lexicon$OLD <- as.numeric(as.character(lexicon$OLD)); lexicon$OLDF <- as.numeric(as.character(lexicon$OLDF)); lexicon$PLD <- as.numeric(as.character(lexicon$PLD));   lexicon$PLDF <- as.numeric(as.character(lexicon$PLDF)); lexicon$BG_Sum <- as.numeric(as.character(lexicon$BG_Sum)); lexicon$BG_Mean <- as.numeric(as.character(lexicon$BG_Mean));
lexicon$BG_Freq_By_Pos <- as.numeric(as.character(lexicon$BG_Freq_By_Pos)); lexicon$NPhon <- as.numeric(as.character(lexicon$NPhon)); lexicon$NMorph <- as.numeric(as.character(lexicon$NMorph));  lexicon$I_Mean_RT <- as.numeric(as.character(lexicon$I_Mean_RT)); lexicon$I_SD <- as.numeric(as.character(lexicon$I_SD)); lexicon$Obs <- as.numeric(as.character(lexicon$Obs)); lexicon$I_Mean_Accuracy <- as.numeric(as.character(lexicon$I_Mean_Accuracy)); lexicon$I_NMG_Mean_RT <- as.numeric(as.character(lexicon$I_NMG_Mean_RT)); lexicon$I_NMG_SD <- as.numeric(as.character(lexicon$I_NMG_SD)); 
lexicon$I_NMG_Obs <- as.numeric(as.character(lexicon$I_NMG_Obs)); lexicon$I_NMG_Mean_Accuracy <- as.numeric(as.character(lexicon$I_NMG_Mean_Accuracy))

# Compute metrics not specified in ELP ------------------------------------

#Orthographic cohort
#How many words match the target in the first two letter positions?
lexicon$Ortho.Cohort <- sapply(substr(lexicon$Word,1,2), function(x)
  length(lexicon$Word[grep(paste0("^",x), lexicon$Word, 
                           ignore.case = FALSE, perl = TRUE)])-1)


#The phonetic transcriptions in the ELP have some problems, so we created a file with edited transcriptions
#(1) Multicharacter phonetic codes are now represented by single character codes
#aI Ice --> !   #aU OUt --> 0   #@` buttER --> #   #dZ baDGe --> J  #tS caTCH --> C   #OI OYster --> Y    #3` bIRd --> # (same as "butter")  l= bottLe --> L, = --> NULL
#(2) Flap was replaced with t, since they have the same features
#(3) Low frequency transcriptions (R, H, x, X, _) were determined to be errors and corrected by item
pronun <- read.csv('ELP.Pron.Corrected.csv', header = TRUE, sep = ',', na.strings = "#N/A")
lexicon <- lexicon[which(colnames(lexicon)!="Pron_NoStress")]; lexicon <- lexicon[which(colnames(lexicon)!="X")]
lexicon <- merge(lexicon, pronun, by = 'Word')

#phonetic transcription legend:
#0=OUt; #=bIRd, buttER; t = Toy, beTTer; !=Ice; @=Above; a=hAt; A=cAr; b=Boy; c=caTCH; d=Dog; D=THis;
#e=Ape; E=Ebb; f=Fig; g=Go; h=Hip; i=EAt; I=If; j = Yes; J=baDGE; k=Kite; l=Lip; L=bottLE; 
#m=deisM; n=buttoN; N=siNG; o=OAt; O=AUto; p=Pig; r=Reead; s=Sew; S=SHow; t=Toy; T=THin; 
#u=rUde; U=pUt; v=Van; V=Up; w=Wind; Y=OYster; z=Zoo; Z=viSion

# Phonological cohort and rhymes
# First, we must identify phonetic onset
lexicon$Pron_NoStress <- as.character(lexicon$Pron_NoStress)
for(n in 1:length(lexicon$Pron_NoStress)){
  vowelpos = -1
  for (i in 1:nchar(lexicon$Pron_NoStress[n])) {
    if(length(which(grepl(substr(lexicon$Pron_NoStress[n],i,i),vowels)))==1){
      if(vowelpos ==-1){
        vowelpos = i
      }
    }
  }
  lexicon$onset[n] <- substr(lexicon$Pron_NoStress[n],1,vowelpos)
}

## And then identify offset

for(n in 1:length(lexicon$Pron_NoStress)){
  vowelpos = -1
  for (i in nchar(lexicon$Pron_NoStress[n]):1) {
    if(length(which(grepl(substr(lexicon$Pron_NoStress[n],i,i),vowels)))==1){
      if(vowelpos ==-1){
        vowelpos = i
      }
    }
  }
  lexicon$offset[n] <- substr(lexicon$Pron_NoStress[n],vowelpos,nchar(lexicon$Pron_NoStress[n]))
}

lexicon <- subset(lexicon, Pron_NoStress != '---')

#Now we can count cohorts based on phonology
lexicon$Phono.Cohort <- sapply(lexicon$onset, function(x) 
  length(lexicon$Word[grep(paste0("^",x), lexicon$Pron_NoStress, 
                           ignore.case = FALSE, perl = TRUE)])-1) 

#And count rhymes
lexicon$NRhymes <- sapply(lexicon$offset, function(x) 
  length(lexicon$Word[grep(paste0(x,"$"), lexicon$Pron_NoStress, 
                           ignore.case = FALSE, perl = TRUE)])-1) 

# We can also count homophones
lexicon$NHomophones <- sapply(lexicon$Word, function(x) 
  length(setdiff(which(lexicon$Pron[match(x,lexicon$Word)]==lexicon$Pron),
                 c(match(x,lexicon$Word),which(lexicon$Pron=="NA")))))

# Create smaller lexica (based on word length) -------------------
lexicon.3letter <- droplevels(subset(lexicon, Length == 3))
lexicon.4letter <- droplevels(subset(lexicon, Length == 4))
lexicon.5letter <- droplevels(subset(lexicon, Length == 5))
lexicon.6letter <- droplevels(subset(lexicon, Length == 6))
lexicon.7letter <- droplevels(subset(lexicon, Length == 7))
lexicon.8letter <- droplevels(subset(lexicon, Length == 8))

# Count enemies by position -------------------------
#3-letter words
lexicon.3letter$Enemies1 <- sapply(lexicon.3letter$Word, function(x) 
  length(lexicon.3letter$Word[grep(substr(x,2,3), substr(lexicon.3letter$Word,2,3), 
                                   value = FALSE, ignore.case = TRUE, perl = TRUE)])-1) 

lexicon.3letter$Enemies2 <- sapply(lexicon.3letter$Word, function(x) 
  length(lexicon.3letter$Word[intersect(grep(substr(x,1,1), substr(lexicon.3letter$Word,1,1), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,3,3), substr(lexicon.3letter$Word,3,3), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.3letter$Enemies3 <- sapply(lexicon.3letter$Word, function(x) 
  length(lexicon.3letter$Word[grep(substr(x,1,2), substr(lexicon.3letter$Word,1,2), value = FALSE, ignore.case = TRUE, perl = TRUE)])-1) 

#4-letter words  
lexicon.4letter$Enemies1 <- sapply(lexicon.4letter$Word, function(x) 
  length(lexicon.4letter$Word[grep(substr(x,2,4), substr(lexicon.4letter$Word,2,4), 
                                   value = FALSE, ignore.case = TRUE, perl = TRUE)])-1) 

lexicon.4letter$Enemies2 <- sapply(lexicon.4letter$Word, function(x) 
  length(lexicon.4letter$Word[intersect(grep(substr(x,1,1), substr(lexicon.4letter$Word,1,1), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,3,4), substr(lexicon.4letter$Word,3,4), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.4letter$Enemies3 <- sapply(lexicon.4letter$Word, function(x) 
  length(lexicon.4letter$Word[intersect(grep(substr(x,1,2), substr(lexicon.4letter$Word,1,2), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,4,4), substr(lexicon.4letter$Word,4,4), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.4letter$Enemies4 <- sapply(lexicon.4letter$Word, function(x) 
  length(lexicon.4letter$Word[grep(substr(x,1,3), substr(lexicon.4letter$Word,1,3), value = FALSE, ignore.case = TRUE, perl = TRUE)])-1) 

#5-letter words  
lexicon.5letter$Enemies1 <- sapply(lexicon.5letter$Word, function(x) 
  length(lexicon.5letter$Word[grep(substr(x,2,5), substr(lexicon.5letter$Word,2,5), value = FALSE, ignore.case = TRUE, perl = TRUE)])-1) 

lexicon.5letter$Enemies2 <- sapply(lexicon.5letter$Word, function(x) 
  length(lexicon.5letter$Word[intersect(grep(substr(x,1,1), substr(lexicon.5letter$Word,1,1), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,3,5), substr(lexicon.5letter$Word,3,5), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.5letter$Enemies3 <- sapply(lexicon.5letter$Word, function(x) 
  length(lexicon.5letter$Word[intersect(grep(substr(x,1,2), substr(lexicon.5letter$Word,1,2), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,4,5), substr(lexicon.5letter$Word,4,5), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.5letter$Enemies4 <- sapply(lexicon.5letter$Word, function(x) 
  length(lexicon.5letter$Word[intersect(grep(substr(x,1,3), substr(lexicon.5letter$Word,1,3), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,5,5), substr(lexicon.5letter$Word,5,5), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.5letter$Enemies5 <- sapply(lexicon.5letter$Word, function(x) 
  length(lexicon.5letter$Word[grep(substr(x,1,4), substr(lexicon.5letter$Word,1,4), value = FALSE, ignore.case = TRUE, perl = TRUE)])-1) 

#6-letter words  
lexicon.6letter$Enemies1 <- sapply(lexicon.6letter$Word, function(x) 
  length(lexicon.6letter$Word[grep(substr(x,2,6), substr(lexicon.6letter$Word,2,6), value = FALSE, ignore.case = TRUE, perl = TRUE)])-1) 

lexicon.6letter$Enemies2 <- sapply(lexicon.6letter$Word, function(x) 
  length(lexicon.6letter$Word[intersect(grep(substr(x,1,1), substr(lexicon.6letter$Word,1,1), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,3,6), substr(lexicon.6letter$Word,3,6), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.6letter$Enemies3 <- sapply(lexicon.6letter$Word, function(x) 
  length(lexicon.6letter$Word[intersect(grep(substr(x,1,2), substr(lexicon.6letter$Word,1,2), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,4,6), substr(lexicon.6letter$Word,4,6), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.6letter$Enemies4 <- sapply(lexicon.6letter$Word, function(x) 
  length(lexicon.6letter$Word[intersect(grep(substr(x,1,3), substr(lexicon.6letter$Word,1,3), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,5,6), substr(lexicon.6letter$Word,5,6), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.6letter$Enemies5 <- sapply(lexicon.6letter$Word, function(x) 
  length(lexicon.6letter$Word[intersect(grep(substr(x,1,4), substr(lexicon.6letter$Word,1,4), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,6,6), substr(lexicon.6letter$Word,6,6), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.6letter$Enemies6 <- sapply(lexicon.6letter$Word, function(x) 
  length(lexicon.6letter$Word[grep(substr(x,1,5), substr(lexicon.6letter$Word,1,5), value = FALSE, ignore.case = TRUE, perl = TRUE)])-1) 

#7-letter words  
lexicon.7letter$Enemies1 <- sapply(lexicon.7letter$Word, function(x) 
  length(lexicon.7letter$Word[grep(substr(x,2,7), substr(lexicon.7letter$Word,2,7), value = FALSE, ignore.case = TRUE, perl = TRUE)])-1) 

lexicon.7letter$Enemies2 <- sapply(lexicon.7letter$Word, function(x) 
  length(lexicon.7letter$Word[intersect(grep(substr(x,1,1), substr(lexicon.7letter$Word,1,1), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,3,7), substr(lexicon.7letter$Word,3,7), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.7letter$Enemies3 <- sapply(lexicon.7letter$Word, function(x) 
  length(lexicon.7letter$Word[intersect(grep(substr(x,1,2), substr(lexicon.7letter$Word,1,2), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,4,7), substr(lexicon.7letter$Word,4,7), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.7letter$Enemies4 <- sapply(lexicon.7letter$Word, function(x) 
  length(lexicon.7letter$Word[intersect(grep(substr(x,1,3), substr(lexicon.7letter$Word,1,3), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,5,7), substr(lexicon.7letter$Word,5,7), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.7letter$Enemies5 <- sapply(lexicon.7letter$Word, function(x) 
  length(lexicon.7letter$Word[intersect(grep(substr(x,1,4), substr(lexicon.7letter$Word,1,4), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,6,7), substr(lexicon.7letter$Word,6,7), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.7letter$Enemies6 <- sapply(lexicon.7letter$Word, function(x) 
  length(lexicon.7letter$Word[intersect(grep(substr(x,1,5), substr(lexicon.7letter$Word,1,5), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,7,7), substr(lexicon.7letter$Word,7,7), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.7letter$Enemies7 <- sapply(lexicon.7letter$Word, function(x) 
  length(lexicon.7letter$Word[grep(substr(x,1,6), substr(lexicon.7letter$Word,1,6), value = FALSE, ignore.case = TRUE, perl = TRUE)])-1) 


#8-letter words  
lexicon.8letter$Enemies1 <- sapply(lexicon.8letter$Word, function(x) 
  length(lexicon.8letter$Word[grep(substr(x,2,8), substr(lexicon.8letter$Word,2,8), value = FALSE, ignore.case = TRUE, perl = TRUE)])-1) 

lexicon.8letter$Enemies2 <- sapply(lexicon.8letter$Word, function(x) 
  length(lexicon.8letter$Word[intersect(grep(substr(x,1,1), substr(lexicon.8letter$Word,1,1), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,3,8), substr(lexicon.8letter$Word,3,8), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.8letter$Enemies3 <- sapply(lexicon.8letter$Word, function(x) 
  length(lexicon.8letter$Word[intersect(grep(substr(x,1,2), substr(lexicon.8letter$Word,1,2), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,4,8), substr(lexicon.8letter$Word,4,8), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.8letter$Enemies4 <- sapply(lexicon.8letter$Word, function(x) 
  length(lexicon.8letter$Word[intersect(grep(substr(x,1,3), substr(lexicon.8letter$Word,1,3), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,5,8), substr(lexicon.8letter$Word,5,8), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.8letter$Enemies5 <- sapply(lexicon.8letter$Word, function(x) 
  length(lexicon.8letter$Word[intersect(grep(substr(x,1,4), substr(lexicon.8letter$Word,1,4), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,6,8), substr(lexicon.8letter$Word,6,8), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.8letter$Enemies6 <- sapply(lexicon.8letter$Word, function(x) 
  length(lexicon.8letter$Word[intersect(grep(substr(x,1,5), substr(lexicon.8letter$Word,1,5), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,7,8), substr(lexicon.8letter$Word,7,8), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.8letter$Enemies7 <- sapply(lexicon.8letter$Word, function(x) 
  length(lexicon.8letter$Word[intersect(grep(substr(x,1,6), substr(lexicon.8letter$Word,1,6), value = FALSE, ignore.case = TRUE, perl = TRUE), 
                                        grep(substr(x,8,8), substr(lexicon.8letter$Word,8,8), value = FALSE, ignore.case = TRUE, perl = TRUE))])-1) 

lexicon.8letter$Enemies8 <- sapply(lexicon.8letter$Word, function(x) 
  length(lexicon.8letter$Word[grep(substr(x,1,7), substr(lexicon.8letter$Word,1,7), value = FALSE, ignore.case = TRUE, perl = TRUE)])-1) 

enemies.allLetter <- as.vector(cbind(cbind(mean(lexicon.3letter$Enemies1),mean(lexicon.3letter$Enemies2),mean(lexicon.3letter$Enemies3)),
                                     cbind(mean(lexicon.4letter$Enemies1),mean(lexicon.4letter$Enemies2),mean(lexicon.4letter$Enemies3),mean(lexicon.4letter$Enemies4)),
                                     cbind(mean(lexicon.5letter$Enemies1),mean(lexicon.5letter$Enemies2),mean(lexicon.5letter$Enemies3),mean(lexicon.5letter$Enemies4),mean(lexicon.5letter$Enemies5)),
                                     cbind(mean(lexicon.6letter$Enemies1),mean(lexicon.6letter$Enemies2),mean(lexicon.6letter$Enemies3),mean(lexicon.6letter$Enemies4),mean(lexicon.6letter$Enemies5),mean(lexicon.6letter$Enemies6)),
                                     cbind(mean(lexicon.7letter$Enemies1),mean(lexicon.7letter$Enemies2),mean(lexicon.7letter$Enemies3),mean(lexicon.7letter$Enemies4),mean(lexicon.7letter$Enemies5),mean(lexicon.7letter$Enemies6),mean(lexicon.7letter$Enemies7)),
                                     cbind(mean(lexicon.8letter$Enemies1),mean(lexicon.8letter$Enemies2),mean(lexicon.8letter$Enemies3),mean(lexicon.8letter$Enemies4),mean(lexicon.8letter$Enemies5),mean(lexicon.8letter$Enemies6),mean(lexicon.8letter$Enemies7)),mean(lexicon.8letter$Enemies8)))
# Count friends by position ----
lexicon.3letter$Friends1 <- lexicon.3letter$Enemies2 + lexicon.3letter$Enemies3 
lexicon.3letter$Friends2 <- lexicon.3letter$Enemies1 + lexicon.3letter$Enemies3 
lexicon.3letter$Friends3 <- lexicon.3letter$Enemies1 + lexicon.3letter$Enemies2 

lexicon.4letter$Friends1 <- lexicon.4letter$Enemies2 + lexicon.4letter$Enemies3 + lexicon.4letter$Enemies4
lexicon.4letter$Friends2 <- lexicon.4letter$Enemies1 + lexicon.4letter$Enemies3 + lexicon.4letter$Enemies4
lexicon.4letter$Friends3 <- lexicon.4letter$Enemies1 + lexicon.4letter$Enemies2 + lexicon.4letter$Enemies4
lexicon.4letter$Friends4 <- lexicon.4letter$Enemies1 + lexicon.4letter$Enemies2 + lexicon.4letter$Enemies3

lexicon.5letter$Friends1 <- lexicon.5letter$Enemies2 + lexicon.5letter$Enemies3 + lexicon.5letter$Enemies4 + lexicon.5letter$Enemies5
lexicon.5letter$Friends2 <- lexicon.5letter$Enemies1 + lexicon.5letter$Enemies3 + lexicon.5letter$Enemies4 + lexicon.5letter$Enemies5
lexicon.5letter$Friends3 <- lexicon.5letter$Enemies1 + lexicon.5letter$Enemies2 + lexicon.5letter$Enemies4 + lexicon.5letter$Enemies5
lexicon.5letter$Friends4 <- lexicon.5letter$Enemies1 + lexicon.5letter$Enemies2 + lexicon.5letter$Enemies3 + lexicon.5letter$Enemies5
lexicon.5letter$Friends5 <- lexicon.5letter$Enemies1 + lexicon.5letter$Enemies2 + lexicon.5letter$Enemies3 + lexicon.5letter$Enemies4

lexicon.6letter$Friends1 <- lexicon.6letter$Enemies2 + lexicon.6letter$Enemies3 + lexicon.6letter$Enemies4 + lexicon.6letter$Enemies5 + lexicon.6letter$Enemies6
lexicon.6letter$Friends2 <- lexicon.6letter$Enemies1 + lexicon.6letter$Enemies3 + lexicon.6letter$Enemies4 + lexicon.6letter$Enemies5 + lexicon.6letter$Enemies6
lexicon.6letter$Friends3 <- lexicon.6letter$Enemies1 + lexicon.6letter$Enemies2 + lexicon.6letter$Enemies4 + lexicon.6letter$Enemies5 + lexicon.6letter$Enemies6
lexicon.6letter$Friends4 <- lexicon.6letter$Enemies1 + lexicon.6letter$Enemies2 + lexicon.6letter$Enemies3 + lexicon.6letter$Enemies5 + lexicon.6letter$Enemies6
lexicon.6letter$Friends5 <- lexicon.6letter$Enemies1 + lexicon.6letter$Enemies2 + lexicon.6letter$Enemies3 + lexicon.6letter$Enemies4 + lexicon.6letter$Enemies6
lexicon.6letter$Friends6 <- lexicon.6letter$Enemies1 + lexicon.6letter$Enemies2 + lexicon.6letter$Enemies3 + lexicon.6letter$Enemies4 + lexicon.6letter$Enemies5

lexicon.7letter$Friends1 <- lexicon.7letter$Enemies2 + lexicon.7letter$Enemies3 + lexicon.7letter$Enemies4 + lexicon.7letter$Enemies5 + lexicon.7letter$Enemies6 + lexicon.7letter$Enemies7
lexicon.7letter$Friends2 <- lexicon.7letter$Enemies1 + lexicon.7letter$Enemies3 + lexicon.7letter$Enemies4 + lexicon.7letter$Enemies5 + lexicon.7letter$Enemies6 + lexicon.7letter$Enemies7
lexicon.7letter$Friends3 <- lexicon.7letter$Enemies1 + lexicon.7letter$Enemies2 + lexicon.7letter$Enemies4 + lexicon.7letter$Enemies5 + lexicon.7letter$Enemies6 + lexicon.7letter$Enemies7
lexicon.7letter$Friends4 <- lexicon.7letter$Enemies1 + lexicon.7letter$Enemies2 + lexicon.7letter$Enemies3 + lexicon.7letter$Enemies5 + lexicon.7letter$Enemies6 + lexicon.7letter$Enemies7
lexicon.7letter$Friends5 <- lexicon.7letter$Enemies1 + lexicon.7letter$Enemies2 + lexicon.7letter$Enemies3 + lexicon.7letter$Enemies4 + lexicon.7letter$Enemies6 + lexicon.7letter$Enemies7
lexicon.7letter$Friends6 <- lexicon.7letter$Enemies1 + lexicon.7letter$Enemies2 + lexicon.7letter$Enemies3 + lexicon.7letter$Enemies4 + lexicon.7letter$Enemies5 + lexicon.7letter$Enemies7
lexicon.7letter$Friends7 <- lexicon.7letter$Enemies1 + lexicon.7letter$Enemies2 + lexicon.7letter$Enemies3 + lexicon.7letter$Enemies4 + lexicon.7letter$Enemies5 + lexicon.7letter$Enemies6

lexicon.8letter$Friends1 <- lexicon.8letter$Enemies2 + lexicon.8letter$Enemies3 + lexicon.8letter$Enemies4 + lexicon.8letter$Enemies5 + lexicon.8letter$Enemies6 + lexicon.8letter$Enemies7  + lexicon.8letter$Enemies8 
lexicon.8letter$Friends2 <- lexicon.8letter$Enemies1 + lexicon.8letter$Enemies3 + lexicon.8letter$Enemies4 + lexicon.8letter$Enemies5 + lexicon.8letter$Enemies6 + lexicon.8letter$Enemies7  + lexicon.8letter$Enemies8 
lexicon.8letter$Friends3 <- lexicon.8letter$Enemies1 + lexicon.8letter$Enemies2 + lexicon.8letter$Enemies4 + lexicon.8letter$Enemies5 + lexicon.8letter$Enemies6 + lexicon.8letter$Enemies7  + lexicon.8letter$Enemies8 
lexicon.8letter$Friends4 <- lexicon.8letter$Enemies1 + lexicon.8letter$Enemies2 + lexicon.8letter$Enemies3 + lexicon.8letter$Enemies5 + lexicon.8letter$Enemies6 + lexicon.8letter$Enemies7  + lexicon.8letter$Enemies8 
lexicon.8letter$Friends5 <- lexicon.8letter$Enemies1 + lexicon.8letter$Enemies2 + lexicon.8letter$Enemies3 + lexicon.8letter$Enemies4 + lexicon.8letter$Enemies6 + lexicon.8letter$Enemies7  + lexicon.8letter$Enemies8 
lexicon.8letter$Friends6 <- lexicon.8letter$Enemies1 + lexicon.8letter$Enemies2 + lexicon.8letter$Enemies3 + lexicon.8letter$Enemies4 + lexicon.8letter$Enemies5 + lexicon.8letter$Enemies7  + lexicon.8letter$Enemies8 
lexicon.8letter$Friends7 <- lexicon.8letter$Enemies1 + lexicon.8letter$Enemies2 + lexicon.8letter$Enemies3 + lexicon.8letter$Enemies4 + lexicon.8letter$Enemies5 + lexicon.8letter$Enemies6  + lexicon.8letter$Enemies8 
lexicon.8letter$Friends8 <- lexicon.8letter$Enemies1 + lexicon.8letter$Enemies2 + lexicon.8letter$Enemies3 + lexicon.8letter$Enemies4 + lexicon.8letter$Enemies5 + lexicon.8letter$Enemies6  + lexicon.8letter$Enemies7 

friends.allLetter <- as.vector(cbind(cbind(mean(lexicon.3letter$Friends1),mean(lexicon.3letter$Friends2),mean(lexicon.3letter$Friends3)),
                                     cbind(mean(lexicon.4letter$Friends1),mean(lexicon.4letter$Friends2),mean(lexicon.4letter$Friends3),mean(lexicon.4letter$Friends4)),
                                     cbind(mean(lexicon.5letter$Friends1),mean(lexicon.5letter$Friends2),mean(lexicon.5letter$Friends3),mean(lexicon.5letter$Friends4),mean(lexicon.5letter$Friends5)),
                                     cbind(mean(lexicon.6letter$Friends1),mean(lexicon.6letter$Friends2),mean(lexicon.6letter$Friends3),mean(lexicon.6letter$Friends4),mean(lexicon.6letter$Friends5),mean(lexicon.6letter$Friends6)),
                                     cbind(mean(lexicon.7letter$Friends1),mean(lexicon.7letter$Friends2),mean(lexicon.7letter$Friends3),mean(lexicon.7letter$Friends4),mean(lexicon.7letter$Friends5),mean(lexicon.7letter$Friends6),mean(lexicon.7letter$Friends7)),
                                     cbind(mean(lexicon.8letter$Friends1),mean(lexicon.8letter$Friends2),mean(lexicon.8letter$Friends3),mean(lexicon.8letter$Friends4),mean(lexicon.8letter$Friends5),mean(lexicon.8letter$Friends6),mean(lexicon.8letter$Friends7)),mean(lexicon.8letter$Friends8)))

# Compute a priori entropy ----
# First, we need to create a column with each letter
lexicon.3letter$FirstLetter <- substr(lexicon.3letter$Word,1,1)
lexicon.3letter$SecondLetter <- substr(lexicon.3letter$Word,2,2)
lexicon.3letter$ThirdLetter <- substr(lexicon.3letter$Word,3,3)

lexicon.4letter$FirstLetter <- substr(lexicon.4letter$Word,1,1)
lexicon.4letter$SecondLetter <- substr(lexicon.4letter$Word,2,2)
lexicon.4letter$ThirdLetter <- substr(lexicon.4letter$Word,3,3)
lexicon.4letter$FourthLetter <- substr(lexicon.4letter$Word,4,4)

lexicon.5letter$FirstLetter <- substr(lexicon.5letter$Word,1,1)
lexicon.5letter$SecondLetter <- substr(lexicon.5letter$Word,2,2)
lexicon.5letter$ThirdLetter <- substr(lexicon.5letter$Word,3,3)
lexicon.5letter$FourthLetter <- substr(lexicon.5letter$Word,4,4)
lexicon.5letter$FifthLetter <- substr(lexicon.5letter$Word,5,5)

lexicon.6letter$FirstLetter <- substr(lexicon.6letter$Word,1,1)
lexicon.6letter$SecondLetter <- substr(lexicon.6letter$Word,2,2)
lexicon.6letter$ThirdLetter <- substr(lexicon.6letter$Word,3,3)
lexicon.6letter$FourthLetter <- substr(lexicon.6letter$Word,4,4)
lexicon.6letter$FifthLetter <- substr(lexicon.6letter$Word,5,5)
lexicon.6letter$SixthLetter <- substr(lexicon.6letter$Word,6,6)

lexicon.7letter$FirstLetter <- substr(lexicon.7letter$Word,1,1)
lexicon.7letter$SecondLetter <- substr(lexicon.7letter$Word,2,2)
lexicon.7letter$ThirdLetter <- substr(lexicon.7letter$Word,3,3)
lexicon.7letter$FourthLetter <- substr(lexicon.7letter$Word,4,4)
lexicon.7letter$FifthLetter <- substr(lexicon.7letter$Word,5,5)
lexicon.7letter$SixthLetter <- substr(lexicon.7letter$Word,6,6)
lexicon.7letter$SeventhLetter <- substr(lexicon.7letter$Word,7,7)

lexicon.8letter$FirstLetter <- substr(lexicon.8letter$Word,1,1)
lexicon.8letter$SecondLetter <- substr(lexicon.8letter$Word,2,2)
lexicon.8letter$ThirdLetter <- substr(lexicon.8letter$Word,3,3)
lexicon.8letter$FourthLetter <- substr(lexicon.8letter$Word,4,4)
lexicon.8letter$FifthLetter <- substr(lexicon.8letter$Word,5,5)
lexicon.8letter$SixthLetter <- substr(lexicon.8letter$Word,6,6)
lexicon.8letter$SeventhLetter <- substr(lexicon.8letter$Word,7,7)
lexicon.8letter$EighthLetter <- substr(lexicon.8letter$Word,8,8)

#Change uppercase letters to lowercase
lexicon.3letter$FirstLetter <- stri_trans_tolower(lexicon.3letter$FirstLetter)
lexicon.4letter$FirstLetter <- stri_trans_tolower(lexicon.4letter$FirstLetter)
lexicon.5letter$FirstLetter <- stri_trans_tolower(lexicon.5letter$FirstLetter)
lexicon.6letter$FirstLetter <- stri_trans_tolower(lexicon.6letter$FirstLetter)
lexicon.7letter$FirstLetter <- stri_trans_tolower(lexicon.7letter$FirstLetter)
lexicon.8letter$FirstLetter <- stri_trans_tolower(lexicon.8letter$FirstLetter)
lexicon.9letter$FirstLetter <- stri_trans_tolower(lexicon.9letter$FirstLetter)
lexicon.10letter$FirstLetter <- stri_trans_tolower(lexicon.10letter$FirstLetter)
lexicon.11letter$FirstLetter <- stri_trans_tolower(lexicon.11letter$FirstLetter)
lexicon.12letter$FirstLetter <- stri_trans_tolower(lexicon.12letter$FirstLetter)

#Set each letter vector as a factor
lexicon.3letter$FirstLetter <- as.factor(lexicon.3letter$FirstLetter)
lexicon.3letter$SecondLetter <- as.factor(lexicon.3letter$SecondLetter)
lexicon.3letter$ThirdLetter <- as.factor(lexicon.3letter$ThirdLetter)

lexicon.4letter$FirstLetter <- as.factor(lexicon.4letter$FirstLetter)
lexicon.4letter$SecondLetter <- as.factor(lexicon.4letter$SecondLetter)
lexicon.4letter$ThirdLetter <- as.factor(lexicon.4letter$ThirdLetter)
lexicon.4letter$FourthLetter <- as.factor(lexicon.4letter$FourthLetter)

lexicon.5letter$FirstLetter <- as.factor(lexicon.5letter$FirstLetter)
lexicon.5letter$SecondLetter <- as.factor(lexicon.5letter$SecondLetter)
lexicon.5letter$ThirdLetter <- as.factor(lexicon.5letter$ThirdLetter)
lexicon.5letter$FourthLetter <- as.factor(lexicon.5letter$FourthLetter)
lexicon.5letter$FifthLetter <- as.factor(lexicon.5letter$FifthLetter)

lexicon.6letter$FirstLetter <- as.factor(lexicon.6letter$FirstLetter)
lexicon.6letter$SecondLetter <- as.factor(lexicon.6letter$SecondLetter)
lexicon.6letter$ThirdLetter <- as.factor(lexicon.6letter$ThirdLetter)
lexicon.6letter$FourthLetter <- as.factor(lexicon.6letter$FourthLetter)
lexicon.6letter$FifthLetter <- as.factor(lexicon.6letter$FifthLetter)
lexicon.6letter$SixthLetter <- as.factor(lexicon.6letter$SixthLetter)

lexicon.7letter$FirstLetter <- as.factor(lexicon.7letter$FirstLetter)
lexicon.7letter$SecondLetter <- as.factor(lexicon.7letter$SecondLetter)
lexicon.7letter$ThirdLetter <- as.factor(lexicon.7letter$ThirdLetter)
lexicon.7letter$FourthLetter <- as.factor(lexicon.7letter$FourthLetter)
lexicon.7letter$FifthLetter <- as.factor(lexicon.7letter$FifthLetter)
lexicon.7letter$SixthLetter <- as.factor(lexicon.7letter$SixthLetter)
lexicon.7letter$SeventhLetter <- as.factor(lexicon.7letter$SeventhLetter)

lexicon.8letter$FirstLetter <- as.factor(lexicon.8letter$FirstLetter)
lexicon.8letter$SecondLetter <- as.factor(lexicon.8letter$SecondLetter)
lexicon.8letter$ThirdLetter <- as.factor(lexicon.8letter$ThirdLetter)
lexicon.8letter$FourthLetter <- as.factor(lexicon.8letter$FourthLetter)
lexicon.8letter$FifthLetter <- as.factor(lexicon.8letter$FifthLetter)
lexicon.8letter$SixthLetter <- as.factor(lexicon.8letter$SixthLetter)
lexicon.8letter$SeventhLetter <- as.factor(lexicon.8letter$SeventhLetter)
lexicon.8letter$EighthLetter <- as.factor(lexicon.8letter$EighthLetter)

#and calculate entropy (low entropy means less uncertainty, i.e., more predictable)
#entropy = − ∑ p(xi)*log2(p(xi))

entropy.3letter <- cbind(
  entropy(lexicon.3letter,"FirstLetter"),
  entropy(lexicon.3letter,"SecondLetter"),
  entropy(lexicon.3letter,"ThirdLetter"))

entropy.4letter <- cbind(
  entropy(lexicon.4letter,"FirstLetter"),
  entropy(lexicon.4letter,"SecondLetter"),
  entropy(lexicon.4letter,"ThirdLetter"),
  entropy(lexicon.4letter,"FourthLetter"))

entropy.5letter <- cbind(
  entropy(lexicon.5letter,"FirstLetter"),
  entropy(lexicon.5letter,"SecondLetter"),
  entropy(lexicon.5letter,"ThirdLetter"),
  entropy(lexicon.5letter,"FourthLetter"),
  entropy(lexicon.5letter,"FifthLetter"))

entropy.6letter <- cbind(
  entropy(lexicon.6letter,"FirstLetter"),
  entropy(lexicon.6letter,"SecondLetter"),
  entropy(lexicon.6letter,"ThirdLetter"),
  entropy(lexicon.6letter,"FourthLetter"),
  entropy(lexicon.6letter,"FifthLetter"),
  entropy(lexicon.6letter,"SixthLetter"))

entropy.7letter <- cbind(
  entropy(lexicon.7letter,"FirstLetter"),
  entropy(lexicon.7letter,"SecondLetter"),
  entropy(lexicon.7letter,"ThirdLetter"),
  entropy(lexicon.7letter,"FourthLetter"),
  entropy(lexicon.7letter,"FifthLetter"),
  entropy(lexicon.7letter,"SixthLetter"),
  entropy(lexicon.7letter,"SeventhLetter"))

entropy.8letter <- cbind(
  entropy(lexicon.8letter,"FirstLetter"),
  entropy(lexicon.8letter,"SecondLetter"),
  entropy(lexicon.8letter,"ThirdLetter"),
  entropy(lexicon.8letter,"FourthLetter"),
  entropy(lexicon.8letter,"FifthLetter"),
  entropy(lexicon.8letter,"SixthLetter"),
  entropy(lexicon.8letter,"SeventhLetter"),
  entropy(lexicon.8letter,"EighthLetter"))

entropy.allLetter <- as.vector(cbind(entropy.3letter,
                                     entropy.4letter,
                                     entropy.5letter,
                                     entropy.6letter,
                                     entropy.7letter,
                                     entropy.8letter))


# frequency-weighted entropy
entropyFW.3letter <- cbind(
  entropyFW(lexicon.3letter,"FirstLetter"),
  entropyFW(lexicon.3letter,"SecondLetter"),
  entropyFW(lexicon.3letter,"ThirdLetter"))

entropyFW.4letter <- cbind(
  entropyFW(lexicon.4letter,"FirstLetter"),
  entropyFW(lexicon.4letter,"SecondLetter"),
  entropyFW(lexicon.4letter,"ThirdLetter"),
  entropyFW(lexicon.4letter,"FourthLetter"))

entropyFW.5letter <- cbind(
  entropyFW(lexicon.5letter,"FirstLetter"),
  entropyFW(lexicon.5letter,"SecondLetter"),
  entropyFW(lexicon.5letter,"ThirdLetter"),
  entropyFW(lexicon.5letter,"FourthLetter"),
  entropyFW(lexicon.5letter,"FifthLetter"))

entropyFW.6letter <- cbind(
  entropyFW(lexicon.6letter,"FirstLetter"),
  entropyFW(lexicon.6letter,"SecondLetter"),
  entropyFW(lexicon.6letter,"ThirdLetter"),
  entropyFW(lexicon.6letter,"FourthLetter"),
  entropyFW(lexicon.6letter,"FifthLetter"),
  entropyFW(lexicon.6letter,"SixthLetter"))

entropyFW.7letter <- cbind(
  entropyFW(lexicon.7letter,"FirstLetter"),
  entropyFW(lexicon.7letter,"SecondLetter"),
  entropyFW(lexicon.7letter,"ThirdLetter"),
  entropyFW(lexicon.7letter,"FourthLetter"),
  entropyFW(lexicon.7letter,"FifthLetter"),
  entropyFW(lexicon.7letter,"SixthLetter"),
  entropyFW(lexicon.7letter,"SeventhLetter"))

entropyFW.8letter <- cbind(
  entropyFW(lexicon.8letter,"FirstLetter"),
  entropyFW(lexicon.8letter,"SecondLetter"),
  entropyFW(lexicon.8letter,"ThirdLetter"),
  entropyFW(lexicon.8letter,"FourthLetter"),
  entropyFW(lexicon.8letter,"FifthLetter"),
  entropyFW(lexicon.8letter,"SixthLetter"),
  entropyFW(lexicon.8letter,"SeventhLetter"),
  entropyFW(lexicon.8letter,"EighthLetter"))

entropyFW.allLetter <- as.vector(cbind(entropyFW.3letter,
                                     entropyFW.4letter,
                                     entropyFW.5letter,
                                     entropyFW.6letter,
                                     entropyFW.7letter,
                                     entropyFW.8letter))

# For figure visualizing entropy
probs.3letter <- as.data.frame(cbind(alphabet,probByLet(lexicon.3letter,"ThirdLetter")))
probs.3letter <- cbind(probs.3letter, probByLet(lexicon.3letter,"SecondLetter"))
probs.3letter <- cbind(probs.3letter, probByLet(lexicon.3letter,"FirstLetter"))
colnames(probs.3letter) <- c("Letter",
                             paste("3rd | Entropy:", entropy(lexicon.3letter, "ThirdLetter")),
                             paste("2nd | Entropy:", entropy(lexicon.3letter, "SecondLetter")),
                             paste("1st | Entropy:", entropy(lexicon.3letter, "FirstLetter")))
probs.3letter <- melt(probs.3letter, id.vars = "Letter")
colnames(probs.3letter) <- c("Letter","Position","Probability")


probs.4letter <- as.data.frame(cbind(alphabet,probByLet(lexicon.4letter,"FourthLetter")))
probs.4letter <- cbind(probs.4letter, probByLet(lexicon.4letter,"ThirdLetter"))
probs.4letter <- cbind(probs.4letter, probByLet(lexicon.4letter,"SecondLetter"))
probs.4letter <- cbind(probs.4letter, probByLet(lexicon.4letter,"FirstLetter"))
colnames(probs.4letter) <- c("Letter",
                             paste("4th | Entropy:", entropy(lexicon.4letter, "FourthLetter")),
                             paste("3rd | Entropy:", entropy(lexicon.4letter, "ThirdLetter")),
                             paste("2nd | Entropy:", entropy(lexicon.4letter, "SecondLetter")),
                             paste("1st | Entropy:", entropy(lexicon.4letter, "FirstLetter")))
probs.4letter <- melt(probs.4letter, id.vars = "Letter")
colnames(probs.4letter) <- c("Letter","Position","Probability")

probs.5letter <- as.data.frame(cbind(alphabet,probByLet(lexicon.5letter,"FifthLetter")))
probs.5letter <- cbind(probs.5letter, probByLet(lexicon.5letter,"FourthLetter"))
probs.5letter <- cbind(probs.5letter, probByLet(lexicon.5letter,"ThirdLetter"))
probs.5letter <- cbind(probs.5letter, probByLet(lexicon.5letter,"SecondLetter"))
probs.5letter <- cbind(probs.5letter, probByLet(lexicon.5letter,"FirstLetter"))
colnames(probs.5letter) <- c("Letter",
                             paste("5th | Entropy:", entropy(lexicon.5letter, "FifthLetter")),
                             paste("4th | Entropy:", entropy(lexicon.5letter, "FourthLetter")),
                             paste("3rd | Entropy:", entropy(lexicon.5letter, "ThirdLetter")),
                             paste("2nd | Entropy:", entropy(lexicon.5letter, "SecondLetter")),
                             paste("1st | Entropy:", entropy(lexicon.5letter, "FirstLetter")))
probs.5letter <- melt(probs.5letter, id.vars = "Letter")
colnames(probs.5letter) <- c("Letter","Position","Probability")


probs.6letter <- as.data.frame(cbind(alphabet,probByLet(lexicon.6letter,"SixthLetter")))
probs.6letter <- cbind(probs.6letter, probByLet(lexicon.6letter,"FifthLetter"))
probs.6letter <- cbind(probs.6letter, probByLet(lexicon.6letter,"FourthLetter"))
probs.6letter <- cbind(probs.6letter, probByLet(lexicon.6letter,"ThirdLetter"))
probs.6letter <- cbind(probs.6letter, probByLet(lexicon.6letter,"SecondLetter"))
probs.6letter <- cbind(probs.6letter, probByLet(lexicon.6letter,"FirstLetter"))
colnames(probs.6letter) <- c("Letter",
                             paste("6th | Entropy:", entropy(lexicon.6letter, "SixthLetter")),
                             paste("5th | Entropy:", entropy(lexicon.6letter, "FifthLetter")),
                             paste("4th | Entropy:", entropy(lexicon.6letter, "FourthLetter")),
                             paste("3rd | Entropy:", entropy(lexicon.6letter, "ThirdLetter")),
                             paste("2nd | Entropy:", entropy(lexicon.6letter, "SecondLetter")),
                             paste("1st | Entropy:", entropy(lexicon.6letter, "FirstLetter")))
probs.6letter <- melt(probs.6letter, id.vars = "Letter")
colnames(probs.6letter) <- c("Letter","Position","Probability")

probs.7letter <- as.data.frame(cbind(alphabet,probByLet(lexicon.7letter,"SeventhLetter")))
probs.7letter <- cbind(probs.7letter, probByLet(lexicon.7letter,"SixthLetter"))
probs.7letter <- cbind(probs.7letter, probByLet(lexicon.7letter,"FifthLetter"))
probs.7letter <- cbind(probs.7letter, probByLet(lexicon.7letter,"FourthLetter"))
probs.7letter <- cbind(probs.7letter, probByLet(lexicon.7letter,"ThirdLetter"))
probs.7letter <- cbind(probs.7letter, probByLet(lexicon.7letter,"SecondLetter"))
probs.7letter <- cbind(probs.7letter, probByLet(lexicon.7letter,"FirstLetter"))
colnames(probs.7letter) <- c("Letter",
                             paste("7th | Entropy:", entropy(lexicon.7letter, "SeventhLetter")),
                             paste("6th | Entropy:", entropy(lexicon.7letter, "SixthLetter")),
                             paste("5th | Entropy:", entropy(lexicon.7letter, "FifthLetter")),
                             paste("4th | Entropy:", entropy(lexicon.7letter, "FourthLetter")),
                             paste("3rd | Entropy:", entropy(lexicon.7letter, "ThirdLetter")),
                             paste("2nd | Entropy:", entropy(lexicon.7letter, "SecondLetter")),
                             paste("1st | Entropy:", entropy(lexicon.7letter, "FirstLetter")))
probs.7letter <- melt(probs.7letter, id.vars = "Letter")
colnames(probs.7letter) <- c("Letter","Position","Probability")

probs.7letter <- as.data.frame(cbind(alphabet,probByLet(lexicon.7letter,"SeventhLetter")))
probs.7letter <- cbind(probs.7letter, probByLet(lexicon.7letter,"SixthLetter"))
probs.7letter <- cbind(probs.7letter, probByLet(lexicon.7letter,"FifthLetter"))
probs.7letter <- cbind(probs.7letter, probByLet(lexicon.7letter,"FourthLetter"))
probs.7letter <- cbind(probs.7letter, probByLet(lexicon.7letter,"ThirdLetter"))
probs.7letter <- cbind(probs.7letter, probByLet(lexicon.7letter,"SecondLetter"))
probs.7letter <- cbind(probs.7letter, probByLet(lexicon.7letter,"FirstLetter"))
colnames(probs.7letter) <- c("Letter",
                             paste("7th | Entropy:", entropy(lexicon.7letter, "SeventhLetter")),
                             paste("6th | Entropy:", entropy(lexicon.7letter, "SixthLetter")),
                             paste("5th | Entropy:", entropy(lexicon.7letter, "FifthLetter")),
                             paste("4th | Entropy:", entropy(lexicon.7letter, "FourthLetter")),
                             paste("3rd | Entropy:", entropy(lexicon.7letter, "ThirdLetter")),
                             paste("2nd | Entropy:", entropy(lexicon.7letter, "SecondLetter")),
                             paste("1st | Entropy:", entropy(lexicon.7letter, "FirstLetter")))
probs.7letter <- melt(probs.7letter, id.vars = "Letter")
colnames(probs.7letter) <- c("Letter","Position","Probability")


probs.8letter <- as.data.frame(cbind(alphabet,probByLet(lexicon.8letter,"EighthLetter")))
probs.8letter <- cbind(probs.8letter, probByLet(lexicon.8letter,"SeventhLetter"))
probs.8letter <- cbind(probs.8letter, probByLet(lexicon.8letter,"SixthLetter"))
probs.8letter <- cbind(probs.8letter, probByLet(lexicon.8letter,"FifthLetter"))
probs.8letter <- cbind(probs.8letter, probByLet(lexicon.8letter,"FourthLetter"))
probs.8letter <- cbind(probs.8letter, probByLet(lexicon.8letter,"ThirdLetter"))
probs.8letter <- cbind(probs.8letter, probByLet(lexicon.8letter,"SecondLetter"))
probs.8letter <- cbind(probs.8letter, probByLet(lexicon.8letter,"FirstLetter"))
colnames(probs.8letter) <- c("Letter",
                             paste("8th | Entropy:", entropy(lexicon.8letter, "EighthLetter")),
                             paste("7th | Entropy:", entropy(lexicon.8letter, "SeventhLetter")),
                             paste("6th | Entropy:", entropy(lexicon.8letter, "SixthLetter")),
                             paste("5th | Entropy:", entropy(lexicon.8letter, "FifthLetter")),
                             paste("4th | Entropy:", entropy(lexicon.8letter, "FourthLetter")),
                             paste("3rd | Entropy:", entropy(lexicon.8letter, "ThirdLetter")),
                             paste("2nd | Entropy:", entropy(lexicon.8letter, "SecondLetter")),
                             paste("1st | Entropy:", entropy(lexicon.8letter, "FirstLetter")))
probs.8letter <- melt(probs.8letter, id.vars = "Letter")
colnames(probs.8letter) <- c("Letter","Position","Probability")

# Plots ----

ggplot(subset(probs.3letter, Position == "1st | Entropy: 4.4"), aes(Letter, Probability)) + geom_bar(stat="identity") + labs(y="Probability", x = "Letter") + coord_cartesian(ylim = c(0, .25)) + draw_label(label = paste("3-letter words, 1st letter \n Entropy: ", entropy(lexicon.3letter, "FirstLetter")), x = 13, y = .22, size = 36)
ggplot(subset(probs.3letter, Position == "2nd | Entropy: 3.36"), aes(Letter, Probability)) + geom_bar(stat="identity") + labs(y="Probability", x = "Letter") + coord_cartesian(ylim = c(0, .25)) + draw_label(label = paste("3-letter words, 2nd letter \n Entropy: ", entropy(lexicon.3letter, "SecondLetter")), x = 13, y = .22, size = 36)
ggplot(subset(probs.3letter, Position == "3rd | Entropy: 4.15"), aes(Letter, Probability)) + geom_bar(stat="identity") + labs(y="Probability", x = "Letter") + coord_cartesian(ylim = c(0, .25)) + draw_label(label = paste("3-letter words, 3rd letter \n Entropy: ", entropy(lexicon.3letter, "ThirdLetter")), x = 13, y = .22, size = 36)

ggplot(probs.3letter) + geom_tile(aes(Letter, Position, fill = Probability), color = "white") + scale_fill_continuous(low="white",high="black") + theme(legend.position = "bottom", legend.text = element_text(angle=90, hjust=0, vjust=0, size=20))  + labs(title="3-letter words")
ggplot(probs.4letter) + geom_tile(aes(Letter, Position, fill = Probability), color = "white") + scale_fill_continuous(low="white",high="black") + theme(legend.position = "bottom", legend.text = element_text(angle=90, hjust=0, vjust=0, size=20))  + labs(title="4-letter words")
ggplot(probs.5letter) + geom_tile(aes(Letter, Position, fill = Probability), color = "white") + scale_fill_continuous(low="white",high="black") + theme(legend.position = "bottom", legend.text = element_text(angle=90, hjust=0, vjust=0, size=20))  + labs(title="5-letter words")
ggplot(probs.6letter) + geom_tile(aes(Letter, Position, fill = Probability), color = "white") + scale_fill_continuous(low="white",high="black") + theme(legend.position = "bottom", legend.text = element_text(angle=90, hjust=0, vjust=0, size=20))  + labs(title="6-letter words")
ggplot(probs.7letter) + geom_tile(aes(Letter, Position, fill = Probability), color = "white") + scale_fill_continuous(low="white",high="black") + theme(legend.position = "bottom", legend.text = element_text(angle=90, hjust=0, vjust=0, size=20))  + labs(title="7-letter words")
ggplot(probs.8letter) + geom_tile(aes(Letter, Position, fill = Probability), color = "white") + scale_fill_continuous(low="white",high="black") + theme(legend.position = "bottom", legend.text = element_text(angle=90, hjust=0, vjust=0, size=20))  + labs(title="8-letter words")

metrics <- cbind.data.frame(entropy.allLetter, entropyFW.allLetter, enemies.allLetter)
metrics$Labels <- c("3:1","3:2","3:3",
                    "4:1","4:2","4:3","4:4",
                    "5:1","5:2","5:3","5:4","5:5",
                    "6:1","6:2","6:3","6:4","6:5","6:6",
                    "7:1","7:2","7:3","7:4","7:5","7:6","7:7",
                    "8:1","8:2","8:3","8:4","8:5","8:6","8:7","8:8")
metrics$Colors <- c("darkred","darkred","darkred",
                    "darkorange","darkorange","darkorange","darkorange",
                    "goldenrod","goldenrod","goldenrod","goldenrod","goldenrod",
                    "darkgreen","darkgreen","darkgreen","darkgreen","darkgreen","darkgreen",
                    "dodgerblue","dodgerblue","dodgerblue","dodgerblue","dodgerblue","dodgerblue","dodgerblue",
                    "darkblue","darkblue","darkblue","darkblue","darkblue","darkblue","darkblue","darkblue")

# Save workspace post-setup -----------------------------------------------
write.csv(lexicon, "lexicon_allMetrics.csv")
save.image("~/friends-and-enemies/PostScript01.RData")  
