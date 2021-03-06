# Friends-and-Enemies

This repository contains:
* R scripts for all analyses 
* .csv files of all analysis inputs/outputs 
* a folder with all the Python scripts used to run the VOISeR simulations
* a folder with scripts associated with VOISeR validation tests (testing for consistency/regularity effects)


# VOISeR model

VOISeR is a simple computational reading model to support the friends-and-enemies research.

## Requirement

tensorflow >= 1.13

If you are using TF 2.0, see the repository below:  
https://github.com/CODEJIN/VOISeR_TF20 

## Structure
![Structure](https://user-images.githubusercontent.com/17133841/66222115-70035a80-e69e-11e9-8a8e-0bc0ef4c71d7.png)

* The using of 'Orthography → Hidden' and 'Hidden → Hidden' is selectable.
* The model reported in the paper has both connections (O->H and H->H).

## Dataset

Data were obtained from 'The English Lexicon Project':

    Balota, D. A., Yap, M. J., Hutchison, K. A., Cortese, M. J., Kessler, B., Loftis, B., ... & Treiman, R. (2007). The English lexicon project. Behavior research methods, 39(3), 445-459.
    
The "ELP_groupData.csv" file was used to train VOISeR.

## Run

### Command
    python VOISeR.py [parameters]
    
### Parameters

* `-dir <path>`
    * Determines the type of hidden layer. You can enter either LSTM, GRU, SCRN, or BPTT.
    * This parameter is required.

* `-ht B|H|O`
    * Determines which layers' activation is used for the hidden activation calculation.
        * B: Using both of previous hidden and output
        * H: Using previous hidden
        * O: Using previous output
    * This parameter is required.
    
* `-hu <int>`
    * Determines the size of the hidden layer. You can enter a positive integer.
    * This parameter is required.

* `-lr <float>`
    * Determine the size of learning rate. You can enter a positive float.
    * This parameter is required.

* `-e <int>`
    * Determine the model's maximum training epoch.
    * This parameter is required.

* `-tt <int>`
    * Determine the frequency of the test during learning. You can enter a positive integer.
    * This parameter is required.
    
* `-fre`
    * If you enter this parameter, model will use the frequency information of words in the training.
    
* `-emb <int>`
    * If you enter this parameter with integer value, model use the embedding about the orthographic input.
    * The inserted integer value become the size of embedding.
    * The default value is None.    
    
* `-dstr <path>`
    * If you enter this parameter, the target pattern become the distributed pattern.
    * If you don't enter, the target pattern become one-hot structure.
    * Please see the example: 'phonetic_feature_definitions_18_features.csv'
    
* `-try <int>`
    * Attach an tag to each result directory.
    * This value does not affect the performance of the model.
    
## Analysis
    
### Command
    python Result_Analysis.py -f <path>
    
### Parameter

* `-f <path>`
    * Results directory to run the analysis on.
    * This parameter is required.
    * Ex. `VOISeR_Results/HT_B.HU_300.LR_0005.E_10000.TT_1000.DSTR_True`
