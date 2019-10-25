import numpy as np
import tensorflow as tf
import os, io, gc
import matplotlib.pyplot as plt
import _pickle as pickle
import argparse
from scipy.stats.stats import pearsonr

class Result_Analyzer:
    def __init__(self, extract_Dir_Name):
        self.marker_List = ["o", "v", "^", "<", ">", "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"]

        self.extract_Dir_Name = extract_Dir_Name + "/Result/"
        self.Loading_Results()

        self.Trained_Count_Dict()
        self.Batch_Hidden_Data_Dict_Generate()
        self.Batch_Data_Dict_Generate(batch_Size=10)

        self.Single_Phoneme_CS_Dict_Generate()
        self.Extract_Single_Phoneme_CS_Txt()

        self.Correlation_Generate()

    def Loading_Results(self):
        with open(self.extract_Dir_Name + "Metadata.pickle", "rb") as f:
            metadata_Dict = pickle.load(f)

        self.orthography_Size = metadata_Dict["Orthography_Size"]
        self.hidden_Size = metadata_Dict["Hidden_Size"]
        self.phonology_Size = metadata_Dict["Phonology_Size"]
        self.is_Phoneme_Distribution = metadata_Dict["Is_Phoneme_Distribution"]
                        
        self.target_Pattern = metadata_Dict["Target_Pattern"]
        self.phoneme_Pattern = metadata_Dict["Phoneme_Pattern"]

        self.word_Index_Dict = metadata_Dict["Word_Index_Dict"]
        self.letter_Index_Dict = metadata_Dict["Letter_Index_Dict"]
        self.phoneme_Index_Dict = metadata_Dict["Phoneme_Index_Dict"]
        self.index_Phoneme_Dict = {index: key for key, index in self.phoneme_Index_Dict.items() if len(key) == 1}
            
        self.pronunciation_Dict = metadata_Dict["Pronunciation_Dict"]
        self.frequency_Dict = metadata_Dict["Frequency_Dict"]
        self.human_RT_Dict = metadata_Dict["Human_RT_Dict"]

        self.max_Cycle = self.target_Pattern.shape[1] // self.phonology_Size

        self.result_Dict = {}
        self.hidden_Dict = {}
        self.trained_Pattern_Count_Dict = {}
        result_File_List = [x for x in os.listdir(self.extract_Dir_Name) if x.endswith(".pickle")]                
        result_File_List.remove('Metadata.pickle')
        for result_File in result_File_List:
            with open(self.extract_Dir_Name + result_File, "rb") as f:
                result_Dict = pickle.load(f)
                self.result_Dict[result_Dict["Epoch"]] = result_Dict["Result"]
                self.hidden_Dict[result_Dict["Epoch"]] = result_Dict["Hidden"]
                self.trained_Pattern_Count_Dict[result_Dict["Epoch"]] = result_Dict["Trained_Pattern_Count_Dict"]
                
    def Trained_Count_Dict(self):
        index_Word_Dict = {value: key for key, value in self.word_Index_Dict.items()}

        self.trained_Count_Dict = {}
        for epoch, trained_Pattern_Count_Dict in self.trained_Pattern_Count_Dict.items():
            for index, value in trained_Pattern_Count_Dict.items():
                self.trained_Count_Dict[epoch, index_Word_Dict[index]] = value

    def Batch_Data_Dict_Generate(self, batch_Size = 20):
        tf_Session = tf.Session()
        target_Tensor = tf.Variable(self.target_Pattern, dtype=tf.float32)  #39313, 306
        phoneme_Tensor = tf.Variable(self.phoneme_Pattern, dtype=tf.float32)  #42, 18
        tf_Session.run(tf.global_variables_initializer())

        result_Tensor = tf.placeholder(tf.float32, shape=[None, self.target_Pattern.shape[1]])  #batch, 306
        
        tiled_Result_Tensor = tf.tile(tf.expand_dims(result_Tensor, [1]), multiples = [1, target_Tensor.shape[0], 1])   #[batch, 39313, 306]
        tiled_Target_Tensor = tf.tile(tf.expand_dims(target_Tensor, [0]), multiples = [tf.shape(result_Tensor)[0], 1, 1])   #[batch, 39313, 306]        
        cosine_Similarity = tf.reduce_sum(tiled_Target_Tensor * tiled_Result_Tensor, axis = 2) / (tf.sqrt(tf.reduce_sum(tf.pow(tiled_Target_Tensor, 2), axis = 2)) * tf.sqrt(tf.reduce_sum(tf.pow(tiled_Result_Tensor, 2), axis = 2)))  #[batch, 39313]        
        mean_Squared_Error = tf.reduce_mean(tf.pow(tiled_Target_Tensor - tiled_Result_Tensor, 2), axis=2)  #[batch, 39313]
        euclidean_Distance = tf.sqrt(tf.reduce_sum(tf.pow(tiled_Target_Tensor - tiled_Result_Tensor, 2), axis=2))  #[batch, 39313]
        cross_Entropy = -tf.reduce_mean(tiled_Target_Tensor * tf.log(tiled_Result_Tensor + 1e-8) + (1 - tiled_Target_Tensor) * tf.log(1 - tiled_Result_Tensor + 1e-8), axis = 2)  #[batch, 39313]

        reshaped_Result_Tensor = tf.reshape(result_Tensor, shape=[-1, self.max_Cycle, self.phonology_Size]) #batch, 17, 18
        if self.is_Phoneme_Distribution:
            tiled_Reshaped_Result_Tensor = tf.tile(tf.expand_dims(reshaped_Result_Tensor, [2]), multiples = [1, 1, phoneme_Tensor.shape[0], 1])   #[batch, 17, 42, 18]                
            tiled_Phoneme_Tensor = tf.tile(tf.expand_dims(tf.expand_dims(phoneme_Tensor, [0]), [0]), multiples = [tf.shape(reshaped_Result_Tensor)[0], tf.shape(reshaped_Result_Tensor)[1], 1, 1])   #[batch, 17, 42, 18]                
            phoneme_Cosine_Similarity = tf.reduce_sum(tiled_Phoneme_Tensor * tiled_Reshaped_Result_Tensor, axis = 3) / (tf.sqrt(tf.reduce_sum(tf.pow(tiled_Phoneme_Tensor, 2), axis = 3)) * tf.sqrt(tf.reduce_sum(tf.pow(tiled_Reshaped_Result_Tensor, 2), axis = 3)))  #[batch, 17, 42]        
            pattern_Argmax = tf.argmax(phoneme_Cosine_Similarity, axis=2)  #[batch, 17]
        else:
            pattern_Argmax = tf.argmax(reshaped_Result_Tensor, axis=2)  #[batch, 17]
        self.data_Dict = {}
        for epoch, result_Array in self.result_Dict.items():
            print("Data dict making of epoch {}...".format(epoch))
            word_Index_List = list(self.word_Index_Dict.items())
            for batch_Index in range(0, len(word_Index_List), batch_Size):
                word_List, index_List = zip(*word_Index_List[batch_Index:batch_Index+batch_Size])                
                cs_Array, mse_Array, ed_Array, ce_Array, phoneme_Argmax_Array = tf_Session.run(
                    fetches = [
                        cosine_Similarity, 
                        mean_Squared_Error, 
                        euclidean_Distance, 
                        cross_Entropy, 
                        pattern_Argmax
                        ],
                    feed_dict={result_Tensor: result_Array[list(index_List)]}
                    )
               
                for word, index in zip(word_List, range(len(index_List))):
                    self.data_Dict[epoch, word] = {}
                    self.data_Dict[epoch, word]["Cosine_Similarity"] = cs_Array[index][self.word_Index_Dict[word]]
                    self.data_Dict[epoch, word]["Mean_Squared_Error"] = mse_Array[index][self.word_Index_Dict[word]]
                    self.data_Dict[epoch, word]["Euclidean_Distance"] = ed_Array[index][self.word_Index_Dict[word]]
                    self.data_Dict[epoch, word]["Cross_Entropy"] = ce_Array[index][self.word_Index_Dict[word]]

                    self.data_Dict[epoch, word]["Exported_Pronunciation"] = self.Pronunciation_Generate(phoneme_Argmax_Array[index])
                    self.data_Dict[epoch, word]["Accuracy_Max_CS"] = np.argmax(cs_Array[index]) == self.word_Index_Dict[word]
                    self.data_Dict[epoch, word]["Accuracy_Min_MSE"] = np.argmin(mse_Array[index]) == self.word_Index_Dict[word]
                    self.data_Dict[epoch, word]["Accuracy_Min_ED"] = np.argmin(ed_Array[index]) == self.word_Index_Dict[word]
                    self.data_Dict[epoch, word]["Accuracy_Min_CE"] = np.argmin(ce_Array[index]) == self.word_Index_Dict[word]
                    self.data_Dict[epoch, word]["Accuracy_Pronunciation"] = self.data_Dict[epoch, word]["Exported_Pronunciation"] == self.pronunciation_Dict[word]                        

    def Batch_Hidden_Data_Dict_Generate(self, batch_Size = 20):
        tf_Session = tf.Session()
        
        hidden_Tensor = tf.placeholder(tf.float32, shape=[None, self.max_Cycle, self.hidden_Size])  #[batch, 17, 500]
        hidden_Compare_Tensor = tf.concat([tf.zeros((tf.shape(hidden_Tensor)[0], 1, self.hidden_Size)), hidden_Tensor], axis=1)[:,:-1,:]    #[batch, 17, 500]

        #Hidden activation function was 'tanh()', so actiation need rescale.  (-1.0 to 1.0) -> (0.0 to 1.0)
        rescaled_Hidden_Tensor = (hidden_Tensor + 1.0) / 2.0
        rescaled_Hidden_Compare_Tensor = (hidden_Compare_Tensor + 1.0) / 2.0

        cosine_Similarity = tf.reduce_sum(tf.reduce_sum(rescaled_Hidden_Tensor * rescaled_Hidden_Compare_Tensor, axis = 2) / (tf.sqrt(tf.reduce_sum(tf.pow(rescaled_Hidden_Tensor, 2), axis = 2)) * tf.sqrt(tf.reduce_sum(tf.pow(rescaled_Hidden_Compare_Tensor, 2), axis = 2))), axis=1)  #[batch]
        mean_Squared_Error = tf.reduce_sum(tf.reduce_mean(tf.pow(rescaled_Hidden_Tensor - rescaled_Hidden_Compare_Tensor, 2), axis=2), axis=1)  #[batch]
        euclidean_Distance = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.pow(rescaled_Hidden_Tensor - rescaled_Hidden_Compare_Tensor, 2), axis=2)), axis=1)  #[batch, 39313]
        cross_Entropy = tf.reduce_sum(-tf.reduce_mean(rescaled_Hidden_Tensor * tf.log(rescaled_Hidden_Compare_Tensor + 1e-8) + (1 - rescaled_Hidden_Tensor) * tf.log(1 - rescaled_Hidden_Compare_Tensor + 1e-8), axis = 2), axis=1)  #[batch, 39313]
        
        self.hidden_Data_Dict = {}
        for epoch, hidden_Array in self.hidden_Dict.items():
            print("Hidden Data dict making of epoch {}...".format(epoch))
            word_Index_List = list(self.word_Index_Dict.items())
            for batch_Index in range(0, len(word_Index_List), batch_Size):
                word_List, index_List = zip(*word_Index_List[batch_Index:batch_Index+batch_Size])                
                hidden_CS, hidden_MSE, hidden_ED, hidden_CE = tf_Session.run(
                    fetches = [
                        cosine_Similarity, 
                        mean_Squared_Error, 
                        euclidean_Distance, 
                        cross_Entropy
                        ],
                    feed_dict={hidden_Tensor: hidden_Array[list(index_List)]}
                    )
               
                for word, index in zip(word_List, range(len(index_List))):
                    self.hidden_Data_Dict[epoch, word] = {}                                        
                    self.hidden_Data_Dict[epoch, word]["Cosine_Similarity"] = hidden_CS[index]
                    self.hidden_Data_Dict[epoch, word]["Mean_Squared_Error"] = hidden_MSE[index]
                    self.hidden_Data_Dict[epoch, word]["Euclidean_Distance"] = hidden_ED[index]
                    self.hidden_Data_Dict[epoch, word]["Cross_Entropy"] = hidden_CE[index]
        
    def Single_Phoneme_CS_Dict_Generate(self):
        tf_Session = tf.Session()

        target_Tensor = tf.placeholder(tf.float32, shape=[None, self.target_Pattern.shape[1]])  #batch, 234
        result_Tensor = tf.placeholder(tf.float32, shape=[None, self.target_Pattern.shape[1]])  #batch, 234
        
        reshaped_Target = tf.reshape(target_Tensor, [-1, self.target_Pattern.shape[1] // self.phonology_Size, self.phonology_Size])
        reshaped_Result = tf.reshape(result_Tensor, [-1, self.target_Pattern.shape[1] // self.phonology_Size, self.phonology_Size])

        cross_Entropy = -tf.reduce_mean(reshaped_Target * tf.log(reshaped_Result + 1e-8) + (1 - reshaped_Target) * tf.log(1 - reshaped_Result + 1e-8), axis=2)   #[batch, 13]
                
        self.single_Phoneme_CS_Dict = {}

        for epoch, result_Array in self.result_Dict.items():
            cs_Array = tf_Session.run(
                fetches = cross_Entropy,
                feed_dict={
                    target_Tensor: self.target_Pattern,
                    result_Tensor: result_Array
                    }
                )
               
            for word, pattern_Index in self.word_Index_Dict.items():
                for phoneme_Index, phoneme in enumerate(self.pronunciation_Dict[word]):
                    self.single_Phoneme_CS_Dict[epoch, word, phoneme_Index] = cs_Array[pattern_Index, phoneme_Index]

    def Pronunciation_Generate(self, phoneme_Argmax_Array):
        pronunciation = "".join([self.index_Phoneme_Dict[phoneme_Index] for phoneme_Index in phoneme_Argmax_Array])
        while len(pronunciation) > 0 and pronunciation[-1] == "_":
            pronunciation = pronunciation[:-1]
        return pronunciation
        
    def Extract_Result_Txt(self):
        extract_Result_List = ["\t".join([
            "Epoch",
            "Ortho",
            "Phono",
            "Length",            
            "LogFreq",
            "MeanRT",
            "Trained_Count",
            "Cosine_Similarity",
            "Mean_Squared_Error",
            "Euclidean_Distance",
            "Cross_Entropy",
            "Exported_Pronunciation",
            "Accuracy_Max_CS",
            "Accuracy_Min_MSE",
            "Accuracy_Min_ED",
            "Accuracy_Min_CE",
            "Accuracy_Pronunciation",
            "Hidden_Cosine_Similarity",
            "Hidden_Mean_Squared_Error",
            "Hidden_Euclidean_Distance",
            "Hidden_Cross_Entropy",
            ])]
        for epoch in sorted(self.result_Dict.keys()):
            for word in sorted(self.word_Index_Dict.keys()):
                line_List = []
                line_List.append(str(epoch))
                line_List.append(word)
                line_List.append(self.pronunciation_Dict[word])
                line_List.append(str(len(word)))
                line_List.append(str(self.frequency_Dict[word]))
                line_List.append(str(self.human_RT_Dict[word]))
                line_List.append(str(self.trained_Count_Dict[epoch, word]))
                line_List.append(str(self.data_Dict[epoch, word]["Cosine_Similarity"]))
                line_List.append(str(self.data_Dict[epoch, word]["Mean_Squared_Error"]))
                line_List.append(str(self.data_Dict[epoch, word]["Euclidean_Distance"]))
                line_List.append(str(self.data_Dict[epoch, word]["Cross_Entropy"]))
                line_List.append(str(self.data_Dict[epoch, word]["Exported_Pronunciation"]))
                line_List.append(str(self.data_Dict[epoch, word]["Accuracy_Max_CS"]))
                line_List.append(str(self.data_Dict[epoch, word]["Accuracy_Min_MSE"]))
                line_List.append(str(self.data_Dict[epoch, word]["Accuracy_Min_ED"]))
                line_List.append(str(self.data_Dict[epoch, word]["Accuracy_Min_CE"]))
                line_List.append(str(self.data_Dict[epoch, word]["Accuracy_Pronunciation"]))
                line_List.append(str(self.hidden_Data_Dict[epoch, word]["Cosine_Similarity"]))
                line_List.append(str(self.hidden_Data_Dict[epoch, word]["Mean_Squared_Error"]))
                line_List.append(str(self.hidden_Data_Dict[epoch, word]["Euclidean_Distance"]))
                line_List.append(str(self.hidden_Data_Dict[epoch, word]["Cross_Entropy"]))

                extract_Result_List.append("\t".join(line_List))

        with open(self.extract_Dir_Name + "/Result.txt", "w") as f:
            f.write("\n".join(extract_Result_List))


    def Extract_Single_Phoneme_CS_Txt(self):
        extract_Result_List = ["\t".join([
            "Epoch",
            "Ortho",
            "Phono",
            "Ort_Length",
            "Pho_Length",
            "Phoneme_Index",
            "Accuracy",
            "Cross_Entropy",
            ])]

        for (epoch, word, phoneme_Index), cross_Entropy in self.single_Phoneme_CS_Dict.items():
            line_List = []
            line_List.append(str(epoch))
            line_List.append(word)
            line_List.append(self.pronunciation_Dict[word])
            line_List.append(str(len(word)))
            line_List.append(str(len(self.pronunciation_Dict[word])))
            line_List.append(str(phoneme_Index + 1))
            line_List.append(str(self.data_Dict[epoch, word]["Accuracy_Pronunciation"]))
            line_List.append(str(cross_Entropy))
            extract_Result_List.append("\t".join(line_List))

        with open(self.extract_Dir_Name + "/Single_Phoneme_CS.txt", "w") as f:
            f.write("\n".join(extract_Result_List))


    def Extract_All_CS_Txt(self):
        word_List = list(self.word_Index_Dict.keys())

        extract_All_CS_List = ["\t".join(["Epoch", "Inserted_Word"] +  word_List)]

        for (epoch, inserted_Word), data in self.data_Dict.items():
            line_List = []
            line_List.append(str(epoch))
            line_List.append(inserted_Word)
            line_List.append(compare_Word)
            line_List.extend([str(data["Cosine_Similarity_Array"][self.word_Index_Dict[word]]) for word in word_List])

            extract_All_CS_List.append("\t".join(line_List))
                                    
        with open(self.extract_Dir_Name + "/All_CS.txt", "w") as f:
            f.write("\n".join(extract_All_CS_List))        

    
    def Correlation_Generate(self, accuracy_Filter = True):
        tuple_List_Dict = {}
        for epoch in self.result_Dict.keys():
            for data_Type in ["Hidden", "Result"]:
                for acc_Type in ["CS", "MSE", "ED", "CE"]:
                    tuple_List_Dict[epoch, data_Type, acc_Type] = []
        
        for epoch, word in self.data_Dict.keys():
            if accuracy_Filter and not (self.data_Dict[epoch, word]["Accuracy_Max_CS"] or self.data_Dict[epoch, word]["Accuracy_Min_MSE"] or self.data_Dict[epoch, word]["Accuracy_Min_ED"] or self.data_Dict[epoch, word]["Accuracy_Min_CE"] or self.data_Dict[epoch, word]["Accuracy_Pronunciation"]):
                continue

            tuple_List_Dict[epoch, "Hidden", "CS"].append((self.hidden_Data_Dict[epoch, word]["Cosine_Similarity"], self.human_RT_Dict[word]))
            tuple_List_Dict[epoch, "Hidden", "MSE"].append((self.hidden_Data_Dict[epoch, word]["Mean_Squared_Error"], self.human_RT_Dict[word]))
            tuple_List_Dict[epoch, "Hidden", "ED"].append((self.hidden_Data_Dict[epoch, word]["Euclidean_Distance"], self.human_RT_Dict[word]))
            tuple_List_Dict[epoch, "Hidden", "CE"].append((self.hidden_Data_Dict[epoch, word]["Cross_Entropy"], self.human_RT_Dict[word]))
            
            tuple_List_Dict[epoch, "Result", "CS"].append((self.data_Dict[epoch, word]["Cosine_Similarity"], self.human_RT_Dict[word]))
            tuple_List_Dict[epoch, "Result", "MSE"].append((self.data_Dict[epoch, word]["Mean_Squared_Error"], self.human_RT_Dict[word]))
            tuple_List_Dict[epoch, "Result", "ED"].append((self.data_Dict[epoch, word]["Euclidean_Distance"], self.human_RT_Dict[word]))
            tuple_List_Dict[epoch, "Result", "CE"].append((self.data_Dict[epoch, word]["Cross_Entropy"], self.human_RT_Dict[word]))

        self.correlation_Dict = {}
        for (epoch, data_Type, acc_Type), tuple_List in tuple_List_Dict.items():
            if len(tuple_List) > 0:
                x, y = np.split(np.array(tuple_List), 2, axis=1)                
                x = x.ravel()
                y = y.ravel()
                self.correlation_Dict[epoch, data_Type, acc_Type] = pearsonr(x,y)
            else:
                self.correlation_Dict[epoch, data_Type, acc_Type] = (np.nan, np.nan)

    def Print_Accuracy(self, file_Export = False):
        accuracy_Dict  = {}
        for epoch in sorted(self.result_Dict.keys()):
            max_CS_Count = 0
            min_MSE_Count = 0
            min_ED_Count = 0
            min_CE_Count = 0
            pronunciation_Count = 0
            for word in self.word_Index_Dict.keys():
                max_CS_Count += self.data_Dict[epoch, word]["Accuracy_Max_CS"]
                min_MSE_Count += self.data_Dict[epoch, word]["Accuracy_Min_MSE"]
                min_ED_Count += self.data_Dict[epoch, word]["Accuracy_Min_ED"]
                min_CE_Count += self.data_Dict[epoch, word]["Accuracy_Min_CE"]
                pronunciation_Count += self.data_Dict[epoch, word]["Accuracy_Pronunciation"]

            accuracy_Dict[epoch, "Max_CS"] = max_CS_Count / len(self.word_Index_Dict)
            accuracy_Dict[epoch, "Min_MSE"] = min_MSE_Count / len(self.word_Index_Dict)
            accuracy_Dict[epoch, "Min_ED"] = min_ED_Count / len(self.word_Index_Dict)
            accuracy_Dict[epoch, "Min_CE"] = min_CE_Count / len(self.word_Index_Dict)
            accuracy_Dict[epoch, "Pronunciation"] = pronunciation_Count / len(self.word_Index_Dict)


        export_Data = ["Epoch\tACC_Max_CS\tACC_Min_MSE\tACC_Min_ED\tACC_Min_CE\tACC_Pronunciation\tCorrelation_CS\tCorrelation_MSE\tCorrelation_ED\tCorrelation_CE\tCorrelation_Hidden_CS\tCorrelation_Hidden_MSE\tCorrelation_Hidden_ED\tCorrelation_Hidden_CE"]
        for epoch in sorted(self.result_Dict.keys()):
            new_Line = []
            new_Line.append(str(epoch))
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Max_CS"] * 100))
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Min_MSE"] * 100))
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Min_ED"] * 100))
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Min_CE"] * 100))
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Pronunciation"] * 100))

            new_Line.append("{0:.3f} (p={1:.3f})".format(self.correlation_Dict[epoch, "Result", "CS"][0], self.correlation_Dict[epoch, "Result", "CS"][1]))
            new_Line.append("{0:.3f} (p={1:.3f})".format(self.correlation_Dict[epoch, "Result", "MSE"][0], self.correlation_Dict[epoch, "Result", "MSE"][1]))
            new_Line.append("{0:.3f} (p={1:.3f})".format(self.correlation_Dict[epoch, "Result", "ED"][0], self.correlation_Dict[epoch, "Result", "ED"][1]))
            new_Line.append("{0:.3f} (p={1:.3f})".format(self.correlation_Dict[epoch, "Result", "CE"][0], self.correlation_Dict[epoch, "Result", "CE"][1]))

            new_Line.append("{0:.3f} (p={1:.3f})".format(self.correlation_Dict[epoch, "Hidden", "CS"][0], self.correlation_Dict[epoch, "Hidden", "CS"][1]))
            new_Line.append("{0:.3f} (p={1:.3f})".format(self.correlation_Dict[epoch, "Hidden", "MSE"][0], self.correlation_Dict[epoch, "Hidden", "MSE"][1]))
            new_Line.append("{0:.3f} (p={1:.3f})".format(self.correlation_Dict[epoch, "Hidden", "ED"][0], self.correlation_Dict[epoch, "Hidden", "ED"][1]))
            new_Line.append("{0:.3f} (p={1:.3f})".format(self.correlation_Dict[epoch, "Hidden", "CE"][0], self.correlation_Dict[epoch, "Hidden", "CE"][1]))

            export_Data.append("\t".join(new_Line))

        print("\n".join(export_Data))

        if file_Export:
            with open(self.extract_Dir_Name + "Accuracy_Table.txt", "w") as f:
                f.write("\n".join(export_Data))

def Batch_Result_Analyze():
    folder_List = [x for x in list(set([x[1] for x in os.walk('./')][0]))]

    for folder in folder_List:        
        if not os.path.isfile(folder + "/Result/Metadata.pickle"):
            continue

        result_Analyzer = Result_Analyzer(extract_Dir_Name = folder)        
        result_Analyzer.Extract_Result_Txt()
        result_Analyzer.Print_Accuracy(file_Export=True)        

if __name__ == "__main__":    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-f", "--folder", required=False)
    argument_Dict = vars(argParser.parse_args())

    if argument_Dict["folder"] == None:        
        Batch_Result_Analyze()
    elif not os.path.isfile(argument_Dict["folder"] + "/Result/Metadata.pickle"):
        print("THERE IS NO RESULT FILE!")
    else:        
        result_Analyzer = Result_Analyzer(extract_Dir_Name = argument_Dict["folder"])
        result_Analyzer.Extract_Result_Txt()
        result_Analyzer.Print_Accuracy(file_Export=True)