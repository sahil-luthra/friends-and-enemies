#1. Only feedback, only hidden
#2. Frequency applied
#3. 학습중 해당 패턴이 얼마나 학습되었는가를 표시해줄 것

import tensorflow as tf
import numpy as np
import _pickle as pickle
from tensorflow.contrib.seq2seq import BasicDecoder, TrainingHelper, dynamic_decode
from tensorflow.contrib.rnn import BasicRNNCell
from threading import Thread
import time, os, sys, argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import Counter
from Cells import *
from Pattern_Feeder import Pattern_Feeder
try: import ctypes
except: pass

class VOISeR:
    def __init__(
        self, 
        pattern_File,
        hidden_Calc_Type = "B",  #"H", "O"
        hidden_Size = 200, 
        learning_Rate = 0.001, 
        batch_Size = 3000, 
        load_Epoch = None, 
        max_Epoch = 1000, 
        use_Frequency = False, 
        orthography_Embedding_Size = None,
        phoneme_Feature_File = "phonetic_feature_definitions_18_features.csv",
        extract_Dir = "Temp"):
        try: ctypes.windll.kernel32.SetConsoleTitleW(extract_Dir.replace("_", ":").replace(".", " "))
        except: pass       

        self.tf_Session = tf.Session()
                
        self.hidden_Calc_Type = hidden_Calc_Type
        self.hidden_Size = hidden_Size
        self.learning_Rate = learning_Rate
        self.orthography_Embedding_Size = orthography_Embedding_Size
        self.is_Phoneme_Distribution = bool(phoneme_Feature_File)
        self.extract_Dir = extract_Dir

        self.pattern_Feeder = Pattern_Feeder(            
            pattern_File = pattern_File,
            batch_Size = batch_Size,
            start_Epoch = load_Epoch or 0,
            max_Epoch = max_Epoch,
            use_Frequency = use_Frequency,
            use_Orthography_Embedding = bool(orthography_Embedding_Size),
            phoneme_Feature_File = phoneme_Feature_File,
            )        

        self.Tensor_Generate()

        self.tf_Saver = tf.train.Saver(max_to_keep=0)

        if load_Epoch is not None:
            self.Restore(load_Epoch)

    def Tensor_Generate(self):        
        with tf.variable_scope('feedback_Model') as scope:
            orthography_Placeholder = self.pattern_Feeder.placeholder_Dict["Orthography"]
            phonology_Placeholder = self.pattern_Feeder.placeholder_Dict["Phonology"]
            length_Placeholder = self.pattern_Feeder.placeholder_Dict["Length"]
            analysis_Input_Placeholder = self.pattern_Feeder.placeholder_Dict["Analysis_Input"]
            phoneme_Info_Placeholder = self.pattern_Feeder.placeholder_Dict["Phoneme_Info"]

            batch_Size = tf.shape(orthography_Placeholder)[0]

            if self.orthography_Embedding_Size is not None:
                orthography_Embedding = tf.get_variable(
                    name = "orthography_Embedding",
                    shape = (self.pattern_Feeder.orthography_Size, self.orthography_Embedding_Size),
                    dtype = tf.float32,
                    initializer = tf.truncated_normal_initializer(stddev=0.5)
                )
                orthography_Pattern = tf.reshape(tf.nn.embedding_lookup(orthography_Embedding, orthography_Placeholder), shape=(-1, self.pattern_Feeder.orthography_Size * self.orthography_Embedding_Size))  #Shape: (batch_Size, orthography_Length * embedded_Pattern_Size)
            else:
                orthography_Pattern = orthography_Placeholder
            
            input_Pattern = tf.tile(
                input= tf.expand_dims(orthography_Pattern, axis=1), 
                multiples= [1, self.pattern_Feeder.max_Pronunciation_Length, 1]
                )

            #RNN
            if self.hidden_Calc_Type.upper() == "B".upper():
                rnnCell = FeedbackCell(
                    num_hidden_units = self.hidden_Size,
                    num_output_units = self.pattern_Feeder.phonology_Size,
                    use_bias = True,
                    output_state_activation= tf.nn.sigmoid if self.is_Phoneme_Distribution else tf.nn.softmax
                    )
            elif self.hidden_Calc_Type.upper() == "O".upper():
                rnnCell = FeedbackOnlyCell(
                    num_hidden_units = self.hidden_Size,
                    num_output_units = self.pattern_Feeder.phonology_Size,
                    use_bias = True,
                    output_state_activation= tf.nn.sigmoid if self.is_Phoneme_Distribution else tf.nn.softmax
                    )
            elif self.hidden_Calc_Type.upper() == "H".upper():
                rnnCell = BasicRNNCell(
                    num_units = self.hidden_Size
                    )
            else:
                assert False
            
            decoder_Initial_State = rnnCell.zero_state(batch_size=batch_Size, dtype = tf.float32)
            helper = TrainingHelper(
                inputs=input_Pattern,
                sequence_length = length_Placeholder,
                )
            decoder = BasicDecoder(
                cell=rnnCell, 
                helper=helper, 
                initial_state=decoder_Initial_State
                )

            if self.hidden_Calc_Type.upper() in ["B".upper(), "O".upper()]:
                hidden_and_Output_Logits, final_State, _ = dynamic_decode(decoder = decoder)
                hidden_Logits, output_Logits = tf.split(hidden_and_Output_Logits.rnn_output, [self.hidden_Size, self.pattern_Feeder.phonology_Size], axis=2)

            elif self.hidden_Calc_Type.upper() == "H".upper():
                hidden_Logits = dynamic_decode(decoder = decoder)[0].rnn_output
                output_Logits = tf.layers.dense(
                    inputs= hidden_Logits,
                    units= self.pattern_Feeder.phonology_Size,
                    use_bias = True
                    )

        with tf.variable_scope('training_Loss') as scope:                        
            if self.is_Phoneme_Distribution:
                loss_Calculation = tf.nn.sigmoid_cross_entropy_with_logits(                
                    labels = phonology_Placeholder,
                    logits = output_Logits
                    )
            else:
                loss_Calculation = tf.nn.softmax_cross_entropy_with_logits(                
                    labels = phonology_Placeholder,
                    logits = output_Logits
                    )
            
            loss = tf.reduce_mean(loss_Calculation)            
            loss_Display = tf.reduce_mean(loss_Calculation)
            
            global_Step = tf.Variable(0, name='global_Step', trainable = False)

            learning_Rate = tf.cast(self.learning_Rate, tf.float32)

            optimizer = tf.train.AdamOptimizer(learning_Rate)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            clipped_Gradients, global_Norm = tf.clip_by_global_norm(gradients, 1.0)
            optimize = optimizer.apply_gradients(zip(clipped_Gradients, variables), global_step=global_Step)

        with tf.variable_scope('test') as scope:
            hidden_Activation = tf.nn.tanh(hidden_Logits)
            phonology_Activation = tf.reshape(
                tf.nn.sigmoid(output_Logits) if self.is_Phoneme_Distribution else tf.nn.softmax(output_Logits),
                shape=[-1, self.pattern_Feeder.phonology_Size * self.pattern_Feeder.max_Pronunciation_Length]
                )

        with tf.variable_scope('inference') as scope:
            split_Cosine_Similarity = tf.reduce_sum(phonology_Placeholder * analysis_Input_Placeholder, axis = -1) / (tf.sqrt(tf.reduce_sum(tf.pow(phonology_Placeholder, 2), axis = -1)) * tf.sqrt(tf.reduce_sum(tf.pow(analysis_Input_Placeholder, 2), axis = -1)))
            split_Mean_Squared_Error = tf.reduce_mean(tf.pow(phonology_Placeholder - analysis_Input_Placeholder, 2), axis=-1)
            split_Euclidean_Distance = tf.sqrt(tf.reduce_sum(tf.pow(phonology_Placeholder - analysis_Input_Placeholder, 2), axis=-1))
            split_Cross_Entropy = -tf.reduce_mean(phonology_Placeholder * tf.log(analysis_Input_Placeholder + 1e-8) + (1 - phonology_Placeholder) * tf.log(1 - analysis_Input_Placeholder + 1e-8), axis = -1)

            reshaped_Phonology_Tensor = tf.reshape(phonology_Placeholder, [-1, self.pattern_Feeder.max_Pronunciation_Length * self.pattern_Feeder.phonology_Size])
            reshaped_Analysis_Input_Tensor = tf.reshape(analysis_Input_Placeholder, [-1, self.pattern_Feeder.max_Pronunciation_Length * self.pattern_Feeder.phonology_Size])            
            all_Cosine_Similarity = tf.reduce_sum(reshaped_Phonology_Tensor * reshaped_Analysis_Input_Tensor, axis = -1) / (tf.sqrt(tf.reduce_sum(tf.pow(reshaped_Phonology_Tensor, 2), axis = -1)) * tf.sqrt(tf.reduce_sum(tf.pow(reshaped_Analysis_Input_Tensor, 2), axis = -1)))
            all_Mean_Squared_Error = tf.reduce_mean(tf.pow(reshaped_Phonology_Tensor - reshaped_Analysis_Input_Tensor, 2), axis=-1)
            all_Euclidean_Distance = tf.sqrt(tf.reduce_sum(tf.pow(reshaped_Phonology_Tensor - reshaped_Analysis_Input_Tensor, 2), axis=-1))
            all_Cross_Entropy = -tf.reduce_mean(reshaped_Phonology_Tensor * tf.log(reshaped_Analysis_Input_Tensor + 1e-8) + (1 - reshaped_Phonology_Tensor) * tf.log(1 - reshaped_Analysis_Input_Tensor + 1e-8), axis = -1)

            if self.is_Phoneme_Distribution:
                tiled_Analysis_Input_Tensor = tf.tile(tf.expand_dims(analysis_Input_Placeholder, [2]), multiples = [1, 1, phoneme_Info_Placeholder.shape[0], 1])   #[batch, 17, 42, 18]                
                tiled_Phoneme_Info_Placeholder = tf.tile(tf.expand_dims(tf.expand_dims(phoneme_Info_Placeholder, [0]), [0]), multiples = [tf.shape(analysis_Input_Placeholder)[0], tf.shape(analysis_Input_Placeholder)[1], 1, 1])   #[batch, 17, 42, 18]                
                phoneme_Cosine_Similarity = tf.reduce_sum(tiled_Phoneme_Info_Placeholder * tiled_Analysis_Input_Tensor, axis = 3) / (tf.sqrt(tf.reduce_sum(tf.pow(tiled_Phoneme_Info_Placeholder, 2), axis = 3)) * tf.sqrt(tf.reduce_sum(tf.pow(tiled_Analysis_Input_Tensor, 2), axis = 3)))  #[batch, 17, 42]        
                pattern_Argmax = tf.argmax(phoneme_Cosine_Similarity, axis=2)  #[batch, 17]
            else:
                pattern_Argmax = tf.argmax(analysis_Input_Placeholder, axis=2)  #[batch, 17]
            
        self.training_Tensor_List = [global_Step, learning_Rate, loss_Display, optimize]        
        self.test_Tensor_List = [global_Step, hidden_Activation, phonology_Activation]
        self.analysis_Tensor_List = [split_Cosine_Similarity, split_Mean_Squared_Error, split_Euclidean_Distance, split_Cross_Entropy, all_Cosine_Similarity, all_Mean_Squared_Error, all_Euclidean_Distance, all_Cross_Entropy, pattern_Argmax]

        self.tf_Session.run(tf.global_variables_initializer())

    def Restore(self, load_Epoch):
        checkpoint_Path = self.extract_Dir + "/Checkpoint/Checkpoint-" + str(self.pattern_Feeder.start_Epoch)
        try:
            self.tf_Saver.restore(self.tf_Session, checkpoint_Path)
        except tf.errors.NotFoundError:
            print("here is no checkpoint about the start epoch. Stopped.")
            sys.exit()
        print("Checkpoint '", checkpoint_Path, "' is loaded.")

    def Train(self, test_Timing, checkpoint_Timing = 1000):
        if not os.path.exists(self.extract_Dir + "/Checkpoint"):
            os.makedirs(self.extract_Dir + "/Checkpoint")
        checkpoint_Path = self.extract_Dir + "/Checkpoint/Checkpoint"
        
        current_Epoch = self.pattern_Feeder.start_Epoch or 0
        self.Test(epoch= current_Epoch)
              
        while not self.pattern_Feeder.is_Finished or len(self.pattern_Feeder.pattern_Queue) > 0:
            current_Epoch, is_New_Epoch, feed_Dict = self.pattern_Feeder.Get_Pattern()            

            if is_New_Epoch and current_Epoch % test_Timing == 0:
                self.Test(epoch=current_Epoch)
            if is_New_Epoch and current_Epoch % checkpoint_Timing == 0:
                self.tf_Saver.save(self.tf_Session, checkpoint_Path, global_step = current_Epoch)
                print("Checkpoint saved")

            start_Time = time.time()
            global_Step, learning_Rate, training_Loss = self.tf_Session.run(
                fetches = self.training_Tensor_List,
                feed_dict = feed_Dict
                )[:3]

            print(
                "Spent_Time:", np.round(time.time() - start_Time, 3), "\t",
                "Global_Step:", global_Step, "\t",
                "Epoch:", current_Epoch, "\t",
                "Learning_Rate:", learning_Rate,
                "Training_Loss:", training_Loss
                )

        pattern_Generate_Thread = self.Test(epoch=current_Epoch + 1)
        self.tf_Saver.save(self.tf_Session, checkpoint_Path, global_step = current_Epoch + 1)
        print("Checkpoint saved")

        pattern_Generate_Thread.join()

    def Test(self, epoch):
        hidden_Activation_List = []
        phonology_Activation_List = []

        test_Feed_Dict_List = self.pattern_Feeder.Get_Test_Pattern_List()
        for feed_Dict in test_Feed_Dict_List:
            global_Step, hidden_Activation, phonology_Activation = self.tf_Session.run(
                fetches = self.test_Tensor_List,
                feed_dict = feed_Dict
                )
            hidden_Activation_List.append(hidden_Activation)
            phonology_Activation_List.append(phonology_Activation)
        
        pattern_Generate_Thread = Thread(
            target=self.Extract,
            args=(
                np.vstack(hidden_Activation_List).astype("float32"),
                np.vstack(phonology_Activation_List).astype("float32"),
                epoch,
                self.pattern_Feeder.trained_Pattern_Index_List_List[:epoch]
                )
            )
        pattern_Generate_Thread.start()

        return pattern_Generate_Thread

    def Extract(self, hidden_Activation, phonology_Activation, epoch, trained_Pattern_Index_List_List):
        if not os.path.exists(self.extract_Dir + "/Result"):
            os.makedirs(self.extract_Dir + "/Result")

        if not os.path.isfile(self.extract_Dir + "/Result/Metadata.pickle"):
            metadata_Dict = {}
            metadata_Dict["Orthography_Size"] = self.pattern_Feeder.orthography_Size
            metadata_Dict["Phonology_Size"] = self.pattern_Feeder.phonology_Size
            metadata_Dict["Hidden_Size"] = self.hidden_Size
            metadata_Dict["Learning_Rate"] = self.learning_Rate
            metadata_Dict["Is_Phoneme_Distribution"] = self.is_Phoneme_Distribution
                        
            metadata_Dict["Target_Pattern"] = self.pattern_Feeder.target_Pattern
            metadata_Dict["Phoneme_Pattern"] = self.pattern_Feeder.phoneme_Pattern

            metadata_Dict["Word_Index_Dict"] = self.pattern_Feeder.word_Index_Dict
            metadata_Dict["Letter_Index_Dict"] = self.pattern_Feeder.letter_Index_Dict
            metadata_Dict["Phoneme_Index_Dict"] = self.pattern_Feeder.phoneme_Index_Dict
            
            metadata_Dict["Pronunciation_Dict"] = self.pattern_Feeder.pronunciation_Dict
            metadata_Dict["Frequency_Dict"] = self.pattern_Feeder.frequency_Dict
            metadata_Dict["Human_RT_Dict"] = self.pattern_Feeder.human_RT_Dict
            
            with open(self.extract_Dir + "/Result/Metadata.pickle", "wb") as f:
                pickle.dump(metadata_Dict, f, protocol=0)

        result_Dict = {}
        result_Dict["Epoch"] = epoch
        result_Dict["Hidden"] = hidden_Activation.astype("float32")
        result_Dict["Result"] = phonology_Activation.astype("float32")
        result_Dict["Trained_Pattern_Count_Dict"] = {index: 0 for index in self.pattern_Feeder.word_Index_Dict.values()}
        result_Dict["Trained_Pattern_Count_Dict"].update(Counter([index for index_List in trained_Pattern_Index_List_List for index in index_List]))
                
        with open(self.extract_Dir + "/Result/%s.pickle" % epoch, "wb") as f:
            pickle.dump(result_Dict, f, protocol=0)

    def Inference(self, word_List, pronunciation_List, file_Prefix= 'Inference', analysis_Batch_Size = 20):
        target_Phonology_Pattern, feed_Dict_List = self.pattern_Feeder.Get_InferenceTest_Pattern_List(word_List, pronunciation_List)
        
        hidden_Activation_List = []
        phonology_Activation_List = []
        for feed_Dict in feed_Dict_List:
            _, hidden_Activation, phonology_Activation = self.tf_Session.run(
                fetches = self.test_Tensor_List,
                feed_dict = feed_Dict
                )
            hidden_Activation_List.append(hidden_Activation)
            phonology_Activation_List.append(np.reshape(phonology_Activation, [-1, self.pattern_Feeder.max_Pronunciation_Length, self.pattern_Feeder.phonology_Size, ]))

        hidden_Activation = np.vstack(hidden_Activation_List)
        phonology_Activation = np.vstack(phonology_Activation_List)


        analysis_List_Dict = {
            'Split': {
                'Cosine_Similarity': [],
                'Mean_Squared_Error': [],
                'Euclidean_Distance': [],
                'Cross_Entropy': [],                
                },
            'All': {
                'Cosine_Similarity': [],
                'Mean_Squared_Error': [],
                'Euclidean_Distance': [],
                'Cross_Entropy': [],     
                },
            'Pattern_Argmax': []
            }

        pattern_Index_List = list(range(len(word_List)))
        pattern_Index_Batch_List = [pattern_Index_List[x:x+analysis_Batch_Size] for x in range(0, len(pattern_Index_List), analysis_Batch_Size)]

        for pattern_Index_Batch in pattern_Index_Batch_List:
            split_Cosine_Similarity, split_Mean_Squared_Error, split_Euclidean_Distance, split_Cross_Entropy, all_Cosine_Similarity, all_Mean_Squared_Error, all_Euclidean_Distance, all_Cross_Entropy, pattern_Argmax = self.tf_Session.run(
                fetches = self.analysis_Tensor_List,
                feed_dict = {
                    self.pattern_Feeder.placeholder_Dict['Phoneme_Info']: self.pattern_Feeder.phoneme_Pattern,
                    self.pattern_Feeder.placeholder_Dict['Phonology']: target_Phonology_Pattern[pattern_Index_Batch],
                    self.pattern_Feeder.placeholder_Dict['Analysis_Input']: phonology_Activation[pattern_Index_Batch],
                    }
                )
            analysis_List_Dict['Split']['Cosine_Similarity'].append(split_Cosine_Similarity)
            analysis_List_Dict['Split']['Mean_Squared_Error'].append(split_Mean_Squared_Error)
            analysis_List_Dict['Split']['Euclidean_Distance'].append(split_Euclidean_Distance)
            analysis_List_Dict['Split']['Cross_Entropy'].append(split_Cross_Entropy)
            analysis_List_Dict['All']['Cosine_Similarity'].append(all_Cosine_Similarity)
            analysis_List_Dict['All']['Mean_Squared_Error'].append(all_Mean_Squared_Error)
            analysis_List_Dict['All']['Euclidean_Distance'].append(all_Euclidean_Distance)
            analysis_List_Dict['All']['Cross_Entropy'].append(all_Cross_Entropy)
            analysis_List_Dict['Pattern_Argmax'].append(pattern_Argmax)

        split_Cosine_Similarity = np.vstack(analysis_List_Dict['Split']['Cosine_Similarity'])
        split_Mean_Squared_Error = np.vstack(analysis_List_Dict['Split']['Mean_Squared_Error'])
        split_Euclidean_Distance = np.vstack(analysis_List_Dict['Split']['Euclidean_Distance'])
        split_Cross_Entropy = np.vstack(analysis_List_Dict['Split']['Cross_Entropy'])
        all_Cosine_Similarity = np.hstack(analysis_List_Dict['All']['Cosine_Similarity'])
        all_Mean_Squared_Error = np.hstack(analysis_List_Dict['All']['Mean_Squared_Error'])
        all_Euclidean_Distance = np.hstack(analysis_List_Dict['All']['Euclidean_Distance'])
        all_Cross_Entropy = np.hstack(analysis_List_Dict['All']['Cross_Entropy'])
        pattern_Argmax = np.vstack(analysis_List_Dict['Pattern_Argmax'])

        self.Extract_Inference(
            word_List= word_List,
            pronunciation_List= pronunciation_List,
            split_Cosine_Similarity_Array= split_Cosine_Similarity,
            split_Mean_Squared_Error_Array= split_Mean_Squared_Error,
            split_Euclidean_Distance_Array= split_Euclidean_Distance,
            split_Cross_Entropy_Array= split_Cross_Entropy,
            all_Cosine_Similarity_Array= all_Cosine_Similarity,
            all_Mean_Squared_Error_Array= all_Mean_Squared_Error,
            all_Euclidean_Distance_Array= all_Euclidean_Distance,
            all_Cross_Entropy_Array= all_Cross_Entropy,
            pattern_Argmax_Array= pattern_Argmax,
            file_Prefix= file_Prefix
            )

    def Extract_Inference(
        self,
        word_List,
        pronunciation_List,
        split_Cosine_Similarity_Array,
        split_Mean_Squared_Error_Array,
        split_Euclidean_Distance_Array,
        split_Cross_Entropy_Array,
        all_Cosine_Similarity_Array,
        all_Mean_Squared_Error_Array,
        all_Euclidean_Distance_Array,
        all_Cross_Entropy_Array,
        pattern_Argmax_Array,
        file_Prefix= 'Inference'
        ):
        index_Phoneme_Dict = {index: phoneme for phoneme, index in self.pattern_Feeder.phoneme_Index_Dict.items() if len(phoneme) == 1}
        def Pronunciation_Generate(phoneme_Argmax_Array):            
            pronunciation = "".join([index_Phoneme_Dict[phoneme_Index] for phoneme_Index in phoneme_Argmax_Array])
            while len(pronunciation) > 0 and pronunciation[-1] == "_":
                pronunciation = pronunciation[:-1]
            return pronunciation

        column_Title_List = ['Word', 'Targe_Pronunciation', 'Exported_Pronunciation', 'Accuracy', 'Loss_Type', 'All_Loss'] + ['Split_Loss_Location_{}'.format(x) for x in range(self.pattern_Feeder.max_Pronunciation_Length)]

        export_List = ['\t'.join(column_Title_List)]
        for word, pronunciation, split_Cosine_Similarity, split_Mean_Squared_Error, split_Euclidean_Distance, split_Cross_Entropy, all_Cosine_Similarity, all_Mean_Squared_Error, all_Euclidean_Distance, all_Cross_Entropy, pattern_Argmax in zip(
            word_List, pronunciation_List, split_Cosine_Similarity_Array, split_Mean_Squared_Error_Array, split_Euclidean_Distance_Array, split_Cross_Entropy_Array, all_Cosine_Similarity_Array, all_Mean_Squared_Error_Array, all_Euclidean_Distance_Array, all_Cross_Entropy_Array, pattern_Argmax_Array
            ):
            exported_Pronunciation = Pronunciation_Generate(pattern_Argmax)
            for loss_Label, all_Loss, split_Loss in zip(
                ['Cosine_Similarity', 'Mean_Squared_Error', 'Euclidean_Distance', 'Cross_Entropy'],
                [all_Cosine_Similarity, all_Mean_Squared_Error, all_Euclidean_Distance, all_Cross_Entropy],
                [split_Cosine_Similarity, split_Mean_Squared_Error, split_Euclidean_Distance, split_Cross_Entropy]
                ):
                new_Line_List = [word, pronunciation, exported_Pronunciation, int(pronunciation == exported_Pronunciation), loss_Label, all_Loss] + [x for x in split_Loss]
                export_List.append('\t'.join(['{}'.format(x) for x in new_Line_List]))

        os.makedirs(os.path.join(self.extract_Dir, "Inference").replace('\\', '/'), exist_ok= True)
        with open(os.path.join(self.extract_Dir, "Inference", '{}.Inference.txt'.format(file_Prefix)).replace('\\', '/'), "w") as f:
            f.write('\n'.join(export_List))



if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-dir", "--directory", required=True)
    argParser.add_argument("-ht", "--hidden_calc_type", required=True)
    argParser.add_argument("-hu", "--hidden_unit", required=True)
    argParser.add_argument("-lr", "--learning_rate", required=True)
    argParser.add_argument("-e", "--epoch", required=True)
    argParser.add_argument("-tt", "--test_timing", required=True)
    argParser.add_argument("-fre", "--frequency", action="store_true")
    argParser.set_defaults(frequency = False)
    argParser.add_argument("-emb", "--embedding", required=False)
    argParser.set_defaults(embedding = None)
    argParser.add_argument("-dstr", "--distribution", required=False)
    argParser.set_defaults(distribution = None)
    argParser.add_argument("-try", "--try", required=False)    
    argument_Dict = vars(argParser.parse_args())

    argument_Dict["hidden_unit"] = int(argument_Dict["hidden_unit"])
    argument_Dict["learning_rate"] = float(argument_Dict["learning_rate"])
    argument_Dict["epoch"] = int(argument_Dict["epoch"])
    argument_Dict["test_timing"] = int(argument_Dict["test_timing"])
    if argument_Dict["try"] is not None:
        argument_Dict["try"] = int(argument_Dict["try"])
    if argument_Dict["embedding"] is not None:
        argument_Dict["embedding"] = int(argument_Dict["embedding"])

    extract_Dir_List = []
    extract_Dir_List.append("HT_{}".format(argument_Dict["hidden_calc_type"]))
    extract_Dir_List.append("HU_{}".format(argument_Dict["hidden_unit"]))
    extract_Dir_List.append("LR_{}".format(str(argument_Dict["learning_rate"])[2:]))
    extract_Dir_List.append("E_{}".format(argument_Dict["epoch"]))
    extract_Dir_List.append("TT_{}".format(argument_Dict["test_timing"]))
    if argument_Dict["frequency"]:
        extract_Dir_List.append("Fre")
    if argument_Dict["embedding"]:
        extract_Dir_List.append("EMB_{}".format(argument_Dict["embedding"]))
    if argument_Dict["distribution"]:
        extract_Dir_List.append("DSTR_True")
    if argument_Dict["try"] is not None:
        extract_Dir_List.append("TRY_{}".format(argument_Dict["try"]))
    extract_Dir = argument_Dict["directory"] +"/" + ".".join(extract_Dir_List)
    
    new_VOISeR = VOISeR(        
        pattern_File = "ELP_groupData.csv",        
        hidden_Calc_Type= argument_Dict["hidden_calc_type"],
        hidden_Size = argument_Dict["hidden_unit"],
        learning_Rate = argument_Dict["learning_rate"],
        batch_Size = 1000,        
        max_Epoch = argument_Dict["epoch"],
        use_Frequency = argument_Dict["frequency"],
        orthography_Embedding_Size = argument_Dict["embedding"],
        phoneme_Feature_File= argument_Dict["distribution"],
        extract_Dir = extract_Dir
        )
    new_VOISeR.Train(test_Timing=argument_Dict["test_timing"], checkpoint_Timing=argument_Dict["test_timing"])