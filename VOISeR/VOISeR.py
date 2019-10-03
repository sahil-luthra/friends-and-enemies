#1. Only feedback, only hidden
#2. Frequency applied
#3. 학습중 해당 패턴이 얼마나 학습되었는가를 표시해줄 것

import tensorflow as tf;
import numpy as np;
import _pickle as pickle;
from tensorflow.contrib.seq2seq import BasicDecoder, TrainingHelper, dynamic_decode;
from tensorflow.contrib.rnn import BasicRNNCell;
from threading import Thread;
import time, os, sys, ctypes, argparse;
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt;
from collections import Counter;
from Cells import *;
from Pattern_Feeder import Pattern_Feeder;

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
        phoneme_Feature_File = None,    #"phonetic_feature_definitions_18_features.csv"
        extract_Dir = "Temp"):
        ctypes.windll.kernel32.SetConsoleTitleW(extract_Dir.replace("_", ":").replace(".", " "));

        self.tf_Session = tf.Session();
                
        self.hidden_Calc_Type = hidden_Calc_Type;
        self.hidden_Size = hidden_Size;
        self.learning_Rate = learning_Rate;
        self.orthography_Embedding_Size = orthography_Embedding_Size;
        self.is_Phoneme_Distribution = bool(phoneme_Feature_File);
        self.extract_Dir = extract_Dir;

        self.pattern_Feeder = Pattern_Feeder(            
            pattern_File = pattern_File,
            batch_Size = batch_Size,
            start_Epoch = load_Epoch or 0,
            max_Epoch = max_Epoch,
            use_Frequency = use_Frequency,
            use_Orthography_Embedding = bool(orthography_Embedding_Size),
            phoneme_Feature_File = phoneme_Feature_File,
            )        

        self.Tensor_Generate();

        self.tf_Saver = tf.train.Saver(max_to_keep=0);

        if load_Epoch is not None:
            self.Restore(load_Epoch);

    def Tensor_Generate(self):        
        with tf.variable_scope('feedback_Model') as scope:
            orthography_Placeholder = self.pattern_Feeder.placeholder_Dict["Orthography"];
            phonology_Placeholder = self.pattern_Feeder.placeholder_Dict["Phonology"];
            length_Placeholder = self.pattern_Feeder.placeholder_Dict["Length"];

            batch_Size = tf.shape(orthography_Placeholder)[0];

            if self.orthography_Embedding_Size is not None:
                orthography_Embedding = tf.get_variable(
                    name = "orthography_Embedding",
                    shape = (self.pattern_Feeder.orthography_Size, self.orthography_Embedding_Size),
                    dtype = tf.float32,
                    initializer = tf.truncated_normal_initializer(stddev=0.5)
                )
                orthography_Pattern = tf.reshape(tf.nn.embedding_lookup(orthography_Embedding, orthography_Placeholder), shape=(-1, self.pattern_Feeder.orthography_Size * self.orthography_Embedding_Size)) ; #Shape: (batch_Size, orthography_Length * embedded_Pattern_Size);
            else:
                orthography_Pattern = orthography_Placeholder
            
            input_Pattern = tf.tile(
                input= tf.expand_dims(orthography_Pattern, axis=1), 
                multiples= [1, self.pattern_Feeder.max_Pronunciation_Length, 1]
                )

            helper = TrainingHelper(
                inputs=input_Pattern,
                sequence_length = length_Placeholder,
                time_major = False
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
                assert False;

            decoder_Initial_State = rnnCell.zero_state(batch_size=batch_Size, dtype = tf.float32);
            helper = TrainingHelper(
                inputs=input_Pattern,
                sequence_length = length_Placeholder,
                )
            decoder = BasicDecoder(
                cell=rnnCell, 
                helper=helper, 
                initial_state=decoder_Initial_State
                );

            if self.hidden_Calc_Type.upper() in ["B".upper(), "O".upper()]:
                hidden_and_Output_Logits, final_State, _ = dynamic_decode(decoder = decoder)
                hidden_Logits, output_Logits = tf.split(hidden_and_Output_Logits.rnn_output, [self.hidden_Size, self.pattern_Feeder.phonology_Size], axis=2);

            elif self.hidden_Calc_Type.upper() == "H".upper():
                hidden_Logits = dynamic_decode(decoder = decoder)[0].rnn_output;
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
            
            loss = tf.reduce_mean(loss_Calculation);            
            loss_Display = tf.reduce_mean(loss_Calculation);
            
            global_Step = tf.Variable(0, name='global_Step', trainable = False);

            learning_Rate = tf.cast(self.learning_Rate, tf.float32)

            optimizer = tf.train.AdamOptimizer(learning_Rate);
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            clipped_Gradients, global_Norm = tf.clip_by_global_norm(gradients, 1.0)
            optimize = optimizer.apply_gradients(zip(clipped_Gradients, variables), global_step=global_Step)

        with tf.variable_scope('test') as scope:
            hidden_Activation = tf.nn.tanh(hidden_Logits);
            phonology_Activation = tf.reshape(
                tf.nn.sigmoid(output_Logits) if self.is_Phoneme_Distribution else tf.nn.softmax(output_Logits),
                shape=[-1, self.pattern_Feeder.phonology_Size * self.pattern_Feeder.max_Pronunciation_Length]
                )
            
        self.training_Tensor_List = [global_Step, learning_Rate, loss_Display, optimize];        
        self.test_Tensor_List = [global_Step, hidden_Activation, phonology_Activation];

        self.tf_Session.run(tf.global_variables_initializer());

    def Restore(self, load_Epoch):
        checkpoint = self.extract_Dir + "/Checkpoint\\Checkpoint-" + str(self.pattern_Feeder.start_Epoch);
        try:
            self.tf_Saver.restore(self.tf_Session, checkpoint);
        except tf.errors.NotFoundError:
            print("here is no checkpoint about the start epoch. Stopped.")
            sys.exit();
        print("Checkpoint '", checkpoint, "' is loaded.");
        
    def Train(self, test_Timing, checkpoint_Timing = 1000):
        if not os.path.exists(self.extract_Dir + "/Checkpoint"):
            os.makedirs(self.extract_Dir + "/Checkpoint");
        checkpoint_Path = self.extract_Dir + "/Checkpoint/Checkpoint";
        
        current_Epoch = self.pattern_Feeder.start_Epoch or 0;
        self.Test(epoch= current_Epoch);
              
        while not self.pattern_Feeder.is_Finished or len(self.pattern_Feeder.pattern_Queue) > 0:
            current_Epoch, is_New_Epoch, feed_Dict = self.pattern_Feeder.Get_Pattern();            

            if is_New_Epoch and current_Epoch % test_Timing == 0:
                self.Test(epoch=current_Epoch);
            if is_New_Epoch and current_Epoch % checkpoint_Timing == 0:
                self.tf_Saver.save(self.tf_Session, checkpoint_Path, global_step = current_Epoch);
                print("Checkpoint saved");

            start_Time = time.time();
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

        pattern_Generate_Thread = self.Test(epoch=current_Epoch + 1);
        self.tf_Saver.save(self.tf_Session, checkpoint_Path, global_step = current_Epoch + 1);
        print("Checkpoint saved");

        pattern_Generate_Thread.join();

    def Test(self, epoch):
        hidden_Activation_List = [];
        phonology_Activation_List = [];

        test_Feed_Dict_List = self.pattern_Feeder.Get_Test_Pattern_List();
        for feed_Dict in test_Feed_Dict_List:
            global_Step, hidden_Activation, phonology_Activation = self.tf_Session.run(
                fetches = self.test_Tensor_List,
                feed_dict = feed_Dict
                )
            hidden_Activation_List.append(hidden_Activation);
            phonology_Activation_List.append(phonology_Activation);
        
        pattern_Generate_Thread = Thread(
            target=self.Extract,
            args=(
                np.vstack(hidden_Activation_List).astype("float32"),
                np.vstack(phonology_Activation_List).astype("float32"),
                epoch,
                self.pattern_Feeder.trained_Pattern_Index_List_List[:epoch]
                )
            )
        pattern_Generate_Thread.start();

        return pattern_Generate_Thread;

    def Extract(self, hidden_Activation, phonology_Activation, epoch, trained_Pattern_Index_List_List):
        if not os.path.exists(self.extract_Dir + "/Result"):
            os.makedirs(self.extract_Dir + "/Result");

        if not os.path.isfile(self.extract_Dir + "/Result/Metadata.pickle"):
            metadata_Dict = {};
            metadata_Dict["Orthography_Size"] = self.pattern_Feeder.orthography_Size;
            metadata_Dict["Phonology_Size"] = self.pattern_Feeder.phonology_Size;
            metadata_Dict["Hidden_Size"] = self.hidden_Size;
            metadata_Dict["Learning_Rate"] = self.learning_Rate;
            metadata_Dict["Is_Phoneme_Distribution"] = self.is_Phoneme_Distribution;
                        
            metadata_Dict["Target_Pattern"] = self.pattern_Feeder.target_Pattern;
            metadata_Dict["Phoneme_Pattern"] = self.pattern_Feeder.phoneme_Pattern;

            metadata_Dict["Word_Index_Dict"] = self.pattern_Feeder.word_Index_Dict;
            metadata_Dict["Letter_Index_Dict"] = self.pattern_Feeder.letter_Index_Dict;
            metadata_Dict["Phoneme_Index_Dict"] = self.pattern_Feeder.phoneme_Index_Dict;
            
            metadata_Dict["Pronunciation_Dict"] = self.pattern_Feeder.pronunciation_Dict;
            metadata_Dict["Frequency_Dict"] = self.pattern_Feeder.frequency_Dict;
            metadata_Dict["Human_RT_Dict"] = self.pattern_Feeder.human_RT_Dict;
            
            with open(self.extract_Dir + "/Result/Metadata.pickle", "wb") as f:
                pickle.dump(metadata_Dict, f, protocol=0);

        result_Dict = {};
        result_Dict["Epoch"] = epoch;
        result_Dict["Hidden"] = hidden_Activation.astype("float32");
        result_Dict["Result"] = phonology_Activation.astype("float32");
        result_Dict["Trained_Pattern_Count_Dict"] = {index: 0 for index in self.pattern_Feeder.word_Index_Dict.values()};
        result_Dict["Trained_Pattern_Count_Dict"].update(Counter([index for index_List in trained_Pattern_Index_List_List for index in index_List]));
                
        with open(self.extract_Dir + "/Result/%s.pickle" % epoch, "wb") as f:
            pickle.dump(result_Dict, f, protocol=0);

if __name__ == "__main__":
    argParser = argparse.ArgumentParser();
    argParser.add_argument("-dir", "--directory", required=True);
    argParser.add_argument("-ht", "--hidden_calc_type", required=True);
    argParser.add_argument("-hu", "--hidden_unit", required=True);
    argParser.add_argument("-lr", "--learning_rate", required=True);
    argParser.add_argument("-e", "--epoch", required=True);
    argParser.add_argument("-tt", "--test_timing", required=True);
    argParser.add_argument("-fre", "--frequency", action="store_true");
    argParser.set_defaults(frequency = False);
    argParser.add_argument("-emb", "--embedding", required=False);
    argParser.set_defaults(embedding = None);
    argParser.add_argument("-dstr", "--distribution", required=False);
    argParser.set_defaults(distribution = None);
    argParser.add_argument("-try", "--try", required=False);    
    argument_Dict = vars(argParser.parse_args());

    argument_Dict["hidden_unit"] = int(argument_Dict["hidden_unit"]);
    argument_Dict["learning_rate"] = float(argument_Dict["learning_rate"]);
    argument_Dict["epoch"] = int(argument_Dict["epoch"]);
    argument_Dict["test_timing"] = int(argument_Dict["test_timing"]);
    if argument_Dict["try"] is not None:
        argument_Dict["try"] = int(argument_Dict["try"]);
    if argument_Dict["embedding"] is not None:
        argument_Dict["embedding"] = int(argument_Dict["embedding"]);

    extract_Dir_List = [];
    extract_Dir_List.append("HT_{}".format(argument_Dict["hidden_calc_type"]));
    extract_Dir_List.append("HU_{}".format(argument_Dict["hidden_unit"]));
    extract_Dir_List.append("LR_{}".format(str(argument_Dict["learning_rate"])[2:]));
    extract_Dir_List.append("E_{}".format(argument_Dict["epoch"]));
    extract_Dir_List.append("TT_{}".format(argument_Dict["test_timing"]));
    if argument_Dict["frequency"]:
        extract_Dir_List.append("Fre");
    if argument_Dict["embedding"]:
        extract_Dir_List.append("EMB_{}".format(argument_Dict["embedding"]));
    if argument_Dict["distribution"]:
        extract_Dir_List.append("DSTR_True");
    if argument_Dict["try"] is not None:
        extract_Dir_List.append("TRY_{}".format(argument_Dict["try"]));
    extract_Dir = argument_Dict["directory"] +"/" + ".".join(extract_Dir_List);
    
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
    new_VOISeR.Train(test_Timing=argument_Dict["test_timing"], checkpoint_Timing=argument_Dict["test_timing"]);