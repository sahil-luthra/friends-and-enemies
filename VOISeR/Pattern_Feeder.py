import numpy as np
import tensorflow as tf
from threading import Thread
from collections import deque
from random import shuffle
import time

class Pattern_Feeder:
    def __init__(
        self, 
        pattern_File,
        batch_Size, start_Epoch, 
        max_Epoch, 
        use_Length = False, 
        use_Frequency = False, 
        use_Orthography_Embedding = False, 
        phoneme_Feature_File = None, 
        max_Queue = 100):
        self.pattern_File = pattern_File
        self.batch_Size = batch_Size        
        self.start_Epoch = start_Epoch
        self.max_Epoch = max_Epoch
        self.use_Length = use_Length
        self.use_Frequency = use_Frequency
        self.use_Orthography_Embedding = use_Orthography_Embedding
        self.phoneme_Feature_File = phoneme_Feature_File
        self.max_Queue = max_Queue
        
        self.Load_Data()
        self.Placeholder_Generate()

        self.is_Finished = False
        self.pattern_Queue = deque()
        self.trained_Pattern_Index_List_List = []

        self.Test_Pattern_Generate()
                
        pattern_Generate_Thread = Thread(target=self.Pattern_Generate)
        pattern_Generate_Thread.daemon = True
        pattern_Generate_Thread.start()
    
    def Load_Data(self):
        with open (self.pattern_File, "r") as f:
            readLines = f.readlines()[1:]
        splited_ReadLine = [readLine.replace('"','').strip().split(",")[1:] for readLine in readLines]
        self.word_Index_Dict = {word.lower(): index for index, (word, _, _, _, _) in enumerate(splited_ReadLine)}
        self.pronunciation_Dict = {word.lower(): pronunciation for word, pronunciation, _, _, _ in splited_ReadLine}
        self.frequency_Dict = {word.lower(): float(frequency) * 0.05 + 0.1 for word, _, _, frequency, _ in splited_ReadLine}
        self.human_RT_Dict = {word.lower(): float(rt) for word, _, _, _, rt in splited_ReadLine}
        
        self.max_Word_Length = max([len(word) for word in self.word_Index_Dict.keys()])
        self.max_Pronunciation_Length = max([len(pronunciation) for pronunciation in self.pronunciation_Dict.values()])

        if len(self.word_Index_Dict.keys()) != len(set(self.word_Index_Dict.keys())):
            raise Exception("The data contains the same word more than once")

        letter_Set = set()
        for word in self.word_Index_Dict.keys():
            letter_Set.update(set(word))
        self.letter_List = list(letter_Set)
        self.letter_List.sort()
        self.letter_List += ["_"]

        phoneme_Set = set()
        for pronunciation in self.pronunciation_Dict.values():
            phoneme_Set.update(set(pronunciation))
        self.phoneme_List = list(phoneme_Set)
        self.phoneme_List.sort()
        self.phoneme_List += ["_"]

        self.letter_Index_Dict = {}
        for letter_Index, letter in enumerate(self.letter_List):
            self.letter_Index_Dict[letter] = letter_Index
        for slot_Index in range(self.max_Word_Length):
            for letter_Index, letter in enumerate(self.letter_List):
                self.letter_Index_Dict[slot_Index, letter] = slot_Index * len(self.letter_List) + letter_Index

        self.phoneme_Index_Dict = {}   #There are two key types: [phoneme], [slot_Index, phoneme]
        for phoneme_Index, phoneme in enumerate(self.phoneme_List):
            self.phoneme_Index_Dict[phoneme] = phoneme_Index
        for slot_Index in range(self.max_Pronunciation_Length):
            for phoneme_Index, phoneme in enumerate(self.phoneme_List):
                self.phoneme_Index_Dict[slot_Index, phoneme] = slot_Index * len(self.phoneme_List) + phoneme_Index

        if bool(self.phoneme_Feature_File) == True:
            with open (self.phoneme_Feature_File, "r") as f:
                readLines = f.readlines()[1:]
            splited_ReadLine = [readLine.replace('"','').strip().split(",") for readLine in readLines]
            self.phoneme_Pattern_Dict = {pattern[0]: np.array([float(x) for x in pattern[1:]]) for pattern in splited_ReadLine}
        else:
            self.phoneme_Pattern_Dict = {}
            for phoneme in self.phoneme_List:
                new_Pattern = np.zeros(shape = len(self.phoneme_List))
                new_Pattern[self.phoneme_Index_Dict[phoneme]] = 1
                self.phoneme_Pattern_Dict[phoneme] = new_Pattern
                        
        if self.use_Orthography_Embedding:
            self.orthography_Size = self.max_Word_Length
        else:
            self.orthography_Size = self.max_Word_Length * len(self.letter_List)
        self.phonology_Size = len(self.phoneme_Pattern_Dict[self.phoneme_List[0]])
                
        self.phoneme_Pattern = np.zeros((len(self.phoneme_List), self.phonology_Size))
        for index, phoneme in enumerate(self.phoneme_List):
            self.phoneme_Pattern[index] = self.phoneme_Pattern_Dict[phoneme]

    def Placeholder_Generate(self):
        self.placeholder_Dict = {}
        with tf.variable_scope('placeHolders') as scope:
            self.placeholder_Dict["Orthography"] = tf.placeholder(
                tf.int32 if self.use_Orthography_Embedding else tf.float32, 
                shape=(None, self.orthography_Size), 
                name = "orthography_Placeholder"
                )
            self.placeholder_Dict["Phonology"] = tf.placeholder(
                tf.float32, 
                shape=(None, self.max_Pronunciation_Length, self.phonology_Size), 
                name = "phonology_Placeholder"
                )
            self.placeholder_Dict["Length"] = tf.placeholder(
                tf.int32, 
                shape=(None,), 
                name = "length_Placeholder"
                )

    def Word_to_Pattern(self, word):
        word = word + "_" * (self.max_Word_Length - len(word))

        if self.use_Orthography_Embedding:
            new_Pattern = np.array([self.letter_Index_Dict[letter] for letter in word]).astype(np.int32)
        else:
            new_Pattern = np.zeros((self.orthography_Size)).astype(np.float32)
            for slot_Index, letter in enumerate(word):
                new_Pattern[self.letter_Index_Dict[slot_Index, letter]] = 1
        
        return new_Pattern

    def Pronunciation_to_Pattern(self, pronunciation):
        pronunciation = pronunciation + "_" * (self.max_Pronunciation_Length - len(pronunciation))

        new_Pattern = np.zeros((self.max_Pronunciation_Length, self.phonology_Size))
        for cycle_Index, phoneme in enumerate(pronunciation):
            new_Pattern[cycle_Index] = self.phoneme_Pattern_Dict[phoneme]
                    
        return new_Pattern

    def Pattern_Generate(self):
        #Batched Pattern Making
        pattern_Count  = len(self.word_Index_Dict)
        
        orthography_Pattern = np.zeros((pattern_Count, self.orthography_Size)).astype(np.int32 if self.use_Orthography_Embedding else np.float32)
        phonology_Pattern = np.zeros((pattern_Count, self.max_Pronunciation_Length, self.phonology_Size)).astype(np.float32)
        cycle_Pattern = np.zeros((pattern_Count)).astype(np.int32)
        frequency_Pattern = np.zeros((pattern_Count)).astype(np.float32)

        for word, index in self.word_Index_Dict.items():
            orthography_Pattern[index] = self.Word_to_Pattern(word)
            phonology_Pattern[index] = self.Pronunciation_to_Pattern(self.pronunciation_Dict[word])
            cycle_Pattern[index] = len(self.pronunciation_Dict[word]) if self.use_Length else self.max_Pronunciation_Length
            frequency_Pattern[index] = self.frequency_Dict[word]   #Should be changed
            
        #Queue
        for epoch in range(self.start_Epoch, self.max_Epoch):
            pattern_Index_List = np.arange(pattern_Count)
            if self.use_Frequency:
                pattern_Index_List = pattern_Index_List[frequency_Pattern > np.random.rand(pattern_Count)]
            self.trained_Pattern_Index_List_List.append(pattern_Index_List)
            shuffle(pattern_Index_List)
            pattern_Index_Batch_List = [pattern_Index_List[x:x+self.batch_Size] for x in range(0, len(pattern_Index_List), self.batch_Size)]
            
            current_Index = 0
            is_New_Epoch = True
            while current_Index < len(pattern_Index_Batch_List):
                if len(self.pattern_Queue) >= self.max_Queue:
                    time.sleep(0.1)
                    continue
                             
                selected_Orthography_Pattern = orthography_Pattern[pattern_Index_Batch_List[current_Index]]
                selected_Phonology_Pattern = phonology_Pattern[pattern_Index_Batch_List[current_Index]]
                selected_Cycle_Pattern = cycle_Pattern[pattern_Index_Batch_List[current_Index]]
                
                new_Feed_Dict= {
                    self.placeholder_Dict["Orthography"]: selected_Orthography_Pattern,
                    self.placeholder_Dict["Phonology"]: selected_Phonology_Pattern,
                    self.placeholder_Dict["Length"]: selected_Cycle_Pattern
                    }
                self.pattern_Queue.append([epoch, is_New_Epoch, new_Feed_Dict])    
                
                current_Index += 1
                is_New_Epoch = False

        self.is_Finished = True

    def Get_Pattern(self):
        while len(self.pattern_Queue) == 0:
            time.sleep(0.01)
        return self.pattern_Queue.popleft()

    def Test_Pattern_Generate(self):
        self.test_Pattern_Count  = len(self.word_Index_Dict)
        
        self.test_Orthography_Pattern = np.zeros((self.test_Pattern_Count, self.orthography_Size)).astype(np.int32 if self.use_Orthography_Embedding else np.float32)
        self.test_Cycle_Pattern = np.zeros((self.test_Pattern_Count)).astype(np.int32)
        phonology_Pattern = np.zeros((self.test_Pattern_Count, self.max_Pronunciation_Length, self.phonology_Size)).astype(np.float32)  #This is not used in the test. This is for the result analysis.

        for word, index in self.word_Index_Dict.items():
            self.test_Orthography_Pattern[index] = self.Word_to_Pattern(word)
            self.test_Cycle_Pattern[index] = self.max_Pronunciation_Length
            phonology_Pattern[index] = self.Pronunciation_to_Pattern(self.pronunciation_Dict[word])

        self.target_Pattern = np.reshape(phonology_Pattern, [-1, self.phonology_Size * self.max_Pronunciation_Length])
    
    def Get_Test_Pattern_List(self):
        pattern_Index_List = list(range(self.test_Pattern_Count))
        pattern_Index_Batch_List = [pattern_Index_List[x:x+self.batch_Size] for x in range(0, len(pattern_Index_List), self.batch_Size)]

        new_Feed_Dict_List = []

        for pattern_Index_Batch in pattern_Index_Batch_List:
            #Semantic pattern is not used in the test.
            new_Feed_Dict= {
                self.placeholder_Dict["Orthography"]: self.test_Orthography_Pattern[pattern_Index_Batch],
                self.placeholder_Dict["Length"]: self.test_Cycle_Pattern[pattern_Index_Batch]
                }
            new_Feed_Dict_List.append(new_Feed_Dict)

        return new_Feed_Dict_List


if __name__ == "__main__":
    from Pattern_Feeder import Pattern_Feeder
    new_Pattern_Feeder = Pattern_Feeder(
        pattern_File = "ELP_groupData.csv",         
        batch_Size = 3000,
        start_Epoch = 0, 
        max_Epoch = 1000, 
        use_Length = False, 
        use_Frequency = False, 
        use_Orthography_Embedding = True, 
        phoneme_Feature_File = None,#'phonetic_feature_definitions_18_features.csv', 
        max_Queue = 100
        )
        
    print(new_Pattern_Feeder.Get_Pattern())