rem python VOISeR.py -dir D:/VOISeR_Results -ht B -hu 300 -lr 0.0005 -e 20000 -tt 2000 -fre -dstr phonetic_feature_definitions_18_features.csv
rem python Result_Analysis.py -f D:/VOISeR_Results/HT_B.HU_300.LR_0005.E_20000.TT_2000.Fre.DSTR_True

rem python VOISeR.py -dir D:/VOISeR_Results -ht O -hu 300 -lr 0.0005 -e 20000 -tt 2000 -fre -dstr phonetic_feature_definitions_18_features.csv
rem python Result_Analysis.py -f D:/VOISeR_Results/HT_O.HU_300.LR_0005.E_20000.TT_2000.Fre.DSTR_True

rem python VOISeR.py -dir D:/VOISeR_Results -ht H -hu 300 -lr 0.0005 -e 20000 -tt 2000 -fre -dstr phonetic_feature_definitions_18_features.csv
rem python Result_Analysis.py -f D:/VOISeR_Results/HT_H.HU_300.LR_0005.E_20000.TT_2000.Fre.DSTR_True


python VOISeR.py -dir D:/VOISeR_Results -ht B -hu 300 -lr 0.0005 -e 10000 -tt 1000 -dstr phonetic_feature_definitions_18_features.csv
python Result_Analysis.py -f D:/VOISeR_Results/HT_B.HU_300.LR_0005.E_10000.TT_1000.DSTR_True
