from tabstar_paper.datasets.curation_objects import CuratedFeature, CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: madeline
====
Examples: 3140
====
URL: https://www.openml.org/search?type=data&id=41144
====
Description: The goal of this challenge is to expose the research community to real world datasets of interest to 4Paradigm. All datasets are formatted in a uniform way, though the type of data might differ. The data are provided as preprocessed matrices, so that participants can focus on classification, although participants are welcome to use additional feature extraction procedures (as long as they do not violate any rule of the challenge). All problems are binary classification problems and are assessed with the normalized Area Under the ROC Curve (AUC) metric (i.e. 2*AUC-1).
                   The identity of the datasets and the type of data is concealed, though its structure is revealed. The final score in  phase 2 will be the average of rankings  on all testing datasets, a ranking will be generated from such results, and winners will be determined according to such ranking.
                   The tasks are constrained by a time budget. The Codalab platform provides computational resources shared by all participants. Each code submission will be exceuted in a compute worker with the following characteristics: 2Cores / 8G Memory / 40G SSD with Ubuntu OS. To ensure the fairness of the evaluation, when a code submission is evaluated, its execution time is limited in time.
                   http://automl.chalearn.org/data
====
Target Variable: class (nominal, 2 distinct): ['1', '0']
====
Features:

V1 (numeric, 214 distinct): ['486.0', '499.0', '468.0', '494.0', '488.0', '502.0', '491.0', '472.0', '505.0', '475.0']
V2 (numeric, 132 distinct): ['495.0', '498.0', '488.0', '496.0', '482.0', '511.0', '500.0', '501.0', '492.0', '503.0']
V3 (numeric, 145 distinct): ['496.0', '488.0', '492.0', '489.0', '491.0', '495.0', '490.0', '484.0', '485.0', '483.0']
V4 (numeric, 127 distinct): ['492.0', '486.0', '491.0', '499.0', '496.0', '483.0', '487.0', '500.0', '489.0', '473.0']
V5 (numeric, 144 distinct): ['484.0', '486.0', '492.0', '488.0', '490.0', '475.0', '478.0', '485.0', '477.0', '482.0']
V6 (numeric, 31 distinct): ['479.0', '480.0', '477.0', '478.0', '481.0', '476.0', '482.0', '475.0', '474.0', '483.0']
V7 (numeric, 167 distinct): ['495.0', '477.0', '493.0', '490.0', '494.0', '487.0', '501.0', '496.0', '503.0', '506.0']
V8 (numeric, 80 distinct): ['484.0', '482.0', '487.0', '486.0', '488.0', '480.0', '483.0', '478.0', '493.0', '479.0']
V9 (numeric, 116 distinct): ['495.0', '492.0', '496.0', '488.0', '504.0', '493.0', '491.0', '489.0', '497.0', '490.0']
V10 (numeric, 12 distinct): ['477.0', '478.0', '476.0', '479.0', '475.0', '480.0', '474.0', '481.0', '473.0', '482.0']
V11 (numeric, 189 distinct): ['504.0', '503.0', '483.0', '494.0', '492.0', '502.0', '518.0', '497.0', '524.0', '496.0']
V12 (numeric, 114 distinct): ['484.0', '491.0', '482.0', '480.0', '483.0', '485.0', '474.0', '492.0', '487.0', '477.0']
V13 (numeric, 93 distinct): ['478.0', '477.0', '473.0', '481.0', '476.0', '483.0', '471.0', '480.0', '482.0', '484.0']
V14 (numeric, 170 distinct): ['492.0', '502.0', '507.0', '484.0', '486.0', '503.0', '504.0', '505.0', '488.0', '491.0']
V15 (numeric, 125 distinct): ['493.0', '489.0', '496.0', '508.0', '494.0', '499.0', '480.0', '484.0', '485.0', '500.0']
V16 (numeric, 155 distinct): ['484.0', '502.0', '481.0', '488.0', '493.0', '495.0', '489.0', '498.0', '496.0', '497.0']
V17 (numeric, 46 distinct): ['480.0', '478.0', '482.0', '479.0', '481.0', '483.0', '477.0', '476.0', '475.0', '484.0']
V18 (numeric, 139 distinct): ['476.0', '492.0', '479.0', '489.0', '483.0', '481.0', '480.0', '471.0', '486.0', '485.0']
V19 (numeric, 195 distinct): ['496.0', '511.0', '512.0', '495.0', '524.0', '498.0', '500.0', '523.0', '526.0', '505.0']
V20 (numeric, 141 distinct): ['492.0', '487.0', '474.0', '481.0', '482.0', '485.0', '476.0', '477.0', '490.0', '494.0']
V21 (numeric, 173 distinct): ['504.0', '501.0', '505.0', '514.0', '508.0', '481.0', '488.0', '496.0', '507.0', '495.0']
V22 (numeric, 8 distinct): ['476.0', '477.0', '475.0', '478.0', '474.0', '479.0', '473.0', '480.0']
V23 (numeric, 106 distinct): ['485.0', '481.0', '482.0', '480.0', '484.0', '477.0', '478.0', '471.0', '475.0', '470.0']
V24 (numeric, 67 distinct): ['477.0', '479.0', '476.0', '480.0', '478.0', '482.0', '484.0', '475.0', '486.0', '483.0']
V25 (numeric, 40 distinct): ['480.0', '478.0', '477.0', '481.0', '482.0', '479.0', '476.0', '483.0', '484.0', '475.0']
V26 (numeric, 169 distinct): ['506.0', '505.0', '487.0', '504.0', '497.0', '493.0', '492.0', '496.0', '494.0', '502.0']
V27 (numeric, 242 distinct): ['500.0', '516.0', '515.0', '528.0', '497.0', '510.0', '522.0', '492.0', '506.0', '520.0']
V28 (numeric, 144 distinct): ['475.0', '482.0', '473.0', '483.0', '488.0', '485.0', '486.0', '479.0', '480.0', '465.0']
V29 (numeric, 241 distinct): ['513.0', '517.0', '510.0', '509.0', '535.0', '529.0', '507.0', '502.0', '508.0', '494.0']
V30 (numeric, 95 distinct): ['484.0', '481.0', '487.0', '489.0', '488.0', '491.0', '494.0', '490.0', '492.0', '493.0']
V31 (numeric, 228 distinct): ['498.0', '510.0', '515.0', '483.0', '489.0', '506.0', '511.0', '508.0', '507.0', '485.0']
V32 (numeric, 174 distinct): ['470.0', '493.0', '486.0', '494.0', '480.0', '484.0', '481.0', '477.0', '482.0', '496.0']
V33 (numeric, 220 distinct): ['493.0', '501.0', '488.0', '478.0', '482.0', '492.0', '503.0', '486.0', '487.0', '477.0']
V34 (numeric, 125 distinct): ['488.0', '494.0', '489.0', '485.0', '492.0', '491.0', '500.0', '495.0', '497.0', '501.0']
V35 (numeric, 207 distinct): ['486.0', '469.0', '488.0', '472.0', '480.0', '492.0', '468.0', '498.0', '473.0', '484.0']
V36 (numeric, 208 distinct): ['499.0', '506.0', '493.0', '510.0', '498.0', '512.0', '505.0', '492.0', '523.0', '511.0']
V37 (numeric, 75 distinct): ['484.0', '486.0', '487.0', '488.0', '490.0', '482.0', '481.0', '493.0', '492.0', '489.0']
V38 (numeric, 54 distinct): ['469.0', '470.0', '471.0', '486.0', '472.0', '484.0', '467.0', '485.0', '468.0', '482.0']
V39 (numeric, 77 distinct): ['474.0', '476.0', '481.0', '475.0', '480.0', '479.0', '472.0', '478.0', '471.0', '483.0']
V40 (numeric, 109 distinct): ['483.0', '482.0', '486.0', '473.0', '476.0', '477.0', '484.0', '479.0', '485.0', '480.0']
V41 (numeric, 210 distinct): ['491.0', '477.0', '512.0', '487.0', '495.0', '489.0', '493.0', '492.0', '504.0', '510.0']
V42 (numeric, 236 distinct): ['491.0', '489.0', '506.0', '501.0', '476.0', '493.0', '454.0', '468.0', '484.0', '497.0']
V43 (numeric, 78 distinct): ['478.0', '475.0', '481.0', '484.0', '479.0', '483.0', '480.0', '477.0', '474.0', '482.0']
V44 (numeric, 223 distinct): ['519.0', '518.0', '501.0', '508.0', '504.0', '510.0', '507.0', '500.0', '514.0', '499.0']
V45 (numeric, 207 distinct): ['491.0', '488.0', '504.0', '493.0', '494.0', '497.0', '513.0', '496.0', '505.0', '511.0']
V46 (numeric, 131 distinct): ['481.0', '489.0', '480.0', '478.0', '469.0', '477.0', '475.0', '474.0', '486.0', '494.0']
V47 (numeric, 105 distinct): ['473.0', '475.0', '480.0', '486.0', '479.0', '482.0', '468.0', '474.0', '484.0', '471.0']
V48 (numeric, 88 distinct): ['489.0', '490.0', '485.0', '482.0', '478.0', '487.0', '486.0', '495.0', '484.0', '491.0']
V49 (numeric, 191 distinct): ['490.0', '503.0', '501.0', '505.0', '507.0', '492.0', '497.0', '496.0', '502.0', '515.0']
V50 (numeric, 50 distinct): ['478.0', '477.0', '479.0', '481.0', '483.0', '476.0', '480.0', '475.0', '474.0', '473.0']
V51 (numeric, 242 distinct): ['507.0', '521.0', '497.0', '501.0', '476.0', '518.0', '523.0', '502.0', '527.0', '512.0']
V52 (numeric, 60 distinct): ['477.0', '479.0', '482.0', '480.0', '483.0', '485.0', '484.0', '478.0', '481.0', '476.0']
V53 (numeric, 271 distinct): ['549.0', '536.0', '553.0', '554.0', '449.0', '539.0', '444.0', '461.0', '450.0', '537.0']
V54 (numeric, 33 distinct): ['476.0', '477.0', '478.0', '475.0', '479.0', '480.0', '473.0', '474.0', '481.0', '472.0']
V55 (numeric, 214 distinct): ['495.0', '497.0', '509.0', '487.0', '486.0', '499.0', '483.0', '498.0', '506.0', '504.0']
V56 (numeric, 214 distinct): ['507.0', '512.0', '518.0', '527.0', '531.0', '508.0', '516.0', '525.0', '526.0', '533.0']
V57 (numeric, 92 distinct): ['488.0', '489.0', '487.0', '485.0', '496.0', '497.0', '493.0', '491.0', '498.0', '482.0']
V58 (numeric, 134 distinct): ['500.0', '502.0', '503.0', '494.0', '489.0', '501.0', '495.0', '498.0', '482.0', '486.0']
V59 (numeric, 183 distinct): ['492.0', '481.0', '479.0', '472.0', '485.0', '484.0', '491.0', '483.0', '482.0', '496.0']
V60 (numeric, 30 distinct): ['479.0', '480.0', '481.0', '478.0', '477.0', '482.0', '476.0', '475.0', '483.0', '474.0']
V61 (numeric, 146 distinct): ['487.0', '486.0', '478.0', '476.0', '488.0', '481.0', '477.0', '479.0', '474.0', '483.0']
V62 (numeric, 168 distinct): ['482.0', '496.0', '498.0', '481.0', '485.0', '490.0', '488.0', '475.0', '484.0', '495.0']
V63 (numeric, 216 distinct): ['488.0', '492.0', '484.0', '497.0', '516.0', '490.0', '493.0', '469.0', '499.0', '521.0']
V64 (numeric, 80 distinct): ['481.0', '485.0', '488.0', '479.0', '480.0', '487.0', '482.0', '477.0', '491.0', '484.0']
V65 (numeric, 240 distinct): ['497.0', '489.0', '479.0', '505.0', '496.0', '504.0', '490.0', '468.0', '508.0', '488.0']
V66 (numeric, 192 distinct): ['495.0', '494.0', '490.0', '510.0', '492.0', '501.0', '506.0', '499.0', '505.0', '489.0']
V67 (numeric, 213 distinct): ['485.0', '498.0', '516.0', '503.0', '513.0', '525.0', '521.0', '510.0', '502.0', '487.0']
V68 (numeric, 145 distinct): ['496.0', '492.0', '500.0', '493.0', '501.0', '491.0', '494.0', '499.0', '513.0', '489.0']
V69 (numeric, 175 distinct): ['494.0', '493.0', '501.0', '478.0', '490.0', '509.0', '508.0', '496.0', '492.0', '503.0']
V70 (numeric, 166 distinct): ['489.0', '475.0', '480.0', '477.0', '481.0', '493.0', '490.0', '473.0', '486.0', '476.0']
V71 (numeric, 232 distinct): ['435.0', '428.0', '533.0', '519.0', '516.0', '518.0', '532.0', '429.0', '432.0', '512.0']
V72 (numeric, 214 distinct): ['501.0', '518.0', '504.0', '514.0', '526.0', '506.0', '490.0', '498.0', '539.0', '530.0']
V73 (numeric, 78 distinct): ['488.0', '484.0', '486.0', '487.0', '483.0', '482.0', '485.0', '479.0', '480.0', '490.0']
V74 (numeric, 161 distinct): ['497.0', '495.0', '498.0', '472.0', '488.0', '493.0', '494.0', '500.0', '501.0', '507.0']
V75 (numeric, 221 distinct): ['443.0', '450.0', '444.0', '520.0', '449.0', '531.0', '454.0', '521.0', '440.0', '533.0']
V76 (numeric, 45 distinct): ['478.0', '479.0', '477.0', '481.0', '482.0', '480.0', '476.0', '483.0', '484.0', '475.0']
V77 (numeric, 236 distinct): ['514.0', '506.0', '531.0', '515.0', '542.0', '533.0', '504.0', '524.0', '482.0', '494.0']
V78 (numeric, 136 distinct): ['478.0', '482.0', '473.0', '487.0', '484.0', '469.0', '477.0', '467.0', '471.0', '468.0']
V79 (numeric, 22 distinct): ['479.0', '478.0', '477.0', '476.0', '480.0', '475.0', '481.0', '482.0', '474.0', '473.0']
V80 (numeric, 53 distinct): ['482.0', '483.0', '484.0', '481.0', '480.0', '479.0', '477.0', '485.0', '478.0', '486.0']
V81 (numeric, 48 distinct): ['481.0', '480.0', '482.0', '477.0', '483.0', '484.0', '479.0', '485.0', '478.0', '486.0']
V82 (numeric, 154 distinct): ['500.0', '488.0', '495.0', '493.0', '506.0', '491.0', '492.0', '502.0', '501.0', '499.0']
V83 (numeric, 176 distinct): ['500.0', '491.0', '499.0', '497.0', '485.0', '495.0', '478.0', '498.0', '494.0', '496.0']
V84 (numeric, 425 distinct): ['408.0', '578.0', '570.0', '413.0', '422.0', '409.0', '563.0', '591.0', '434.0', '399.0']
V85 (numeric, 106 distinct): ['486.0', '494.0', '495.0', '484.0', '488.0', '491.0', '493.0', '490.0', '482.0', '483.0']
V86 (numeric, 454 distinct): ['507.0', '528.0', '518.0', '483.0', '506.0', '485.0', '512.0', '500.0', '459.0', '511.0']
V87 (numeric, 54 distinct): ['479.0', '478.0', '481.0', '480.0', '477.0', '482.0', '476.0', '475.0', '483.0', '473.0']
V88 (numeric, 77 distinct): ['479.0', '484.0', '483.0', '478.0', '482.0', '475.0', '476.0', '480.0', '486.0', '481.0']
V89 (numeric, 132 distinct): ['481.0', '485.0', '476.0', '480.0', '487.0', '484.0', '471.0', '470.0', '473.0', '468.0']
V90 (numeric, 42 distinct): ['476.0', '478.0', '480.0', '479.0', '477.0', '475.0', '481.0', '482.0', '473.0', '474.0']
V91 (numeric, 190 distinct): ['502.0', '489.0', '506.0', '495.0', '498.0', '509.0', '484.0', '490.0', '473.0', '497.0']
V92 (numeric, 179 distinct): ['494.0', '488.0', '493.0', '485.0', '503.0', '489.0', '487.0', '475.0', '483.0', '496.0']
V93 (numeric, 123 distinct): ['500.0', '489.0', '482.0', '477.0', '490.0', '488.0', '492.0', '494.0', '495.0', '493.0']
V94 (numeric, 207 distinct): ['489.0', '486.0', '481.0', '470.0', '501.0', '491.0', '472.0', '477.0', '497.0', '482.0']
V95 (numeric, 134 distinct): ['501.0', '490.0', '504.0', '483.0', '505.0', '508.0', '500.0', '493.0', '502.0', '506.0']
V96 (numeric, 149 distinct): ['497.0', '493.0', '472.0', '492.0', '484.0', '488.0', '486.0', '485.0', '491.0', '501.0']
V97 (numeric, 121 distinct): ['477.0', '489.0', '487.0', '486.0', '474.0', '483.0', '480.0', '479.0', '478.0', '481.0']
V98 (numeric, 103 distinct): ['489.0', '479.0', '490.0', '482.0', '487.0', '484.0', '474.0', '481.0', '477.0', '491.0']
V99 (numeric, 37 distinct): ['483.0', '480.0', '479.0', '482.0', '481.0', '478.0', '484.0', '476.0', '485.0', '477.0']
V100 (numeric, 232 distinct): ['514.0', '486.0', '500.0', '498.0', '507.0', '499.0', '487.0', '521.0', '496.0', '495.0']
V101 (numeric, 51 distinct): ['477.0', '481.0', '476.0', '483.0', '480.0', '479.0', '478.0', '482.0', '474.0', '486.0']
V102 (numeric, 146 distinct): ['486.0', '487.0', '479.0', '495.0', '490.0', '485.0', '474.0', '482.0', '480.0', '478.0']
V103 (numeric, 244 distinct): ['510.0', '503.0', '513.0', '517.0', '507.0', '502.0', '490.0', '494.0', '492.0', '491.0']
V104 (numeric, 38 distinct): ['481.0', '480.0', '482.0', '477.0', '479.0', '478.0', '476.0', '484.0', '483.0', '475.0']
V105 (numeric, 236 distinct): ['488.0', '500.0', '493.0', '509.0', '505.0', '501.0', '489.0', '497.0', '503.0', '495.0']
V106 (numeric, 217 distinct): ['517.0', '496.0', '489.0', '506.0', '493.0', '480.0', '497.0', '509.0', '507.0', '508.0']
V107 (numeric, 239 distinct): ['494.0', '483.0', '507.0', '473.0', '481.0', '492.0', '485.0', '496.0', '484.0', '476.0']
V108 (numeric, 223 distinct): ['496.0', '510.0', '505.0', '516.0', '485.0', '504.0', '507.0', '499.0', '522.0', '495.0']
V109 (numeric, 158 distinct): ['493.0', '494.0', '495.0', '486.0', '496.0', '479.0', '482.0', '491.0', '499.0', '500.0']
V110 (numeric, 20 distinct): ['476.0', '477.0', '475.0', '478.0', '474.0', '479.0', '473.0', '480.0', '481.0', '472.0']
V111 (numeric, 89 distinct): ['482.0', '486.0', '478.0', '484.0', '483.0', '477.0', '487.0', '485.0', '481.0', '488.0']
V112 (numeric, 212 distinct): ['480.0', '486.0', '487.0', '494.0', '491.0', '505.0', '495.0', '483.0', '489.0', '463.0']
V113 (numeric, 50 distinct): ['481.0', '482.0', '483.0', '479.0', '477.0', '480.0', '478.0', '485.0', '476.0', '475.0']
V114 (numeric, 235 distinct): ['513.0', '489.0', '495.0', '503.0', '496.0', '510.0', '493.0', '515.0', '499.0', '485.0']
V115 (numeric, 133 distinct): ['496.0', '494.0', '497.0', '503.0', '492.0', '485.0', '500.0', '488.0', '495.0', '490.0']
V116 (numeric, 81 distinct): ['481.0', '478.0', '482.0', '483.0', '485.0', '477.0', '486.0', '488.0', '484.0', '480.0']
V117 (numeric, 228 distinct): ['509.0', '505.0', '504.0', '516.0', '501.0', '494.0', '497.0', '511.0', '523.0', '502.0']
V118 (numeric, 87 distinct): ['483.0', '478.0', '476.0', '475.0', '480.0', '485.0', '482.0', '481.0', '477.0', '484.0']
V119 (numeric, 166 distinct): ['479.0', '478.0', '480.0', '489.0', '465.0', '476.0', '482.0', '472.0', '487.0', '491.0']
V120 (numeric, 236 distinct): ['471.0', '466.0', '478.0', '498.0', '477.0', '468.0', '485.0', '484.0', '453.0', '493.0']
V121 (numeric, 204 distinct): ['506.0', '496.0', '491.0', '487.0', '484.0', '525.0', '521.0', '497.0', '486.0', '488.0']
V122 (numeric, 134 distinct): ['483.0', '489.0', '485.0', '482.0', '494.0', '488.0', '491.0', '479.0', '498.0', '495.0']
V123 (numeric, 116 distinct): ['497.0', '493.0', '483.0', '492.0', '482.0', '494.0', '488.0', '496.0', '491.0', '489.0']
V124 (numeric, 73 distinct): ['477.0', '479.0', '481.0', '476.0', '474.0', '480.0', '483.0', '473.0', '482.0', '478.0']
V125 (numeric, 232 distinct): ['480.0', '492.0', '477.0', '495.0', '490.0', '505.0', '500.0', '519.0', '493.0', '526.0']
V126 (numeric, 25 distinct): ['476.0', '478.0', '477.0', '475.0', '474.0', '479.0', '480.0', '473.0', '472.0', '481.0']
V127 (numeric, 145 distinct): ['482.0', '477.0', '467.0', '486.0', '471.0', '480.0', '492.0', '478.0', '472.0', '496.0']
V128 (numeric, 233 distinct): ['497.0', '484.0', '501.0', '480.0', '516.0', '504.0', '507.0', '489.0', '462.0', '468.0']
V129 (numeric, 69 distinct): ['478.0', '481.0', '484.0', '477.0', '485.0', '482.0', '480.0', '483.0', '475.0', '486.0']
V130 (numeric, 223 distinct): ['513.0', '502.0', '490.0', '495.0', '483.0', '488.0', '478.0', '500.0', '512.0', '504.0']
V131 (numeric, 10 distinct): ['477.0', '476.0', '478.0', '475.0', '479.0', '480.0', '474.0', '473.0', '481.0', '472.0']
V132 (numeric, 33 distinct): ['479.0', '478.0', '480.0', '481.0', '477.0', '476.0', '475.0', '482.0', '483.0', '474.0']
V133 (numeric, 175 distinct): ['499.0', '490.0', '501.0', '503.0', '495.0', '483.0', '486.0', '494.0', '487.0', '497.0']
V134 (numeric, 170 distinct): ['478.0', '490.0', '475.0', '497.0', '480.0', '485.0', '472.0', '498.0', '469.0', '495.0']
V135 (numeric, 31 distinct): ['478.0', '477.0', '479.0', '475.0', '476.0', '480.0', '474.0', '473.0', '481.0', '482.0']
V136 (numeric, 207 distinct): ['524.0', '509.0', '447.0', '453.0', '460.0', '528.0', '527.0', '457.0', '448.0', '525.0']
V137 (numeric, 204 distinct): ['491.0', '512.0', '492.0', '509.0', '501.0', '485.0', '486.0', '497.0', '507.0', '484.0']
V138 (numeric, 125 distinct): ['477.0', '480.0', '488.0', '496.0', '482.0', '492.0', '486.0', '494.0', '476.0', '487.0']
V139 (numeric, 12 distinct): ['476.0', '477.0', '475.0', '478.0', '474.0', '479.0', '473.0', '472.0', '480.0', '471.0']
V140 (numeric, 198 distinct): ['468.0', '490.0', '470.0', '484.0', '472.0', '493.0', '469.0', '476.0', '471.0', '473.0']
V141 (numeric, 125 distinct): ['478.0', '486.0', '489.0', '483.0', '480.0', '490.0', '473.0', '491.0', '487.0', '479.0']
V142 (numeric, 45 distinct): ['479.0', '480.0', '478.0', '481.0', '483.0', '477.0', '476.0', '482.0', '486.0', '484.0']
V143 (numeric, 16 distinct): ['476.0', '477.0', '475.0', '478.0', '474.0', '479.0', '473.0', '480.0', '472.0', '471.0']
V144 (numeric, 200 distinct): ['498.0', '514.0', '489.0', '515.0', '496.0', '520.0', '504.0', '513.0', '503.0', '505.0']
V145 (numeric, 84 distinct): ['484.0', '486.0', '481.0', '479.0', '487.0', '480.0', '489.0', '488.0', '482.0', '477.0']
V146 (numeric, 220 distinct): ['495.0', '484.0', '505.0', '500.0', '526.0', '541.0', '514.0', '501.0', '494.0', '518.0']
V147 (numeric, 115 distinct): ['492.0', '489.0', '499.0', '482.0', '487.0', '491.0', '493.0', '495.0', '506.0', '483.0']
V148 (numeric, 48 distinct): ['480.0', '483.0', '481.0', '478.0', '482.0', '479.0', '485.0', '476.0', '477.0', '484.0']
V149 (numeric, 229 distinct): ['500.0', '503.0', '494.0', '481.0', '489.0', '502.0', '474.0', '525.0', '477.0', '485.0']
V150 (numeric, 75 distinct): ['483.0', '479.0', '481.0', '485.0', '477.0', '488.0', '476.0', '473.0', '484.0', '478.0']
V151 (numeric, 116 distinct): ['493.0', '490.0', '498.0', '501.0', '485.0', '492.0', '482.0', '488.0', '483.0', '478.0']
V152 (numeric, 30 distinct): ['477.0', '476.0', '478.0', '475.0', '479.0', '480.0', '474.0', '481.0', '473.0', '482.0']
V153 (numeric, 69 distinct): ['487.0', '481.0', '480.0', '489.0', '483.0', '485.0', '482.0', '484.0', '490.0', '479.0']
V154 (numeric, 99 distinct): ['485.0', '487.0', '490.0', '489.0', '486.0', '483.0', '488.0', '495.0', '496.0', '494.0']
V155 (numeric, 57 distinct): ['483.0', '480.0', '486.0', '478.0', '481.0', '484.0', '482.0', '485.0', '487.0', '479.0']
V156 (numeric, 9 distinct): ['477.0', '476.0', '478.0', '475.0', '479.0', '474.0', '480.0', '481.0', '473.0']
V157 (numeric, 60 distinct): ['481.0', '485.0', '482.0', '483.0', '486.0', '487.0', '479.0', '480.0', '488.0', '484.0']
V158 (numeric, 50 distinct): ['482.0', '484.0', '480.0', '481.0', '485.0', '479.0', '486.0', '483.0', '478.0', '477.0']
V159 (numeric, 232 distinct): ['468.0', '472.0', '489.0', '493.0', '485.0', '478.0', '497.0', '492.0', '487.0', '477.0']
V160 (numeric, 93 distinct): ['490.0', '492.0', '484.0', '489.0', '487.0', '494.0', '483.0', '491.0', '479.0', '486.0']
V161 (numeric, 122 distinct): ['483.0', '492.0', '490.0', '486.0', '484.0', '487.0', '498.0', '482.0', '499.0', '497.0']
V162 (numeric, 45 distinct): ['477.0', '478.0', '480.0', '482.0', '479.0', '476.0', '475.0', '481.0', '484.0', '473.0']
V163 (numeric, 462 distinct): ['485.0', '531.0', '504.0', '411.0', '472.0', '509.0', '539.0', '467.0', '468.0', '494.0']
V164 (numeric, 352 distinct): ['561.0', '444.0', '550.0', '423.0', '580.0', '459.0', '573.0', '455.0', '461.0', '436.0']
V165 (numeric, 102 distinct): ['489.0', '488.0', '490.0', '491.0', '495.0', '485.0', '484.0', '482.0', '477.0', '479.0']
V166 (numeric, 77 distinct): ['485.0', '481.0', '479.0', '480.0', '482.0', '486.0', '488.0', '487.0', '475.0', '483.0']
V167 (numeric, 202 distinct): ['505.0', '498.0', '495.0', '507.0', '486.0', '501.0', '487.0', '496.0', '494.0', '520.0']
V168 (numeric, 50 distinct): ['481.0', '484.0', '482.0', '483.0', '480.0', '478.0', '485.0', '477.0', '479.0', '486.0']
V169 (numeric, 43 distinct): ['483.0', '481.0', '480.0', '484.0', '482.0', '479.0', '478.0', '477.0', '485.0', '476.0']
V170 (numeric, 61 distinct): ['481.0', '484.0', '477.0', '478.0', '482.0', '480.0', '483.0', '485.0', '476.0', '475.0']
V171 (numeric, 192 distinct): ['484.0', '506.0', '495.0', '500.0', '503.0', '497.0', '480.0', '507.0', '488.0', '494.0']
V172 (numeric, 64 distinct): ['479.0', '477.0', '480.0', '473.0', '476.0', '472.0', '478.0', '471.0', '475.0', '474.0']
V173 (numeric, 134 distinct): ['498.0', '496.0', '487.0', '501.0', '483.0', '488.0', '503.0', '485.0', '493.0', '486.0']
V174 (numeric, 225 distinct): ['486.0', '469.0', '474.0', '485.0', '472.0', '480.0', '476.0', '465.0', '470.0', '488.0']
V175 (numeric, 74 distinct): ['473.0', '476.0', '482.0', '478.0', '475.0', '471.0', '481.0', '470.0', '484.0', '474.0']
V176 (numeric, 143 distinct): ['497.0', '486.0', '493.0', '492.0', '500.0', '487.0', '489.0', '498.0', '496.0', '480.0']
V177 (numeric, 69 distinct): ['474.0', '481.0', '475.0', '479.0', '478.0', '477.0', '470.0', '473.0', '480.0', '483.0']
V178 (numeric, 159 distinct): ['509.0', '485.0', '504.0', '483.0', '516.0', '491.0', '497.0', '513.0', '498.0', '481.0']
V179 (numeric, 496 distinct): ['552.0', '492.0', '524.0', '467.0', '415.0', '476.0', '431.0', '469.0', '428.0', '579.0']
V180 (numeric, 193 distinct): ['509.0', '494.0', '510.0', '518.0', '491.0', '498.0', '489.0', '488.0', '503.0', '512.0']
V181 (numeric, 87 distinct): ['481.0', '477.0', '483.0', '479.0', '475.0', '478.0', '487.0', '484.0', '492.0', '485.0']
V182 (numeric, 11 distinct): ['477.0', '476.0', '478.0', '475.0', '479.0', '474.0', '480.0', '481.0', '473.0', '482.0']
V183 (numeric, 143 distinct): ['493.0', '497.0', '483.0', '495.0', '490.0', '496.0', '486.0', '502.0', '508.0', '498.0']
V184 (numeric, 610 distinct): ['419.0', '430.0', '541.0', '502.0', '489.0', '560.0', '417.0', '454.0', '518.0', '602.0']
V185 (numeric, 73 distinct): ['476.0', '475.0', '478.0', '480.0', '479.0', '471.0', '485.0', '473.0', '481.0', '488.0']
V186 (numeric, 231 distinct): ['480.0', '495.0', '487.0', '472.0', '484.0', '475.0', '468.0', '500.0', '469.0', '493.0']
V187 (numeric, 99 distinct): ['484.0', '482.0', '486.0', '477.0', '483.0', '485.0', '479.0', '487.0', '478.0', '490.0']
V188 (numeric, 112 distinct): ['490.0', '488.0', '492.0', '485.0', '479.0', '493.0', '483.0', '476.0', '487.0', '482.0']
V189 (numeric, 104 distinct): ['488.0', '494.0', '487.0', '482.0', '491.0', '489.0', '493.0', '490.0', '495.0', '484.0']
V190 (numeric, 200 distinct): ['476.0', '494.0', '482.0', '480.0', '484.0', '488.0', '483.0', '481.0', '471.0', '475.0']
V191 (numeric, 178 distinct): ['483.0', '485.0', '492.0', '482.0', '489.0', '498.0', '491.0', '484.0', '493.0', '487.0']
V192 (numeric, 343 distinct): ['431.0', '430.0', '413.0', '554.0', '536.0', '540.0', '535.0', '561.0', '424.0', '386.0']
V193 (numeric, 36 distinct): ['479.0', '480.0', '478.0', '481.0', '482.0', '477.0', '483.0', '484.0', '476.0', '485.0']
V194 (numeric, 133 distinct): ['479.0', '484.0', '487.0', '480.0', '490.0', '483.0', '489.0', '486.0', '492.0', '482.0']
V195 (numeric, 40 distinct): ['481.0', '479.0', '480.0', '483.0', '482.0', '478.0', '477.0', '484.0', '485.0', '476.0']
V196 (numeric, 244 distinct): ['508.0', '505.0', '520.0', '504.0', '513.0', '522.0', '501.0', '498.0', '525.0', '530.0']
V197 (numeric, 125 distinct): ['483.0', '477.0', '469.0', '468.0', '479.0', '481.0', '473.0', '485.0', '474.0', '484.0']
V198 (numeric, 525 distinct): ['590.0', '545.0', '524.0', '425.0', '562.0', '521.0', '541.0', '451.0', '556.0', '582.0']
V199 (numeric, 62 distinct): ['482.0', '483.0', '486.0', '481.0', '485.0', '488.0', '489.0', '484.0', '480.0', '487.0']
V200 (numeric, 115 distinct): ['478.0', '483.0', '486.0', '485.0', '484.0', '491.0', '476.0', '481.0', '490.0', '487.0']
V201 (numeric, 202 distinct): ['501.0', '508.0', '500.0', '492.0', '506.0', '513.0', '483.0', '494.0', '496.0', '484.0']
V202 (numeric, 73 distinct): ['483.0', '479.0', '482.0', '476.0', '484.0', '481.0', '485.0', '486.0', '471.0', '478.0']
V203 (numeric, 198 distinct): ['509.0', '512.0', '504.0', '508.0', '519.0', '506.0', '502.0', '489.0', '500.0', '482.0']
V204 (numeric, 154 distinct): ['473.0', '487.0', '480.0', '481.0', '486.0', '488.0', '483.0', '471.0', '478.0', '469.0']
V205 (numeric, 38 distinct): ['481.0', '480.0', '483.0', '479.0', '484.0', '478.0', '482.0', '477.0', '485.0', '486.0']
V206 (numeric, 30 distinct): ['479.0', '480.0', '478.0', '477.0', '481.0', '476.0', '482.0', '483.0', '475.0', '474.0']
V207 (numeric, 169 distinct): ['492.0', '481.0', '491.0', '501.0', '486.0', '484.0', '500.0', '507.0', '496.0', '495.0']
V208 (numeric, 72 distinct): ['479.0', '483.0', '481.0', '482.0', '484.0', '480.0', '474.0', '476.0', '485.0', '473.0']
V209 (numeric, 82 distinct): ['484.0', '486.0', '489.0', '485.0', '487.0', '481.0', '492.0', '494.0', '482.0', '483.0']
V210 (numeric, 127 distinct): ['477.0', '481.0', '488.0', '476.0', '490.0', '472.0', '489.0', '468.0', '487.0', '484.0']
V211 (numeric, 217 distinct): ['475.0', '491.0', '470.0', '485.0', '474.0', '482.0', '486.0', '467.0', '459.0', '498.0']
V212 (numeric, 128 distinct): ['485.0', '496.0', '497.0', '481.0', '482.0', '487.0', '483.0', '494.0', '489.0', '492.0']
V213 (numeric, 244 distinct): ['481.0', '496.0', '510.0', '489.0', '502.0', '501.0', '495.0', '478.0', '504.0', '518.0']
V214 (numeric, 92 distinct): ['478.0', '480.0', '477.0', '487.0', '481.0', '472.0', '486.0', '489.0', '483.0', '485.0']
V215 (numeric, 143 distinct): ['486.0', '476.0', '479.0', '491.0', '465.0', '487.0', '467.0', '478.0', '483.0', '481.0']
V216 (numeric, 153 distinct): ['497.0', '494.0', '496.0', '493.0', '479.0', '487.0', '474.0', '495.0', '502.0', '490.0']
V217 (numeric, 243 distinct): ['504.0', '519.0', '497.0', '508.0', '527.0', '475.0', '494.0', '481.0', '487.0', '501.0']
V218 (numeric, 229 distinct): ['494.0', '478.0', '488.0', '492.0', '480.0', '479.0', '485.0', '491.0', '477.0', '465.0']
V219 (numeric, 203 distinct): ['498.0', '499.0', '508.0', '493.0', '513.0', '500.0', '484.0', '525.0', '496.0', '512.0']
V220 (numeric, 181 distinct): ['487.0', '501.0', '492.0', '497.0', '489.0', '479.0', '511.0', '480.0', '502.0', '477.0']
V221 (numeric, 9 distinct): ['477.0', '476.0', '478.0', '479.0', '475.0', '480.0', '474.0', '481.0', '473.0']
V222 (numeric, 224 distinct): ['501.0', '479.0', '492.0', '471.0', '494.0', '488.0', '497.0', '491.0', '485.0', '505.0']
V223 (numeric, 119 distinct): ['497.0', '508.0', '499.0', '491.0', '489.0', '501.0', '493.0', '480.0', '502.0', '490.0']
V224 (numeric, 208 distinct): ['500.0', '507.0', '505.0', '509.0', '517.0', '503.0', '506.0', '510.0', '514.0', '494.0']
V225 (numeric, 116 distinct): ['494.0', '486.0', '495.0', '485.0', '479.0', '489.0', '493.0', '490.0', '500.0', '488.0']
V226 (numeric, 134 distinct): ['482.0', '479.0', '471.0', '478.0', '472.0', '483.0', '477.0', '470.0', '488.0', '481.0']
V227 (numeric, 233 distinct): ['505.0', '489.0', '474.0', '475.0', '476.0', '485.0', '480.0', '491.0', '484.0', '492.0']
V228 (numeric, 65 distinct): ['474.0', '479.0', '481.0', '477.0', '476.0', '478.0', '482.0', '483.0', '475.0', '480.0']
V229 (numeric, 251 distinct): ['499.0', '502.0', '521.0', '510.0', '517.0', '498.0', '507.0', '505.0', '492.0', '501.0']
V230 (numeric, 240 distinct): ['500.0', '514.0', '487.0', '501.0', '512.0', '488.0', '516.0', '483.0', '529.0', '478.0']
V231 (numeric, 120 distinct): ['488.0', '495.0', '493.0', '492.0', '494.0', '490.0', '498.0', '484.0', '502.0', '489.0']
V232 (numeric, 219 distinct): ['490.0', '489.0', '500.0', '486.0', '482.0', '485.0', '455.0', '494.0', '467.0', '503.0']
V233 (numeric, 309 distinct): ['435.0', '571.0', '428.0', '557.0', '422.0', '440.0', '430.0', '556.0', '549.0', '540.0']
V234 (numeric, 182 distinct): ['494.0', '502.0', '492.0', '493.0', '471.0', '483.0', '489.0', '473.0', '506.0', '505.0']
V235 (numeric, 71 distinct): ['479.0', '482.0', '481.0', '480.0', '485.0', '483.0', '477.0', '484.0', '478.0', '486.0']
V236 (numeric, 99 distinct): ['480.0', '474.0', '475.0', '468.0', '477.0', '478.0', '473.0', '481.0', '472.0', '476.0']
V237 (numeric, 95 distinct): ['486.0', '485.0', '493.0', '491.0', '494.0', '484.0', '483.0', '489.0', '487.0', '498.0']
V238 (numeric, 255 distinct): ['454.0', '449.0', '450.0', '538.0', '531.0', '446.0', '534.0', '532.0', '462.0', '546.0']
V239 (numeric, 203 distinct): ['495.0', '505.0', '487.0', '469.0', '483.0', '481.0', '500.0', '479.0', '490.0', '491.0']
V240 (numeric, 78 distinct): ['467.0', '468.0', '471.0', '466.0', '492.0', '469.0', '464.0', '494.0', '470.0', '493.0']
V241 (numeric, 83 distinct): ['486.0', '488.0', '482.0', '479.0', '485.0', '483.0', '491.0', '477.0', '487.0', '484.0']
V242 (numeric, 85 distinct): ['488.0', '480.0', '483.0', '482.0', '485.0', '487.0', '479.0', '491.0', '481.0', '486.0']
V243 (numeric, 208 distinct): ['487.0', '496.0', '492.0', '494.0', '491.0', '506.0', '493.0', '508.0', '504.0', '477.0']
V244 (numeric, 100 distinct): ['485.0', '473.0', '479.0', '476.0', '483.0', '475.0', '489.0', '482.0', '474.0', '484.0']
V245 (numeric, 223 distinct): ['505.0', '488.0', '514.0', '491.0', '518.0', '517.0', '503.0', '496.0', '504.0', '482.0']
V246 (numeric, 72 distinct): ['483.0', '486.0', '481.0', '479.0', '485.0', '488.0', '482.0', '484.0', '478.0', '477.0']
V247 (numeric, 166 distinct): ['482.0', '484.0', '480.0', '491.0', '499.0', '493.0', '486.0', '508.0', '481.0', '490.0']
V248 (numeric, 134 distinct): ['477.0', '481.0', '485.0', '494.0', '480.0', '479.0', '487.0', '486.0', '474.0', '475.0']
V249 (numeric, 89 distinct): ['484.0', '485.0', '481.0', '482.0', '487.0', '479.0', '486.0', '493.0', '477.0', '483.0']
V250 (numeric, 120 distinct): ['487.0', '486.0', '488.0', '476.0', '491.0', '479.0', '481.0', '484.0', '480.0', '494.0']
V251 (numeric, 203 distinct): ['505.0', '489.0', '498.0', '511.0', '494.0', '504.0', '509.0', '522.0', '508.0', '507.0']
V252 (numeric, 235 distinct): ['487.0', '480.0', '518.0', '499.0', '475.0', '476.0', '482.0', '492.0', '501.0', '481.0']
V253 (numeric, 438 distinct): ['475.0', '521.0', '502.0', '570.0', '503.0', '499.0', '548.0', '567.0', '426.0', '479.0']
V254 (numeric, 230 distinct): ['502.0', '489.0', '466.0', '476.0', '492.0', '500.0', '483.0', '465.0', '484.0', '510.0']
V255 (numeric, 236 distinct): ['486.0', '498.0', '491.0', '496.0', '471.0', '497.0', '477.0', '481.0', '469.0', '499.0']
V256 (numeric, 181 distinct): ['494.0', '506.0', '507.0', '513.0', '502.0', '510.0', '490.0', '488.0', '493.0', '492.0']
V257 (numeric, 220 distinct): ['497.0', '501.0', '493.0', '488.0', '499.0', '503.0', '505.0', '528.0', '506.0', '496.0']
V258 (numeric, 45 distinct): ['479.0', '482.0', '481.0', '477.0', '478.0', '480.0', '476.0', '484.0', '483.0', '475.0']
V259 (numeric, 64 distinct): ['479.0', '478.0', '477.0', '473.0', '475.0', '474.0', '476.0', '484.0', '483.0', '482.0']
'''

CONTEXT = "Anonymized Dataset: Madeline"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []