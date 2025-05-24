from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: yprop_4_1
====
Examples: 8885
====
URL: https://www.openml.org/search?type=data&id=416
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

This is one of 41 drug design datasets. The datasets with 1143 features are formed using Adriana.Code software  (www.molecular-networks.com/software/adrianacode).
The molecules and outputs are taken from the original studies (see below). The other datasets are taken exactly from the original studies. 

The last attribute in each file is the target.

Original studies:

carbolenes
"B. D. Silverman and Daniel. E. Platt, J. Med. Chem. 1996, 39, 2129-2140"

mtp2
"Bergstrom, C. A. S.; Norinder, U.; Luthman, K.; Artursson, P. Molecular Descriptors Influencing Melting Point and Their Role in Classification of Solid Drugs. J. Chem. Inf. Comput. Sci.; (Article); 2003; 43(4); 1177-1185"

chang, cristalli, depreux, doherty, garrat2, garrat, heyl, krystek, lewis, penning, rosowsky, siddiqi, stevenson, strupcz, svensson, thompson, tsutumi, uejling, yokoyama1, yokoyama2	
"David E Patterson, Richard D Cramer, Allan M Ferguson, Robert D Clark, Laurence W Weinberger. Neighbourhood Behaviour: A Useful Concept for Validation of ""Molecular Diversity"" Descriptors. J. Med. Chem. 1996 (39) 3049 - 3059."

mtp
"Karthikeyan, M.; Glen, R.C.; Bender, A. General melting point prediction based on a diverse compound dataset and artificial neural networks. J. Chem. Inf. Model.; 2005; 45(3); 581-590"

benzo32
"Harrison,P.W. and Barlin,G.B. and Davies,L.P. and Ireland,S.J. and Matyus,P. and Wong,M.G., Syntheses, pharmacological evaluation and molecular modelling of substituted 6-alkoxyimidazo[1,2-b]pyridazines as new ligands for the benzodiazepine receptor, European Journal of Medicinal Chemistry, (31), 1996, 651-662"

PHENETYL1	
"H. Kubinyi (Ed.): ""QSAR: Hansch Analysis and Related Approaches"", VCH, Weinhein (Ger), 1993, pp.57-68"

pah	
"Todeschini, R.; Gramatica, P.; Marengo, E.; Provenzani, R. Weighted Holistic  Invariant Molecular Descriptors. Part 2. Theory Development and Applications on Modeling Physico-Chemical Properties of PolyAromatic Hydrocarbons (PAH). Chemom. Intell. Lab. Syst. 1995, 27, 221-229."

pdgfr	
"R. Guha and P. Jurs. The Development of Linear, Ensemble and Non-linear Models for the Prediction and Interpretation of the Biological Activity of a Set of PDGFR Inhibitors. J. Chem. Inf. Comput. Sci. 2004, 44 (6), 2179-2189"

Phen	
"Cammarata, A. Interrelationship of the Regression Models Used for Structure-Activity Analyses. J. Med. Chem. 1972, 15, 573-577"

topo_2_1, yprop_4_1	
"Jun Feng et al, Predictive Toxicology: Benchmarking Molecular Descriptors and Statistical Methods, J. Chem. Inf Comput. Sci., 2003 (43) 1463-1470"

qsabr1, qsabr2	
"Damborsky, J., Schultz, T.W., Comparison of the QSAR models for toxicity and biodegradability of anilines and phenols, Chemosphere 34: 429-446, 1997"

qsartox	
"Blaha, L., Damborsky, J., Nemec, M., QSAR for acute toxicity of saturated and unsaturated halogenated aliphatic compounds, Chemosphere 36: 1345-1365, 1998"

qsbr_rw1	
"Damborsky, J. et al., Structure-biodegradability relationships for chlorinated dibenzo-p-dioxins and dibenzofurans, In: Wittich, R.-M., Biodegradation of dioxins and furans, R.G. Landes Company, Austin, 1998"

qsbr_y2	
"Damborsky, J. et al., A mechanistic approach to deriving QSBR- A case study: dehalogenation of haloaliphatic compounds, In: Peijnenburg, W.J.G.M., Damborsky, J., Biodegradability Prediction, Kluwer Academic Publishers"

qsbralks	
"Damborsky, J. et al., Mechanism-based Quantitative Structure-Biodegradability Relationships for hydrolytic dehalogenation of chloro- and bromo-alkenes, Quantitative Structure-Activity Relationships 17: 450-458, 1998"

qsfrdhla	
"Damborsky, J., Quantitative structure-function relationships of the single-point mutants of haloalkane dehalogenase: A multivariate approach, Qunatitative Structure-Activity Relationships 16: 126-135, 1997"

qsfsr1	
"Damborsky, J., Quantitative structure-function and structure-stability relationships of purposely modified proteins, Protein Engineering 11: 21-30, 1998"

qsfsr2	
"Damborsky, J., Quantitative structure-function and structure-stability relationships of purposely modified proteins, Protein Engineering 11: 21-30, 1998"

qsprcmpx	
"Cajan, M. et al., Stability of Aromatic Amides with Bromide Anion: Quantitative Structure-Property Relationships, Journal of Chemical Information and Computer Sciences, in press, 2000"

selwood	
"Selwood, D. L.; Livingstone, D. J.; Comley, J. C.; O'Dowd, A. B.; Hudson, A. T.; Jackson, P.; Jandu, K. S.; Rose, V. S.; Stables, J. N. Structure-Activity Relationships of Antifilarial Antimycin Analogues: A Multivariate Pattern Recognition Study J. Med. Chem., 1990, 33, 136-142"
====
Target Variable: oz252 (numeric, 1336 distinct): ['0.9099', '0.9107', '0.9089', '0.9149', '0.9103', '0.9114', '0.9107', '0.9076', '0.9086', '0.9062']
====
Features:

oz1 (numeric, 800 distinct): ['1.0', '0.9988', '0.9996', '0.9998', '0.9983', '0.9993', '0.9995', '1.0', '0.9994', '1.0']
oz2 (numeric, 26 distinct): ['0.1741', '0.1647', '0.0', '0.1176', '0.1776', '0.1706', '0.2082', '0.1188', '0.1435', '0.1576']
oz3 (numeric, 586 distinct): ['1.0', '0.9973', '0.9997', '0.9999', '0.9984', '0.9995', '0.9996', '0.9986', '0.9997', '0.998']
oz4 (numeric, 1125 distinct): ['0.9963', '0.9967', '0.9972', '0.9967', '0.9971', '0.9963', '0.9967', '0.9976', '0.9966', '0.997']
oz5 (numeric, 15 distinct): ['0.0', '0.0667', '0.1333', '0.2', '0.2667', '0.3333', '0.4', '0.4667', '0.5333', '0.6667']
oz6 (numeric, 33 distinct): ['0.0', '0.0286', '0.0571', '0.0857', '0.1143', '0.1429', '0.1714', '0.2', '0.2857', '0.2286']
oz7 (numeric, 8 distinct): ['0.0', '0.125', '0.25', '0.375', '0.5', '0.625', '0.75', '1.0']
oz8 (numeric, 8 distinct): ['0.0', '0.125', '0.25', '0.375', '0.5', '0.75', '0.875', '1.0']
oz9 (numeric, 21 distinct): ['0.0', '0.0455', '0.0909', '0.1818', '0.1364', '0.2273', '0.2727', '0.3636', '0.3182', '0.4545']
oz10 (numeric, 19 distinct): ['0.0', '0.0417', '0.0833', '0.1667', '0.125', '0.2083', '0.25', '0.2917', '0.3333', '0.4167']
oz11 (numeric, 10 distinct): ['0.0', '0.1', '0.2', '0.3', '0.4', '0.6', '0.5', '0.7', '1.0', '0.8']
oz12 (numeric, 31 distinct): ['0.0', '0.0769', '0.0962', '0.1538', '0.0577', '0.1346', '0.1731', '0.1154', '0.1923', '0.0385']
oz13 (numeric, 22 distinct): ['0.0', '0.0909', '0.1818', '0.1364', '0.2727', '0.2273', '0.0455', '0.3182', '0.3636', '0.4091']
oz14 (numeric, 8 distinct): ['0.0', '0.1429', '0.2857', '0.4286', '0.5714', '0.8571', '0.7143', '1.0']
oz15 (numeric, 4 distinct): ['0.0', '0.25', '0.5', '1.0']
oz16 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '0.75', '1.0']
oz17 (numeric, 4 distinct): ['0.0', '0.3333', '0.6667', '1.0']
oz18 (numeric, 3 distinct): ['0.0', '1.0', '0.3333']
oz19 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz20 (numeric, 5 distinct): ['0.0', '0.3333', '0.1667', '0.6667', '1.0']
oz21 (numeric, 1 distinct): ['0']
oz22 (numeric, 1 distinct): ['0']
oz23 (numeric, 1 distinct): ['0']
oz24 (numeric, 1 distinct): ['0']
oz25 (numeric, 1 distinct): ['0']
oz26 (numeric, 1 distinct): ['0']
oz27 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz28 (numeric, 1 distinct): ['0']
oz29 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '0.75', '1.0']
oz30 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '0.75', '1.0']
oz31 (numeric, 10 distinct): ['0.0', '0.0909', '0.1818', '0.3636', '0.2727', '0.4545', '0.7273', '0.6364', '0.5455', '1.0']
oz32 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '0.75', '1.0']
oz33 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz34 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz35 (numeric, 7 distinct): ['0.0', '0.1667', '0.3333', '0.5', '0.6667', '0.8333', '1.0']
oz36 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '1.0', '0.75']
oz37 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '1.0', '0.75']
oz38 (numeric, 4 distinct): ['0.0', '0.3333', '0.6667', '1.0']
oz39 (numeric, 4 distinct): ['0.0', '0.25', '0.5', '1.0']
oz40 (numeric, 2 distinct): ['0', '1']
oz41 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz42 (numeric, 2 distinct): ['0', '1']
oz43 (numeric, 1 distinct): ['0']
oz44 (numeric, 1 distinct): ['0']
oz45 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz46 (numeric, 2 distinct): ['0', '1']
oz47 (numeric, 4 distinct): ['0.0', '0.3333', '0.6667', '1.0']
oz48 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz49 (numeric, 4 distinct): ['0.0', '0.25', '0.5', '1.0']
oz50 (numeric, 2 distinct): ['0', '1']
oz51 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz52 (numeric, 1 distinct): ['0']
oz53 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '1.0', '0.75']
oz54 (numeric, 1 distinct): ['0']
oz55 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '0.75', '1.0']
oz56 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz57 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '1.0', '0.75']
oz58 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '0.75', '1.0']
oz59 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '1.0', '0.75']
oz60 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '0.75', '1.0']
oz61 (numeric, 4 distinct): ['0.0', '0.3333', '0.6667', '1.0']
oz62 (numeric, 5 distinct): ['0.25', '0.5', '0.75', '0.0', '1.0']
oz63 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz64 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz65 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz66 (numeric, 6 distinct): ['0.0', '0.1667', '0.3333', '0.5', '0.6667', '1.0']
oz67 (numeric, 1 distinct): ['0']
oz68 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz69 (numeric, 2 distinct): ['0', '1']
oz70 (numeric, 4 distinct): ['0.0', '0.3333', '0.6667', '1.0']
oz71 (numeric, 2 distinct): ['0', '1']
oz72 (numeric, 3 distinct): ['0.0', '0.3333', '1.0']
oz73 (numeric, 2 distinct): ['0', '1']
oz74 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz75 (numeric, 1 distinct): ['0']
oz76 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz77 (numeric, 1 distinct): ['0']
oz78 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz79 (numeric, 2 distinct): ['0', '1']
oz80 (numeric, 5 distinct): ['0.0', '0.25', '0.75', '0.5', '1.0']
oz81 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '1.0', '0.75']
oz82 (numeric, 4 distinct): ['0.0', '0.25', '0.5', '1.0']
oz83 (numeric, 12 distinct): ['0.0', '0.0769', '0.1538', '0.2308', '0.3077', '0.3846', '0.6154', '0.5385', '0.4615', '0.9231']
oz84 (numeric, 8 distinct): ['0.0', '0.125', '0.25', '0.375', '0.5', '0.75', '0.625', '1.0']
oz85 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '1.0', '0.75']
oz86 (numeric, 1 distinct): ['0']
oz87 (numeric, 12 distinct): ['0.0', '0.0909', '0.1818', '0.2727', '0.3636', '0.4545', '0.7273', '0.9091', '0.6364', '0.8182']
oz88 (numeric, 1 distinct): ['0']
oz89 (numeric, 4 distinct): ['0.0', '0.3333', '0.6667', '1.0']
oz90 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz91 (numeric, 9 distinct): ['0.0', '0.125', '0.25', '0.375', '0.5', '0.625', '0.75', '0.875', '1.0']
oz92 (numeric, 7 distinct): ['0.0', '0.1667', '0.3333', '0.5', '0.6667', '1.0', '0.8333']
oz93 (numeric, 3 distinct): ['0.0', '1.0', '0.5']
oz94 (numeric, 1 distinct): ['0']
oz95 (numeric, 1 distinct): ['0']
oz96 (numeric, 2 distinct): ['0', '1']
oz97 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz98 (numeric, 4 distinct): ['0.0', '0.3333', '0.6667', '1.0']
oz99 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz100 (numeric, 2 distinct): ['0', '1']
oz101 (numeric, 6 distinct): ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
oz102 (numeric, 4 distinct): ['0.0', '0.3333', '0.6667', '1.0']
oz103 (numeric, 4 distinct): ['0.0', '0.25', '0.5', '1.0']
oz104 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '0.75', '1.0']
oz105 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz106 (numeric, 4 distinct): ['0.0', '0.1667', '0.3333', '1.0']
oz107 (numeric, 2 distinct): ['0', '1']
oz108 (numeric, 2 distinct): ['0', '1']
oz109 (numeric, 1 distinct): ['0']
oz110 (numeric, 4 distinct): ['0.0', '0.25', '0.5', '1.0']
oz111 (numeric, 2 distinct): ['0', '1']
oz112 (numeric, 2 distinct): ['0', '1']
oz113 (numeric, 2 distinct): ['0', '1']
oz114 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz115 (numeric, 2 distinct): ['0', '1']
oz116 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz117 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz118 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz119 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '1.0', '0.75']
oz120 (numeric, 9 distinct): ['0.0', '0.1', '0.2', '0.3', '0.4', '0.6', '0.5', '1.0', '0.7']
oz121 (numeric, 7 distinct): ['0.0', '0.1667', '0.3333', '0.5', '0.6667', '1.0', '0.8333']
oz122 (numeric, 1 distinct): ['0']
oz123 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz124 (numeric, 15 distinct): ['0.0', '0.0667', '0.1333', '0.2', '0.2667', '0.3333', '0.4', '0.4667', '0.5333', '0.6667']
oz125 (numeric, 27 distinct): ['0.1429', '0.1071', '0.1786', '0.0714', '0.2143', '0.25', '0.2857', '0.0357', '0.3214', '0.3571']
oz126 (numeric, 14 distinct): ['0.0', '0.0667', '0.1333', '0.2', '0.2667', '0.3333', '0.4', '0.4667', '0.5333', '0.6']
oz127 (numeric, 31 distinct): ['0.0', '0.0286', '0.0571', '0.0857', '0.1143', '0.1429', '0.1714', '0.2286', '0.2857', '0.2']
oz128 (numeric, 15 distinct): ['0.0', '0.0588', '0.1176', '0.1765', '0.2353', '0.2941', '0.3529', '0.4118', '0.4706', '0.5294']
oz129 (numeric, 7 distinct): ['0.0', '0.1667', '0.3333', '0.5', '0.6667', '1.0', '0.8333']
oz130 (numeric, 9 distinct): ['0.0', '0.125', '0.25', '0.375', '0.5', '0.625', '0.75', '1.0', '0.875']
oz131 (numeric, 14 distinct): ['0.0', '0.05', '0.1', '0.2', '0.15', '0.3', '0.25', '0.4', '0.6', '0.35']
oz132 (numeric, 6 distinct): ['0.0', '0.1667', '0.3333', '0.5', '0.6667', '1.0']
oz133 (numeric, 15 distinct): ['0.0', '0.0667', '0.1333', '0.2', '0.2667', '0.3333', '0.4', '0.4667', '0.5333', '0.9333']
oz134 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '0.75', '1.0']
oz135 (numeric, 2 distinct): ['0', '1']
oz136 (numeric, 6 distinct): ['0.0', '0.125', '0.25', '0.375', '0.5', '1.0']
oz137 (numeric, 7 distinct): ['0.0', '0.125', '0.25', '0.375', '0.5', '0.75', '1.0']
oz138 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '1.0', '0.75']
oz139 (numeric, 4 distinct): ['0.0', '0.3333', '0.6667', '1.0']
oz140 (numeric, 4 distinct): ['0.0', '0.25', '0.5', '1.0']
oz141 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '0.75', '1.0']
oz142 (numeric, 4 distinct): ['0.0', '0.3333', '0.6667', '1.0']
oz143 (numeric, 4 distinct): ['0.0', '0.25', '0.5', '1.0']
oz144 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz145 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz146 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz147 (numeric, 6 distinct): ['0.0', '0.3333', '0.1667', '0.6667', '0.5', '1.0']
oz148 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz149 (numeric, 31 distinct): ['0.0', '0.0769', '0.0962', '0.1538', '0.0577', '0.0385', '0.1731', '0.1154', '0.1346', '0.1923']
oz150 (numeric, 16 distinct): ['0.0', '0.0455', '0.0909', '0.1364', '0.1818', '0.2273', '0.2727', '0.3182', '0.3636', '0.4091']
oz151 (numeric, 14 distinct): ['0.0', '0.0714', '0.1429', '0.2143', '0.2857', '0.3571', '0.4286', '0.5714', '0.5', '0.7143']
oz152 (numeric, 7 distinct): ['0.0', '0.1667', '0.3333', '0.5', '0.6667', '0.8333', '1.0']
oz153 (numeric, 9 distinct): ['0.0', '0.0833', '0.1667', '0.25', '0.3333', '0.5', '0.4167', '0.6667', '1.0']
oz154 (numeric, 6 distinct): ['0.0', '0.1667', '0.3333', '0.5', '0.6667', '1.0']
oz155 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz156 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '1.0', '0.75']
oz157 (numeric, 6 distinct): ['0.0', '0.1667', '0.3333', '0.5', '1.0', '0.6667']
oz158 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '0.75', '1.0']
oz159 (numeric, 6 distinct): ['0.0', '0.1667', '0.3333', '0.6667', '0.5', '1.0']
oz160 (numeric, 4 distinct): ['0.0', '0.25', '0.5', '1.0']
oz161 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz162 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz163 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '1.0', '0.75']
oz164 (numeric, 6 distinct): ['0.0', '0.1667', '0.3333', '0.5', '0.6667', '1.0']
oz165 (numeric, 11 distinct): ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '1.0']
oz166 (numeric, 6 distinct): ['0.0', '0.1667', '0.3333', '0.5', '0.6667', '1.0']
oz167 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz168 (numeric, 4 distinct): ['0.0', '0.3333', '0.6667', '1.0']
oz169 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz170 (numeric, 1 distinct): ['0']
oz171 (numeric, 58 distinct): ['0.0', '0.0375', '0.075', '0.025', '0.05', '0.1125', '0.0625', '0.1', '0.15', '0.0875']
oz172 (numeric, 44 distinct): ['0.1538', '0.0769', '0.1154', '0.1346', '0.1923', '0.1731', '0.0962', '0.2115', '0.2308', '0.0385']
oz173 (numeric, 10 distinct): ['0.0', '0.0833', '0.1667', '0.3333', '0.25', '0.5', '0.4167', '1.0', '0.6667', '0.5833']
oz174 (numeric, 7 distinct): ['0.0', '0.1667', '0.3333', '0.5', '0.6667', '1.0', '0.8333']
oz175 (numeric, 14 distinct): ['0.0', '0.0769', '0.1538', '0.2308', '0.3077', '0.3846', '0.4615', '0.6154', '0.5385', '0.6923']
oz176 (numeric, 20 distinct): ['0.0', '0.0769', '0.1154', '0.0385', '0.1538', '0.2308', '0.1923', '0.4615', '0.3462', '0.3077']
oz177 (numeric, 28 distinct): ['0.0', '0.0833', '0.0556', '0.1111', '0.1667', '0.2222', '0.1389', '0.0278', '0.25', '0.1944']
oz178 (numeric, 11 distinct): ['0.0', '0.1667', '0.0833', '0.3333', '0.25', '0.5', '0.4167', '1.0', '0.6667', '0.5833']
oz179 (numeric, 7 distinct): ['0.0', '0.3333', '0.1667', '0.6667', '0.5', '1.0', '0.8333']
oz180 (numeric, 4 distinct): ['0.0', '0.25', '0.5', '1.0']
oz181 (numeric, 12 distinct): ['0.0', '0.0769', '0.1538', '0.2308', '0.3077', '0.3846', '0.6154', '0.5385', '0.4615', '0.9231']
oz182 (numeric, 9 distinct): ['0.0', '0.0833', '0.1667', '0.25', '0.3333', '0.5', '0.4167', '1.0', '0.6667']
oz183 (numeric, 13 distinct): ['0.0', '0.0833', '0.1667', '0.25', '0.3333', '0.4167', '0.5', '0.5833', '0.6667', '0.8333']
oz184 (numeric, 9 distinct): ['0.0', '0.125', '0.25', '0.375', '0.5', '0.625', '0.75', '0.875', '1.0']
oz185 (numeric, 10 distinct): ['0.0', '0.0833', '0.1667', '0.25', '0.3333', '0.4167', '0.6667', '0.5833', '0.5', '1.0']
oz186 (numeric, 8 distinct): ['0.0', '0.25', '0.5', '0.125', '0.75', '1.0', '0.375', '0.875']
oz187 (numeric, 1 distinct): ['0']
oz188 (numeric, 1 distinct): ['0']
oz189 (numeric, 1 distinct): ['0']
oz190 (numeric, 1 distinct): ['0']
oz191 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz192 (numeric, 4 distinct): ['0.0', '0.25', '0.5', '1.0']
oz193 (numeric, 4 distinct): ['0.0', '0.25', '0.5', '1.0']
oz194 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '1.0', '0.75']
oz195 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '1.0', '0.75']
oz196 (numeric, 4 distinct): ['0.0', '0.3333', '0.6667', '1.0']
oz197 (numeric, 10 distinct): ['0.0', '0.1', '0.2', '0.3', '0.4', '0.6', '0.5', '0.8', '0.7', '1.0']
oz198 (numeric, 7 distinct): ['0.0', '0.3333', '0.1667', '0.5', '0.6667', '0.8333', '1.0']
oz199 (numeric, 6 distinct): ['0.0', '0.1667', '0.3333', '0.5', '0.6667', '1.0']
oz200 (numeric, 8 distinct): ['0.0', '0.1667', '0.0833', '0.25', '0.3333', '0.5', '0.4167', '1.0']
oz201 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '1.0', '0.75']
oz202 (numeric, 5 distinct): ['0.0', '0.25', '0.75', '0.5', '1.0']
oz203 (numeric, 5 distinct): ['0.0', '0.1667', '0.3333', '0.5', '1.0']
oz204 (numeric, 1 distinct): ['0']
oz205 (numeric, 1 distinct): ['0']
oz206 (numeric, 2 distinct): ['0', '1']
oz207 (numeric, 6 distinct): ['0.0', '0.25', '0.75', '0.125', '0.375', '1.0']
oz208 (numeric, 9 distinct): ['0.0', '0.25', '0.5', '0.4167', '1.0', '0.1667', '0.3333', '0.75', '0.8333']
oz209 (numeric, 9 distinct): ['0.0', '0.1', '0.2', '0.4', '0.3', '0.5', '0.6', '1.0', '0.7']
oz210 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz211 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '1.0', '0.75']
oz212 (numeric, 5 distinct): ['0.0', '0.5', '0.25', '1.0', '0.75']
oz213 (numeric, 6 distinct): ['0.0', '0.5', '0.3333', '0.6667', '1.0', '0.1667']
oz214 (numeric, 7 distinct): ['0.0', '0.1667', '0.3333', '0.5', '0.6667', '0.8333', '1.0']
oz215 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '0.75', '1.0']
oz216 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz217 (numeric, 4 distinct): ['0.0', '0.5', '0.25', '1.0']
oz218 (numeric, 3 distinct): ['0.0', '1.0', '0.3333']
oz219 (numeric, 6 distinct): ['0.0', '0.1667', '0.3333', '0.6667', '0.5', '1.0']
oz220 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz221 (numeric, 4 distinct): ['0.0', '0.25', '0.5', '1.0']
oz222 (numeric, 2 distinct): ['0', '1']
oz223 (numeric, 1 distinct): ['0']
oz224 (numeric, 4 distinct): ['0.0', '0.25', '0.5', '1.0']
oz225 (numeric, 4 distinct): ['0.0', '0.25', '1.0', '0.5']
oz226 (numeric, 1 distinct): ['0']
oz227 (numeric, 1 distinct): ['0']
oz228 (numeric, 1 distinct): ['0']
oz229 (numeric, 1 distinct): ['0']
oz230 (numeric, 1 distinct): ['0']
oz231 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz232 (numeric, 6 distinct): ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
oz233 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '1.0', '0.75']
oz234 (numeric, 2 distinct): ['0', '1']
oz235 (numeric, 5 distinct): ['0.0', '0.25', '0.5', '0.75', '1.0']
oz236 (numeric, 1 distinct): ['0']
oz237 (numeric, 1 distinct): ['0']
oz238 (numeric, 1 distinct): ['0']
oz239 (numeric, 1 distinct): ['0']
oz240 (numeric, 1 distinct): ['0']
oz241 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz242 (numeric, 4 distinct): ['0.0', '0.1667', '0.3333', '1.0']
oz243 (numeric, 3 distinct): ['0.0', '0.5', '1.0']
oz244 (numeric, 5 distinct): ['0.0', '0.25', '1.0', '0.5', '0.75']
oz245 (numeric, 1 distinct): ['0']
oz246 (numeric, 54 distinct): ['0.4973', '0.4706', '0.5972', '0.4403', '0.5804', '0.6129', '0.1569', '0.5426', '0.5211', '0.6411']
oz247 (numeric, 1231 distinct): ['0.0205', '0.0268', '0.0353', '0.0354', '0.0328', '0.0286', '0.0227', '0.0377', '0.0281', '0.0254']
oz248 (numeric, 501 distinct): ['0.0', '0.5', '0.667', '0.4', '0.333', '0.462', '0.6', '0.429', '0.571', '0.545']
oz249 (numeric, 8379 distinct): ['0.0934', '0.1782', '0.1007', '0.0997', '0.1878', '0.2407', '0.1902', '0.1398', '0.172', '0.223']
oz250 (numeric, 1773 distinct): ['0.0533', '0.0', '0.0821', '0.1066', '0.1354', '0.1642', '0.0634', '0.1109', '0.0288', '0.1598']
oz251 (numeric, 4638 distinct): ['0.5045', '0.5568', '0.5524', '0.5521', '0.5684', '0.564', '0.4694', '0.5412', '0.5248', '0.5644']
'''

CONTEXT = "YProp Molecules"
TARGET = CuratedTarget(raw_name="oz252", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []