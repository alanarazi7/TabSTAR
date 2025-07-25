from tabstar_paper.datasets.curation_objects import CuratedFeature, CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: AVIDa-hIL6
====
Examples: 573891
====
URL: https://www.openml.org/search?type=data&id=46648
====
Description: AVIDa-hIL6 is an antigen-variable domain of heavy chain of heavy chain antibody (VHH) 
interaction dataset produced from an alpaca immunized with the human interleukin-6 (IL-6) 
protein. By leveraging the simple structure of VHHs, which facilitates identification of 
full-length amino acid sequences by DNA sequencing technology, AVIDa-hIL6 contains 573,891 
antigen-VHH pairs with amino acid sequences. All the antigen-VHH pairs have reliable 
labels for binding or non-binding, as generated by a novel labeling method. Furthermore,
AVIDa-hIL6 has the wild type and 30 mutants of the IL-6 protein as antigens, and it 
includes many sensitive cases in which point mutations in IL-6 enhance or inhibit antibody
binding. We envision that AVIDa-hIL6 will serve as a valuable benchmark for machine 
learning research in the growing field of predicting antigen-antibody interactions.
====
Target Variable: label (numeric, 2 distinct): ['0', '1']
====
Features:

VHH_sequence (string, 38599 distinct): ['MKYLLPTAAAGLLLLAAQPAMAQVQLQESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNANNTVYLQMNSLKPEDTAVYACAADLLYHPGSWNDYWGQGTQVTVSSHHHHHH', 'MKYLLPTAAAGLLLLAAQPAMAQVQLQESGGGLVQAGGSLRLSCAASGIISSINAMGWYRQAPGKQRELVATITNGGSTNYADSVKGRFTISRDNAKNTLYLQMNSLKPEDTAVYYCRADLVVAGTRFPSWGQGTQVTVSSHHHHHH', 'MKYLLPTAAAGLLLLAAQPALAQVQLQESGGGLVQAGGSLRLSCAGSGITFNRYNMGWFRQAPGKEREFVAGIIWSGGITDYGDFAKGRFTISMDHAKKEVSLQMLSLKPEDTAVYYCAADLGNPYSGYDRSRLAAYDAWGQGTQVTVSSHHHHHH', 'MKYLLPTAAAGLLLLAAQPAMAQVQLQESGGGLVRTGDSLTLSCAASGRASRGYAMGWFRQAPGKEREFVSCIGSGGSTYYANSVKGRFTVSRDNANDTVYLQMNSLRPNDTAAYYCAADRTTTRDYCYTAPVVYTYWGQGTQVTVSSHHHHHH', 'MKYLLPTAAAGLLLLAAQPAMAQVQLQESGGGSVQPGGSLRLSCVASGFTFDDYAMTWVRQAPGKGLEWVSTITWNGGSTRYGESMKGRFTVSRDNAKNTLYLQMNSLRSEDTAVYYCVIGTYNSDYDFVPLGQGTQVTVSSHHHHHH', 'MKYLLPTAAAGLLLLAAQPAMAQVQLQESGGGLVQPGGSLRLSCVVSGYTLDYLNVGWFRQAPGKEREGVSCSRSAGDYTIYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTALYFCVATHQRHASGAYVCHDFMIGDFAAWGQGTQVTVSSHHHHHH', 'MKYLLPTAAAGLLLLAAQPAMAQVQLQESGGGLVQPGGSLTLACAASGSILDIDIMRWYRQAPGEQREIVATITNSGTTTYRDSVKGRFTISRDTAENTVYLQMNSLKPEDTAVYTCQADVYVNGDDDKFQFFGFWGQGTQVTVPSHHHHHH', 'MKYLLPTAAAGLLLLAAQPAMAQVQLQESGGGLVQPGGSLRLSCATSQSIFDFTVKGWYRQAPGKQRELVATITRAGTTIYGDSVKGRFSISKDNAQNTVYLQMDALKEEDTAVYYCNGVLSFYLAAQTNYWGRGTQVTVSSHHHHHH', 'MKYLLPTAAAGLLLLAAQPAMAQVQLQESGGGLVQPGGSLTLACAASGSILDIDIMRWYRQAPGEQREIVATITNSGTTTYRDSVKGRFTISRDTAENTVYLQMNSLKPEDTAVYTCQADVYVNGDDDKFQFFGFWGQGTQVTVSPHHHHHH', 'MKYLLPTAAAGLLLLAAQPAMAQVQLQESGGGLVQPGGSLTLSCAASGSIFGINRMGWYRQAPGKQRELVATVTGGGNTVYSDSVKGRFTVSRDNAKNTVTLQMNSLKPEDTAVYYCNYRRVVAGEDYWGQGTQVTVSSHHHHHH']
Ag_label (string, 31 distinct): ['IL-6_D54A', 'IL-6_I57A', 'IL-6_I60A', 'IL-6_G63A', 'IL-6_K69A', 'IL-6_E87A', 'IL-6_C72A', 'IL-6_S81A', 'IL-6_F102A', 'IL-6_D99A']
Ag_sequence (string, 31 distinct): ['MNSFSTSAFGPVAFSLGLLLVLPAAFPAPVPPGEDSKDVAAPHRQPLTSSERIAKQIRYILDGISALRKETCNKSNMCESSKEALAENNLNLPKMAEKDGCFQSGFNEETCLVKIITGLLEFEVYLEYLQNRFESSEEQARAVQMSTKVLIQFLQKKAKNLDAITTPDPTTNASLLTKLQAQNQWLQDMTTHLILRSFKEFLQSSLRALRQMHHHHHH', 'MNSFSTSAFGPVAFSLGLLLVLPAAFPAPVPPGEDSKDVAAPHRQPLTSSERIDKQARYILDGISALRKETCNKSNMCESSKEALAENNLNLPKMAEKDGCFQSGFNEETCLVKIITGLLEFEVYLEYLQNRFESSEEQARAVQMSTKVLIQFLQKKAKNLDAITTPDPTTNASLLTKLQAQNQWLQDMTTHLILRSFKEFLQSSLRALRQMHHHHHH', 'MNSFSTSAFGPVAFSLGLLLVLPAAFPAPVPPGEDSKDVAAPHRQPLTSSERIDKQIRYALDGISALRKETCNKSNMCESSKEALAENNLNLPKMAEKDGCFQSGFNEETCLVKIITGLLEFEVYLEYLQNRFESSEEQARAVQMSTKVLIQFLQKKAKNLDAITTPDPTTNASLLTKLQAQNQWLQDMTTHLILRSFKEFLQSSLRALRQMHHHHHH', 'MNSFSTSAFGPVAFSLGLLLVLPAAFPAPVPPGEDSKDVAAPHRQPLTSSERIDKQIRYILDAISALRKETCNKSNMCESSKEALAENNLNLPKMAEKDGCFQSGFNEETCLVKIITGLLEFEVYLEYLQNRFESSEEQARAVQMSTKVLIQFLQKKAKNLDAITTPDPTTNASLLTKLQAQNQWLQDMTTHLILRSFKEFLQSSLRALRQMHHHHHH', 'MNSFSTSAFGPVAFSLGLLLVLPAAFPAPVPPGEDSKDVAAPHRQPLTSSERIDKQIRYILDGISALRAETCNKSNMCESSKEALAENNLNLPKMAEKDGCFQSGFNEETCLVKIITGLLEFEVYLEYLQNRFESSEEQARAVQMSTKVLIQFLQKKAKNLDAITTPDPTTNASLLTKLQAQNQWLQDMTTHLILRSFKEFLQSSLRALRQMHHHHHH', 'MNSFSTSAFGPVAFSLGLLLVLPAAFPAPVPPGEDSKDVAAPHRQPLTSSERIDKQIRYILDGISALRKETCNKSNMCESSKEALAANNLNLPKMAEKDGCFQSGFNEETCLVKIITGLLEFEVYLEYLQNRFESSEEQARAVQMSTKVLIQFLQKKAKNLDAITTPDPTTNASLLTKLQAQNQWLQDMTTHLILRSFKEFLQSSLRALRQMHHHHHH', 'MNSFSTSAFGPVAFSLGLLLVLPAAFPAPVPPGEDSKDVAAPHRQPLTSSERIDKQIRYILDGISALRKETANKSNMCESSKEALAENNLNLPKMAEKDGCFQSGFNEETCLVKIITGLLEFEVYLEYLQNRFESSEEQARAVQMSTKVLIQFLQKKAKNLDAITTPDPTTNASLLTKLQAQNQWLQDMTTHLILRSFKEFLQSSLRALRQMHHHHHH', 'MNSFSTSAFGPVAFSLGLLLVLPAAFPAPVPPGEDSKDVAAPHRQPLTSSERIDKQIRYILDGISALRKETCNKSNMCESAKEALAENNLNLPKMAEKDGCFQSGFNEETCLVKIITGLLEFEVYLEYLQNRFESSEEQARAVQMSTKVLIQFLQKKAKNLDAITTPDPTTNASLLTKLQAQNQWLQDMTTHLILRSFKEFLQSSLRALRQMHHHHHH', 'MNSFSTSAFGPVAFSLGLLLVLPAAFPAPVPPGEDSKDVAAPHRQPLTSSERIDKQIRYILDGISALRKETCNKSNMCESSKEALAENNLNLPKMAEKDGCAQSGFNEETCLVKIITGLLEFEVYLEYLQNRFESSEEQARAVQMSTKVLIQFLQKKAKNLDAITTPDPTTNASLLTKLQAQNQWLQDMTTHLILRSFKEFLQSSLRALRQMHHHHHH', 'MNSFSTSAFGPVAFSLGLLLVLPAAFPAPVPPGEDSKDVAAPHRQPLTSSERIDKQIRYILDGISALRKETCNKSNMCESSKEALAENNLNLPKMAEKAGCFQSGFNEETCLVKIITGLLEFEVYLEYLQNRFESSEEQARAVQMSTKVLIQFLQKKAKNLDAITTPDPTTNASLLTKLQAQNQWLQDMTTHLILRSFKEFLQSSLRALRQMHHHHHH']
'''

CONTEXT = "AVIDa-hIL6 Antigen-antibody binding dataset"
TARGET = CuratedTarget(raw_name="label", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []