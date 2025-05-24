from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: audiology
====
Examples: 226
====
URL: https://www.openml.org/search?type=data&id=7
====
Description: **Author**: Professor Jergen at Baylor College of Medicine
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Audiology+(Standardized))
**Please cite**: Bareiss, E. Ray, & Porter, Bruce (1987). Protos: An Exemplar-Based Learning Apprentice. In the Proceedings of the 4th International Workshop on Machine Learning, 12-23, Irvine, CA: Morgan Kaufmann

**Audiology Database**
This database is a standardized version of the original audiology database (see audiology.* in this directory). The non-standard set of attributes have been converted to a standard set of attributes according to the rules that follow.

* Each property that appears anywhere in the original .data or .test file has been represented as a separate attribute in this file.

* A property such as age_gt_60 is represented as a boolean attribute with values f and t.

* In most cases, a property of the form x(y) is represented as a discrete attribute x() whose possible values are the various y's; air() is an example. There are two exceptions:
** when only one value of y appears anywhere, e.g. static(normal). In this case, x_y appears as a boolean attribute.
** when one case can have two or more values of x, e.g. history(..). All possible values of history are treated as separate boolean attributes.

* Since boolean attributes only appear as positive conditions, each boolean attribute is assumed to be false unless noted as true. The value of multi-value discrete attributes taken as unknown ("?") unless a value is specified.

* The original case identifications, p1 to p200 in the .data file and t1 to t26 in the .test file, have been added as a unique identifier attribute.

[Note: in the original .data file, p165 has a repeated specification of o_ar_c(normal); p166 has repeated specification of speech(normal) and conflicting values air(moderate) and air(mild). No other problems with the original data were noted.]


### Attribute Information:

age_gt_60: f, t.
air(): mild,moderate,severe,normal,profound.
airBoneGap: f, t.
ar_c(): normal,elevated,absent.
ar_u(): normal,absent,elevated.
bone(): mild,moderate,normal,unmeasured.
boneAbnormal: f, t.
bser(): normal,degraded.
history_buzzing: f, t.
history_dizziness: f, t.
history_fluctuating: f, t.
history_fullness: f, t.
history_heredity: f, t.
history_nausea: f, t.
history_noise: f, t.
history_recruitment: f, t.
history_ringing: f, t.
history_roaring: f, t.
history_vomiting: f, t.
late_wave_poor: f, t.
m_at_2k: f, t.
m_cond_lt_1k: f, t.
m_gt_1k: f, t.
m_m_gt_2k: f, t.
m_m_sn: f, t.
m_m_sn_gt_1k: f, t.
m_m_sn_gt_2k: f, t.
m_m_sn_gt_500: f, t.
m_p_sn_gt_2k: f, t.
m_s_gt_500: f, t.
m_s_sn: f, t.
m_s_sn_gt_1k: f, t.
m_s_sn_gt_2k: f, t.
m_s_sn_gt_3k: f, t.
m_s_sn_gt_4k: f, t.
m_sn_2_3k: f, t.
m_sn_gt_1k: f, t.
m_sn_gt_2k: f, t.
m_sn_gt_3k: f, t.
m_sn_gt_4k: f, t.
m_sn_gt_500: f, t.
m_sn_gt_6k: f, t.
m_sn_lt_1k: f, t.
m_sn_lt_2k: f, t.
m_sn_lt_3k: f, t.
middle_wave_poor: f, t.
mod_gt_4k: f, t.
mod_mixed: f, t.
mod_s_mixed: f, t.
mod_s_sn_gt_500: f, t.
mod_sn: f, t.
mod_sn_gt_1k: f, t.
mod_sn_gt_2k: f, t.
mod_sn_gt_3k: f, t.
mod_sn_gt_4k: f, t.
mod_sn_gt_500: f, t.
notch_4k: f, t.
notch_at_4k: f, t.
o_ar_c(): normal,elevated,absent.
o_ar_u(): normal,absent,elevated.
s_sn_gt_1k: f, t.
s_sn_gt_2k: f, t.
s_sn_gt_4k: f, t.
speech(): normal,good,very_good,very_poor,poor,unmeasured.
static_normal: f, t.
tymp(): a,as,b,ad,c.
viith_nerve_signs: f, t.
wave_V_delayed: f, t.
waveform_ItoV_prolonged: f, t.
indentifier (unique for each instance)

class:
cochlear_unknown,mixed_cochlear_age_fixation,poss_central
mixed_cochlear_age_otitis_media,mixed_poss_noise_om,
cochlear_age,normal_ear,cochlear_poss_noise,cochlear_age_and_noise,
acoustic_neuroma,mixed_cochlear_unk_ser_om,conductive_discontinuity,
retrocochlear_unknown,conductive_fixation,bells_palsy,
cochlear_noise_and_heredity,mixed_cochlear_unk_fixation,
otitis_media,possible_menieres,possible_brainstem_disorder,
cochlear_age_plus_poss_menieres,mixed_cochlear_age_s_om,
mixed_cochlear_unk_discontinuity,mixed_poss_central_om
====
Target Variable: class (nominal, 24 distinct): ['cochlear_age', 'cochlear_unknown', 'cochlear_age_and_noise', 'normal_ear', 'cochlear_poss_noise', 'mixed_cochlear_unk_fixation', 'possible_menieres', 'conductive_fixation', 'possible_brainstem_disorder', 'otitis_media']
====
Features:

age_gt_60 (nominal, 2 distinct): ['f', 't']
air (nominal, 5 distinct): ['mild', 'normal', 'moderate', 'severe', 'profound']
airBoneGap (nominal, 2 distinct): ['f', 't']
ar_c (nominal, 4 distinct): ['normal', 'absent', 'elevated']
ar_u (nominal, 4 distinct): ['normal', 'absent', 'elevated']
bone (nominal, 5 distinct): ['mild', 'unmeasured', 'normal', 'moderate']
boneAbnormal (nominal, 2 distinct): ['f', 't']
bser (nominal, 3 distinct): ['degraded', 'normal']
history_buzzing (nominal, 2 distinct): ['f', 't']
history_dizziness (nominal, 2 distinct): ['f', 't']
history_fluctuating (nominal, 2 distinct): ['f', 't']
history_fullness (nominal, 2 distinct): ['f', 't']
history_heredity (nominal, 2 distinct): ['f', 't']
history_nausea (nominal, 2 distinct): ['f', 't']
history_noise (nominal, 2 distinct): ['f', 't']
history_recruitment (nominal, 2 distinct): ['f', 't']
history_ringing (nominal, 2 distinct): ['f', 't']
history_roaring (nominal, 2 distinct): ['f', 't']
history_vomiting (nominal, 2 distinct): ['f', 't']
late_wave_poor (nominal, 2 distinct): ['f', 't']
m_at_2k (nominal, 2 distinct): ['f', 't']
m_cond_lt_1k (nominal, 2 distinct): ['f', 't']
m_gt_1k (nominal, 2 distinct): ['f', 't']
m_m_gt_2k (nominal, 2 distinct): ['f', 't']
m_m_sn (nominal, 2 distinct): ['f', 't']
m_m_sn_gt_1k (nominal, 2 distinct): ['f', 't']
m_m_sn_gt_2k (nominal, 2 distinct): ['f', 't']
m_m_sn_gt_500 (nominal, 2 distinct): ['f', 't']
m_p_sn_gt_2k (nominal, 2 distinct): ['f', 't']
m_s_gt_500 (nominal, 2 distinct): ['f', 't']
m_s_sn (nominal, 2 distinct): ['f', 't']
m_s_sn_gt_1k (nominal, 2 distinct): ['f', 't']
m_s_sn_gt_2k (nominal, 2 distinct): ['f', 't']
m_s_sn_gt_3k (nominal, 2 distinct): ['f', 't']
m_s_sn_gt_4k (nominal, 2 distinct): ['f', 't']
m_sn_2_3k (nominal, 2 distinct): ['f', 't']
m_sn_gt_1k (nominal, 2 distinct): ['f', 't']
m_sn_gt_2k (nominal, 2 distinct): ['f', 't']
m_sn_gt_3k (nominal, 2 distinct): ['f', 't']
m_sn_gt_4k (nominal, 2 distinct): ['f', 't']
m_sn_gt_500 (nominal, 2 distinct): ['f', 't']
m_sn_gt_6k (nominal, 2 distinct): ['f', 't']
m_sn_lt_1k (nominal, 2 distinct): ['f', 't']
m_sn_lt_2k (nominal, 2 distinct): ['f', 't']
m_sn_lt_3k (nominal, 2 distinct): ['f', 't']
middle_wave_poor (nominal, 2 distinct): ['f', 't']
mod_gt_4k (nominal, 2 distinct): ['f', 't']
mod_mixed (nominal, 2 distinct): ['f', 't']
mod_s_mixed (nominal, 2 distinct): ['f', 't']
mod_s_sn_gt_500 (nominal, 2 distinct): ['f', 't']
mod_sn (nominal, 2 distinct): ['f', 't']
mod_sn_gt_1k (nominal, 2 distinct): ['f', 't']
mod_sn_gt_2k (nominal, 2 distinct): ['f', 't']
mod_sn_gt_3k (nominal, 2 distinct): ['f', 't']
mod_sn_gt_4k (nominal, 2 distinct): ['f', 't']
mod_sn_gt_500 (nominal, 2 distinct): ['f', 't']
notch_4k (nominal, 2 distinct): ['f', 't']
notch_at_4k (nominal, 2 distinct): ['f', 't']
o_ar_c (nominal, 4 distinct): ['normal', 'absent', 'elevated']
o_ar_u (nominal, 4 distinct): ['normal', 'absent', 'elevated']
s_sn_gt_1k (nominal, 2 distinct): ['f', 't']
s_sn_gt_2k (nominal, 2 distinct): ['f', 't']
s_sn_gt_4k (nominal, 2 distinct): ['f', 't']
speech (nominal, 7 distinct): ['normal', 'good', 'very_good', 'very_poor', 'poor', 'unmeasured']
static_normal (nominal, 2 distinct): ['t', 'f']
tymp (nominal, 5 distinct): ['a', 'as', 'b', 'c', 'ad']
viith_nerve_signs (nominal, 2 distinct): ['f', 't']
wave_V_delayed (nominal, 2 distinct): ['f', 't']
waveform_ItoV_prolonged (nominal, 2 distinct): ['f', 't']
'''


CONTEXT = "Audiology Hearing Tests and Diagnostic Classification"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.MULTICLASS,)
COLS_TO_DROP = []
FEATURES = []