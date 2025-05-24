from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: colleges
====
Examples: 7063
====
URL: https://www.openml.org/search?type=data&id=42727
====
Description: Modified version for the automl benchmark.
Regroups information for about 7800 different US colleges. Including geographical information, stats about the population attending and post graduation career earnings.
====
Target Variable: percent_pell_grant (numeric, 4502 distinct): ['1.0', '0.0', '0.5', '0.6667', '0.3333', '0.6', '0.75', '0.5714', '0.7143', '0.7']
====
Features:

city (nominal, 2460 distinct): ['New York', 'Chicago', 'Houston', 'Brooklyn', 'Los Angeles', 'San Antonio', 'Miami', 'Phoenix', 'Springfield', 'Jacksonville']
state (nominal, 59 distinct): ['CA', 'TX', 'NY', 'FL', 'PA', 'OH', 'IL', 'MO', 'MI', 'NC']
zip (nominal, 6039 distinct): ['78229', '90010', '85021', '7306', '45246', '80226', '11219', '961', '12205', '32216']
latitude (numeric, 6509 distinct): ['34.0617', '40.4375', '40.6355', '34.1954', '41.5496', '42.1979', '38.7882', '34.0946', '39.9549', '32.8195']
longitude (numeric, 6599 distinct): ['-112.1075', '-87.6248', '-112.1184', '-73.9899', '-83.2384', '-117.158', '-118.2724', '-73.9822', '-71.8086', '-73.9938']
admission_rate (numeric, 1740 distinct): ['1.0', '0.5', '0.8', '0.6667', '0.9524', '0.8889', '0.3333', '0.6', '0.75', '0.8065']
sat_verbal_midrange (numeric, 165 distinct): ['500.0', '495.0', '520.0', '540.0', '475.0', '505.0', '515.0', '490.0', '485.0', '525.0']
sat_math_midrange (numeric, 164 distinct): ['490.0', '510.0', '500.0', '530.0', '495.0', '505.0', '515.0', '560.0', '475.0', '525.0']
sat_writing_midrange (numeric, 132 distinct): ['490.0', '515.0', '480.0', '470.0', '520.0', '475.0', '530.0', '510.0', '495.0', '505.0']
act_combined_midrange (numeric, 23 distinct): ['22.0', '23.0', '24.0', '21.0', '20.0', '25.0', '26.0', '19.0', '27.0', '28.0']
act_english_midrange (numeric, 25 distinct): ['23.0', '22.0', '21.0', '24.0', '20.0', '19.0', '25.0', '26.0', '17.0', '18.0']
act_math_midrange (numeric, 25 distinct): ['21.0', '22.0', '20.0', '23.0', '24.0', '25.0', '19.0', '26.0', '18.0', '17.0']
act_writing_midrange (numeric, 9 distinct): ['7.0', '9.0', '8.0', '6.0', '10.0', '11.0', '5.0', '12.0']
sat_total_average (numeric, 478 distinct): ['1030.0', '1070.0', '1050.0', '1010.0', '950.0', '990.0', '1090.0', '970.0', '930.0', '1105.0']
undergrad_size (numeric, 3021 distinct): ['33.0', '96.0', '35.0', '58.0', '105.0', '53.0', '80.0', '39.0', '45.0', '28.0']
percent_white (numeric, 4454 distinct): ['0.0', '1.0', '0.5', '0.75', '0.8333', '0.6667', '0.8', '0.8571', '0.8889', '0.7778']
percent_black (numeric, 3278 distinct): ['0.0', '1.0', '0.0714', '0.0435', '0.037', '0.0909', '0.0556', '0.2', '0.0571', '0.0208']
percent_hispanic (numeric, 2803 distinct): ['0.0', '1.0', '0.0455', '0.0303', '0.0769', '0.0286', '0.0556', '0.0417', '0.0278', '0.1111']
percent_asian (numeric, 1240 distinct): ['0.0', '0.0079', '0.0078', '0.0083', '0.0044', '0.0091', '0.0058', '0.0065', '0.0112', '0.0056']
percent_part_time (numeric, 3467 distinct): ['0.0', '1.0', '0.25', '0.3333', '0.2857', '0.5', '0.0769', '0.3636', '0.0655', '0.037']
average_cost_academic_year (numeric, 3803 distinct): ['32554.0', '23434.0', '27139.0', '26437.0', '27753.0', '26994.0', '29943.0', '24365.0', '26194.0', '24420.0']
average_cost_program_year (numeric, 2350 distinct): ['18601.0', '25465.0', '15263.0', '20441.0', '18708.0', '14243.0', '15852.0', '12798.0', '17970.0', '12815.0']
tuition_(instate) (numeric, 2982 distinct): ['18048.0', '16010.0', '15495.0', '14040.0', '16565.0', '12114.0', '10240.0', '18000.0', '13663.0', '15995.0']
tuition_(out_of_state) (numeric, 3040 distinct): ['18048.0', '16010.0', '15495.0', '14040.0', '16565.0', '10240.0', '12114.0', '18000.0', '13663.0', '15995.0']
spend_per_student (numeric, 5295 distinct): ['3558.0', '3397.0', '3999.0', '2167.0', '2133.0', '6654.0', '7588.0', '4936.0', '5069.0', '3252.0']
faculty_salary (numeric, 3298 distinct): ['3500.0', '4167.0', '2500.0', '5525.0', '3750.0', '6057.0', '5154.0', '6667.0', '3333.0', '5000.0']
percent_part_time_faculty (numeric, 2333 distinct): ['1.0', '0.5', '0.3333', '0.0', '0.6667', '0.2', '0.25', '0.75', '0.125', '0.1']
completion_rate (numeric, 1913 distinct): ['0.0', '0.5', '0.3333', '0.6667', '1.0', '0.25', '0.2', '0.4', '0.1', '0.2857']
predominant_degree (nominal, 4 distinct): ['Certificate', 'Bachelors', 'Associate', 'Graduate']
highest_degree (nominal, 5 distinct): ['Certificate', 'Graduate', 'Associate', 'Bachelors', 'Non-degree']
ownership (nominal, 3 distinct): ['Private for-profit', 'Public', 'Private nonprofit']
region (nominal, 10 distinct): ['Southeast (AL, AR, FL, GA, KY, LA, MS, NC, SC, TN, VA, WV)', 'Mid East (DE, DC, MD, NJ, NY, PA)', 'Great Lakes (IL, IN, MI, OH, WI)', 'Far West (AK, CA, HI, NV, OR, WA)', 'Southwest (AZ, NM, OK, TX)', 'Plains (IA, KS, MN, MO, NE, ND, SD)', 'New England (CT, ME, MA, NH, RI, VT)', 'Rocky Mountains (CO, ID,MT, UT, WY)', 'Outlying Areas (AS, FM,GU, MH, MP, PR, PW, VI)', 'US Services Schools']
gender (nominal, 3 distinct): ['COED', 'MENONLY', 'WOMENONLY']
carnegie_basic_classification (nominal, 34 distinct): ["Associate's--Private For-profit", "Master's Colleges and Universities (larger programs)", 'Baccalaureate Colleges--Diverse Fields', "Associate's--Public Rural-serving Medium", 'Baccalaureate Colleges--Arts & Sciences', "Master's Colleges and Universities (medium programs)", 'Special Focus Institutions--Theological seminaries, Bible colleges, and other faith-related institutions', "Baccalaureate/Associate's Colleges", "Associate's--Public Rural-serving Large", "Associate's--Public Urban-serving Multicampus"]
carnegie_undergraduate (nominal, 14 distinct): ['Mixed part/full-time two-year', 'Higher full-time two-year', 'Higher part-time two-year', 'Full-time four-year, inclusive', 'Medium full-time two-year', 'Full-time four-year, selective, higher transfer-in', 'Full-time four-year, selective, lower transfer-in', 'Full-time four-year, more selective, lower transfer-in', 'Higher part-time four-year', 'Medium full-time four-year, inclusivestudents with varying levels academic preparation and achievement)']
carnegie_size (nominal, 18 distinct): ['Small 2-year (500 to 1,999)', 'Very small 2-year (less than 500)', 'Medium 2-year (2000 to 4,999)', 'Small 4-year, highly residential (1,000 to 2,999)', 'Very small 4-year, primarily nonresidential (less than 1,000,)', 'Large 2-year ( 5000 to 9,999)', 'Small 4-year, primarily nonresidential (1,000 to 2,999)', 'Medium 4-year, primarily residential (3,000 to 9,999)', 'Medium 4-year, primarily nonresidential (3,000 to 9,999)', 'Small 4-year, primarily residential (1,000 to 2,999)']
religious_affiliation (nominal, 56 distinct): ['Roman Catholic', 'United Methodist', 'Baptist', 'Presbyterian Church (USA)', 'Jewish', 'Interdenominational', 'Evangelical Lutheran Church', 'Southern Baptist', 'Christian Churches and Churches of Christ', 'Other Protestant']
percent_female (numeric, 102 distinct): ['0.67', '0.95', '0.65', '0.61', '0.59', '0.62', '0.63', '0.6', '0.64', '0.57']
agege24 (numeric, 100 distinct): ['0.49', '0.56', '0.47', '0.48', '0.45', '0.43', '0.62', '0.38', '0.52', '0.51']
faminc (numeric, 4778 distinct): ['47023.93', '81964.64', '17218.54', '24120.35', '49787.34', '38521.77', '25464.65', '21757.84', '36711.29', '14773.61']
mean_earnings_6_years (numeric, 495 distinct): ['33700.0', '20600.0', '44300.0', '22900.0', '26800.0', '37100.0', '25700.0', '28300.0', '21100.0', '24500.0']
median_earnings_6_years (numeric, 490 distinct): ['30300.0', '19400.0', '38200.0', '25500.0', '20900.0', '33800.0', '20400.0', '23600.0', '16700.0', '21700.0']
mean_earnings_10_years (numeric, 608 distinct): ['42300.0', '22400.0', '60500.0', '29100.0', '30500.0', '41800.0', '30900.0', '24100.0', '33900.0', '31100.0']
median_earnings_10_years (numeric, 569 distinct): ['38400.0', '19600.0', '53400.0', '25200.0', '26400.0', '38200.0', '29700.0', '29800.0', '22300.0', '27200.0']
'''

CONTEXT = "US Colleges and their grants"
TARGET = CuratedTarget(raw_name="percent_pell_grant", new_name="Percent Pell Grant", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []