from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: college_scorecard
====
Examples: 124699
====
URL: https://www.openml.org/search?type=data&id=46674
====
Description: Higher education is increasingly critical to securing strong job and income opportunities for persons in the United States. At the same time, the cost of obtaining a four-year college degree is extremely high: The average cost of college* in the United States is $35,551 per student per year, including books, supplies, and daily living expenses and this cost has more than doubled in the 21st century alone, with an annual growth rate of 7.1% (cite).

However, not all institutions have similar outcomes for students. Graduation rates across institutions in the U.S. vary widely, and failure to complete a degree can leave a student with significant debt and a reduced ability to repay it. Understanding factors related to degree completion is an area of active research.

For this task, our goal is to predict whether an institution has a low completion rate, based on other characteristics of that institution. While the definition of a ''low'' completion rate is ultimately subjective and context-dependent, we use a thredhold of 50%, which is approximately equivalent to the median graduate rate across the institutions in the dataset. We use the completion rate for first-time, full-time students at four-year institutions (150 percentage of expected time to completion/6 years).

We constrained the feature names to 64 characters.
Please refer to the original data source for more information on the features and their meanings.
paper_url = "https://papers.nips.cc/paper_files/paper/2023/file/a76a757ed479a1e6a5f8134bea492f83-Paper-Datasets_and_Benchmarks.pdf"
original_data_url = "https://collegescorecard.ed.gov/data/"

Column names are restricted to 64 characters and special characters are replaced by an underscore.
====
Target Variable: Completion_rate_for_first_time_full_time_target (numeric, 2 distinct): ['0', '1']
====
Features:

Predominant_degree_awarded__recoded_0s_and_4s (numeric, 3 distinct): ['1.0', '0.0', '-1.0']
Number_of_branch_campuses (numeric, 12 distinct): ['0', '1', '2', '3', '5', '4', '7', '6', '10', '11']
Degree_of_urbanization_of_institution (numeric, 1 distinct): ['-1.0']
Admission_rate (numeric, 92 distinct): ['-1.0', '90.0', '19.0', '55.0', '5.0', '14.0', '28.0', '30.0', '83.0', '40.0']
Admission_rate_for_all_campuses_rolled_up_to_the_6_digi (numeric, 93 distinct): ['-1.0', '91.0', '18.0', '48.0', '4.0', '29.0', '16.0', '72.0', '27.0', '13.0']
Midpoint_of_SAT_scores_at_the_institution__critical_rea (numeric, 64 distinct): ['-1.0', '22.0', '21.0', '25.0', '23.0', '26.0', '20.0', '16.0', '28.0', '34.0']
Midpoint_of_SAT_scores_at_the_institution__math (numeric, 58 distinct): ['-1.0', '20.0', '24.0', '23.0', '19.0', '18.0', '27.0', '25.0', '17.0', '21.0']
Midpoint_of_SAT_scores_at_the_institution__writing (numeric, 60 distinct): ['-1.0', '19.0', '17.0', '15.0', '18.0', '25.0', '26.0', '24.0', '16.0', '28.0']
Midpoint_of_the_ACT_cumulative_score (numeric, 19 distinct): ['-1.0', '7.0', '6.0', '8.0', '9.0', '5.0', '10.0', '4.0', '11.0', '12.0']
Midpoint_of_the_ACT_English_score (numeric, 20 distinct): ['-1.0', '8.0', '7.0', '9.0', '6.0', '10.0', '5.0', '11.0', '4.0', '12.0']
Midpoint_of_the_ACT_math_score (numeric, 19 distinct): ['-1.0', '6.0', '7.0', '5.0', '8.0', '4.0', '9.0', '10.0', '3.0', '11.0']
Midpoint_of_the_ACT_writing_score (numeric, 21 distinct): ['-1.0', '3.0', '5.0', '4.0', '2.0', '0.0', '15.0', '16.0', '7.0', '14.0']
Percentage_of_degrees_awarded_in_Agriculture__Agricultu (numeric, 11 distinct): ['0.0', '-1.0', '8.0', '5.0', '9.0', '2.0', '4.0', '3.0', '6.0', '1.0']
Percentage_of_degrees_awarded_in_Natural_Resources_And_ (numeric, 11 distinct): ['0.0', '-1.0', '5.0', '2.0', '7.0', '6.0', '4.0', '8.0', '3.0', '9.0']
Percentage_of_degrees_awarded_in_Architecture_And_Relat (numeric, 5 distinct): ['0.0', '-1.0', '3.0', '2.0', '1.0']
Percentage_of_degrees_awarded_in_Area__Ethnic__Cultural (numeric, 9 distinct): ['0.0', '-1.0', '1.0', '3.0', '4.0', '5.0', '7.0', '6.0', '2.0']
Percentage_of_degrees_awarded_in_Communication__Journal (numeric, 22 distinct): ['0.0', '-1.0', '5.0', '15.0', '9.0', '16.0', '20.0', '7.0', '11.0', '13.0']
Percentage_of_degrees_awarded_in_Communications_Technol (numeric, 10 distinct): ['0.0', '-1.0', '8.0', '1.0', '3.0', '4.0', '5.0', '2.0', '7.0', '6.0']
Percentage_of_degrees_awarded_in_Computer_And_Informati (numeric, 46 distinct): ['0.0', '-1.0', '27.0', '2.0', '16.0', '23.0', '29.0', '18.0', '4.0', '9.0']
Percentage_of_degrees_awarded_in_Personal_And_Culinary_ (numeric, 16 distinct): ['0.0', '14.0', '-1.0', '7.0', '12.0', '2.0', '4.0', '5.0', '8.0', '6.0']
Percentage_of_degrees_awarded_in_Education (numeric, 30 distinct): ['0.0', '-1.0', '14.0', '19.0', '16.0', '21.0', '3.0', '5.0', '23.0', '15.0']
Percentage_of_degrees_awarded_in_Engineering (numeric, 13 distinct): ['0.0', '-1.0', '8.0', '5.0', '4.0', '1.0', '7.0', '10.0', '6.0', '9.0']
Percentage_of_degrees_awarded_in_Engineering_Technologi (numeric, 28 distinct): ['0.0', '-1.0', '25.0', '24.0', '7.0', '23.0', '5.0', '1.0', '11.0', '26.0']
Percentage_of_degrees_awarded_in_Foreign_Languages__Lit (numeric, 18 distinct): ['0.0', '-1.0', '8.0', '10.0', '4.0', '5.0', '6.0', '12.0', '16.0', '2.0']
Percentage_of_degrees_awarded_in_Family_And_Consumer_Sc (numeric, 19 distinct): ['0.0', '-1.0', '11.0', '9.0', '3.0', '17.0', '15.0', '14.0', '2.0', '16.0']
Percentage_of_degrees_awarded_in_Legal_Professions_And_ (numeric, 18 distinct): ['0.0', '-1.0', '13.0', '6.0', '8.0', '4.0', '15.0', '12.0', '1.0', '16.0']
Percentage_of_degrees_awarded_in_English_Language_And_L (numeric, 24 distinct): ['0.0', '-1.0', '11.0', '15.0', '7.0', '12.0', '16.0', '4.0', '19.0', '9.0']
Percentage_of_degrees_awarded_in_Liberal_Arts_And_Scien (numeric, 33 distinct): ['0.0', '-1.0', '4.0', '1.0', '26.0', '6.0', '11.0', '7.0', '30.0', '14.0']
Percentage_of_degrees_awarded_in_Library_Science (numeric, 2 distinct): ['0.0', '-1.0']
Percentage_of_degrees_awarded_in_Biological_And_Biomedi (numeric, 25 distinct): ['0.0', '-1.0', '12.0', '15.0', '20.0', '14.0', '8.0', '21.0', '19.0', '7.0']
Percentage_of_degrees_awarded_in_Mathematics_And_Statis (numeric, 22 distinct): ['0.0', '-1.0', '10.0', '6.0', '13.0', '12.0', '16.0', '17.0', '5.0', '20.0']
Percentage_of_degrees_awarded_in_Military_Technologies_ (numeric, 2 distinct): ['0.0', '-1.0']
Percentage_of_degrees_awarded_in_Multi_Interdisciplinar (numeric, 18 distinct): ['0.0', '-1.0', '4.0', '9.0', '14.0', '3.0', '2.0', '15.0', '12.0', '11.0']
Percentage_of_degrees_awarded_in_Parks__Recreation__Lei (numeric, 15 distinct): ['0.0', '-1.0', '13.0', '7.0', '12.0', '10.0', '9.0', '11.0', '5.0', '2.0']
Percentage_of_degrees_awarded_in_Philosophy_And_Religio (numeric, 16 distinct): ['0.0', '-1.0', '7.0', '6.0', '3.0', '2.0', '5.0', '11.0', '13.0', '9.0']
Percentage_of_degrees_awarded_in_Theology_And_Religious (numeric, 7 distinct): ['0.0', '-1.0', '5.0', '4.0', '3.0', '2.0', '1.0']
Percentage_of_degrees_awarded_in_Physical_Sciences (numeric, 21 distinct): ['0.0', '-1.0', '3.0', '8.0', '13.0', '11.0', '19.0', '12.0', '5.0', '17.0']
Percentage_of_degrees_awarded_in_Science_Technologies_T (numeric, 4 distinct): ['0.0', '-1.0', '2.0', '1.0']
Percentage_of_degrees_awarded_in_Psychology (numeric, 25 distinct): ['0.0', '-1.0', '8.0', '17.0', '20.0', '14.0', '16.0', '18.0', '5.0', '21.0']
Percentage_of_degrees_awarded_in_Homeland_Security__Law (numeric, 28 distinct): ['0.0', '-1.0', '19.0', '4.0', '25.0', '18.0', '26.0', '23.0', '10.0', '3.0']
Percentage_of_degrees_awarded_in_Public_Administration_ (numeric, 18 distinct): ['0.0', '-1.0', '16.0', '3.0', '12.0', '15.0', '11.0', '5.0', '13.0', '7.0']
Percentage_of_degrees_awarded_in_Social_Sciences (numeric, 25 distinct): ['0.0', '-1.0', '17.0', '23.0', '16.0', '12.0', '4.0', '8.0', '6.0', '5.0']
Percentage_of_degrees_awarded_in_Construction_Trades (numeric, 13 distinct): ['0.0', '-1.0', '1.0', '6.0', '7.0', '9.0', '8.0', '4.0', '10.0', '11.0']
Percentage_of_degrees_awarded_in_Mechanic_And_Repair_Te (numeric, 20 distinct): ['0.0', '-1.0', '10.0', '14.0', '8.0', '4.0', '16.0', '2.0', '3.0', '9.0']
Percentage_of_degrees_awarded_in_Precision_Production (numeric, 14 distinct): ['0.0', '-1.0', '2.0', '5.0', '7.0', '1.0', '10.0', '8.0', '11.0', '12.0']
Percentage_of_degrees_awarded_in_Transportation_And_Mat (numeric, 7 distinct): ['0.0', '-1.0', '2.0', '4.0', '5.0', '1.0', '3.0']
Percentage_of_degrees_awarded_in_Visual_And_Performing_ (numeric, 33 distinct): ['0.0', '-1.0', '31.0', '21.0', '17.0', '20.0', '9.0', '2.0', '8.0', '4.0']
Percentage_of_degrees_awarded_in_Health_Professions_And (numeric, 52 distinct): ['0.0', '50.0', '-1.0', '38.0', '36.0', '19.0', '10.0', '2.0', '43.0', '29.0']
Percentage_of_degrees_awarded_in_Business__Management__ (numeric, 57 distinct): ['0.0', '-1.0', '38.0', '11.0', '23.0', '27.0', '37.0', '35.0', '19.0', '47.0']
Percentage_of_degrees_awarded_in_History (numeric, 21 distinct): ['0.0', '-1.0', '6.0', '13.0', '17.0', '2.0', '4.0', '11.0', '9.0', '10.0']
Enrollment_of_undergraduate_degree_seeking_students (numeric, 101 distinct): ['-1.0', '3.0', '15.0', '8.0', '17.0', '21.0', '5.0', '6.0', '33.0', '11.0']
Enrollment_of_all_undergraduate_students (numeric, 101 distinct): ['-1.0', '37.0', '3.0', '15.0', '10.0', '17.0', '68.0', '12.0', '8.0', '59.0']
Total_share_of_enrollment_of_undergraduate_degree_seeki (numeric, 77 distinct): ['-1.0', '0.0', '75.0', '48.0', '63.0', '60.0', '70.0', '5.0', '3.0', '65.0']
Total_share_of_enrollment_of_undergraduate_degree_seeki_1 (numeric, 74 distinct): ['-1.0', '0.0', '18.0', '33.0', '8.0', '21.0', '65.0', '34.0', '31.0', '27.0']
Total_share_of_enrollment_of_undergraduate_degree_seeki_2 (numeric, 72 distinct): ['-1.0', '0.0', '70.0', '32.0', '30.0', '8.0', '35.0', '10.0', '18.0', '34.0']
Total_share_of_enrollment_of_undergraduate_degree_seeki_3 (numeric, 62 distinct): ['-1.0', '0.0', '25.0', '10.0', '4.0', '13.0', '8.0', '34.0', '51.0', '20.0']
Total_share_of_enrollment_of_undergraduate_degree_seeki_4 (numeric, 52 distinct): ['-1.0', '0.0', '22.0', '6.0', '24.0', '11.0', '13.0', '14.0', '7.0', '29.0']
Total_share_of_enrollment_of_undergraduate_degree_seeki_5 (numeric, 36 distinct): ['-1.0', '0.0', '11.0', '13.0', '14.0', '23.0', '17.0', '8.0', '28.0', '20.0']
Total_share_of_enrollment_of_undergraduate_degree_seeki_6 (numeric, 52 distinct): ['-1.0', '0.0', '30.0', '40.0', '16.0', '26.0', '20.0', '24.0', '41.0', '23.0']
Total_share_of_enrollment_of_undergraduate_degree_seeki_7 (numeric, 47 distinct): ['0.0', '-1.0', '7.0', '4.0', '10.0', '20.0', '45.0', '14.0', '24.0', '25.0']
Total_share_of_enrollment_of_undergraduate_degree_seeki_8 (numeric, 60 distinct): ['0.0', '-1.0', '12.0', '22.0', '31.0', '25.0', '5.0', '10.0', '3.0', '18.0']
Total_share_of_enrollment_of_undergraduate_degree_seeki_9 (numeric, 89 distinct): ['-1.0', '0.0', '87.0', '26.0', '71.0', '34.0', '77.0', '83.0', '44.0', '69.0']
Total_share_of_enrollment_of_undergraduate_degree_seeki_10 (numeric, 85 distinct): ['-1.0', '0.0', '62.0', '56.0', '63.0', '75.0', '8.0', '12.0', '3.0', '82.0']
Total_share_of_enrollment_of_undergraduate_degree_seeki_11 (numeric, 73 distinct): ['-1.0', '0.0', '10.0', '13.0', '7.0', '17.0', '34.0', '30.0', '3.0', '20.0']
Total_share_of_enrollment_of_undergraduate_degree_seeki_12 (numeric, 60 distinct): ['-1.0', '0.0', '11.0', '17.0', '13.0', '15.0', '9.0', '29.0', '8.0', '18.0']
Total_share_of_enrollment_of_undergraduate_degree_seeki_13 (numeric, 81 distinct): ['-1.0', '0.0', '79.0', '19.0', '35.0', '21.0', '14.0', '33.0', '31.0', '11.0']
Total_share_of_enrollment_of_undergraduate_students_who (numeric, 50 distinct): ['-1.0', '0.0', '8.0', '25.0', '4.0', '36.0', '16.0', '6.0', '3.0', '47.0']
Total_share_of_enrollment_of_undergraduate_students_who_1 (numeric, 52 distinct): ['-1.0', '0.0', '8.0', '24.0', '4.0', '28.0', '31.0', '42.0', '5.0', '36.0']
Total_share_of_enrollment_of_undergraduate_students_who_2 (numeric, 93 distinct): ['-1.0', '0.0', '91.0', '86.0', '12.0', '58.0', '83.0', '19.0', '41.0', '60.0']
Total_share_of_enrollment_of_undergraduate_students_who_3 (numeric, 89 distinct): ['-1.0', '0.0', '25.0', '17.0', '71.0', '15.0', '9.0', '8.0', '68.0', '60.0']
Total_share_of_enrollment_of_undergraduate_students_who_4 (numeric, 75 distinct): ['-1.0', '0.0', '13.0', '10.0', '8.0', '5.0', '40.0', '23.0', '29.0', '61.0']
Total_share_of_enrollment_of_undergraduate_students_who_5 (numeric, 61 distinct): ['-1.0', '0.0', '3.0', '10.0', '7.0', '17.0', '14.0', '24.0', '30.0', '34.0']
Total_share_of_enrollment_of_undergraduate_students_who_6 (numeric, 82 distinct): ['-1.0', '0.0', '80.0', '33.0', '79.0', '36.0', '4.0', '40.0', '22.0', '5.0']
Share_of_undergraduate__degree__certificate_seeking_stu (numeric, 73 distinct): ['0.0', '-1.0', '55.0', '33.0', '21.0', '8.0', '29.0', '6.0', '37.0', '52.0']
Share_of_undergraduate__degree__certificate_seeking_stu_1 (numeric, 74 distinct): ['-1.0', '0.0', '3.0', '11.0', '51.0', '56.0', '72.0', '43.0', '47.0', '45.0']
Average_net_price_for_the_largest_program_at_the_instit (numeric, 101 distinct): ['-1.0', '61.0', '65.0', '4.0', '81.0', '45.0', '17.0', '88.0', '14.0', '8.0']
Average_cost_of_attendance__academic_year_institutions (numeric, 101 distinct): ['-1.0', '88.0', '93.0', '99.0', '83.0', '61.0', '85.0', '64.0', '87.0', '73.0']
Average_cost_of_attendance__program_year_institutions (numeric, 101 distinct): ['-1.0', '39.0', '23.0', '49.0', '45.0', '41.0', '63.0', '44.0', '60.0', '1.0']
In_state_tuition_and_fees (numeric, 101 distinct): ['-1.0', '83.0', '47.0', '79.0', '63.0', '86.0', '94.0', '95.0', '97.0', '23.0']
Out_of_state_tuition_and_fees (numeric, 101 distinct): ['-1.0', '75.0', '94.0', '84.0', '80.0', '70.0', '57.0', '93.0', '97.0', '65.0']
Tuition_and_fees_for_program_year_institutions (numeric, 101 distinct): ['-1.0', '46.0', '40.0', '35.0', '43.0', '77.0', '51.0', '22.0', '14.0', '69.0']
Net_tuition_revenue_per_full_time_equivalent_student (numeric, 101 distinct): ['-1.0', '85.0', '92.0', '50.0', '87.0', '94.0', '59.0', '79.0', '93.0', '84.0']
Instructional_expenditures_per_full_time_equivalent_stu (numeric, 101 distinct): ['-1.0', '80.0', '87.0', '84.0', '76.0', '91.0', '56.0', '60.0', '86.0', '77.0']
Average_faculty_salary (numeric, 100 distinct): ['-1.0', '0.0', '85.0', '81.0', '86.0', '65.0', '93.0', '78.0', '88.0', '50.0']
Proportion_of_faculty_that_is_full_time (numeric, 84 distinct): ['-1.0', '82.0', '43.0', '65.0', '16.0', '58.0', '62.0', '26.0', '71.0', '67.0']
Percentage_of_undergraduates_who_receive_a_Pell_Grant (numeric, 101 distinct): ['-1.0', '66.0', '88.0', '63.0', '11.0', '47.0', '48.0', '60.0', '59.0', '30.0']
Share_of_students_who_received_a_federal_loan_while_in_ (numeric, 51 distinct): ['-1.0', '0.0', '46.0', '47.0', '49.0', '45.0', '48.0', '44.0', '43.0', '42.0']
Share_of_students_who_received_a_Pell_Grant_while_in_sc (numeric, 69 distinct): ['-1.0', '67.0', '0.0', '11.0', '49.0', '48.0', '51.0', '54.0', '50.0', '53.0']
Average_age_of_entry__via_SSA_data (numeric, 101 distinct): ['-1.0', '61.0', '64.0', '15.0', '41.0', '88.0', '35.0', '39.0', '48.0', '55.0']
Average_of_the_age_of_entry_squared (numeric, 101 distinct): ['-1.0', '87.0', '80.0', '8.0', '65.0', '88.0', '34.0', '86.0', '38.0', '64.0']
Percent_of_students_over_23_at_entry (numeric, 79 distinct): ['-1.0', '43.0', '51.0', '47.0', '24.0', '42.0', '53.0', '44.0', '45.0', '46.0']
Share_of_female_students__via_SSA_data (numeric, 67 distinct): ['-1.0', '30.0', '26.0', '29.0', '61.0', '28.0', '25.0', '32.0', '27.0', '33.0']
Share_of_married_students (numeric, 50 distinct): ['-1.0', '23.0', '25.0', '21.0', '22.0', '20.0', '24.0', '27.0', '18.0', '19.0']
Share_of_dependent_students (numeric, 82 distinct): ['-1.0', '15.0', '18.0', '13.0', '17.0', '16.0', '14.0', '12.0', '19.0', '24.0']
Share_of_veteran_students (numeric, 18 distinct): ['-1.0', '0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0']
Share_of_first_generation_students (numeric, 58 distinct): ['-1.0', '30.0', '33.0', '31.0', '36.0', '34.0', '35.0', '37.0', '32.0', '38.0']
Average_family_income (numeric, 101 distinct): ['-1.0', '11.0', '23.0', '43.0', '93.0', '81.0', '73.0', '92.0', '15.0', '99.0']
Median_family_income (numeric, 99 distinct): ['-1.0', '0.0', '11.0', '26.0', '90.0', '23.0', '40.0', '15.0', '69.0', '84.0']
Percent_of_the_population_from_students__zip_codes_that (numeric, 101 distinct): ['-1.0', '17.0', '36.0', '4.0', '91.0', '53.0', '88.0', '54.0', '66.0', '19.0']
Percent_of_the_population_from_students__zip_codes_that_1 (numeric, 101 distinct): ['-1.0', '16.0', '69.0', '14.0', '47.0', '4.0', '91.0', '30.0', '98.0', '17.0']
Percent_of_the_population_from_students__zip_codes_that_2 (numeric, 101 distinct): ['-1.0', '25.0', '4.0', '16.0', '45.0', '33.0', '60.0', '43.0', '18.0', '28.0']
Percent_of_the_population_from_students__zip_codes_that_3 (numeric, 101 distinct): ['-1.0', '28.0', '10.0', '7.0', '22.0', '43.0', '39.0', '13.0', '67.0', '32.0']
Percent_of_the_population_from_students__zip_codes_with (numeric, 101 distinct): ['-1.0', '13.0', '40.0', '65.0', '99.0', '36.0', '70.0', '20.0', '73.0', '96.0']
Percent_of_the_population_from_students__zip_codes_over (numeric, 101 distinct): ['-1.0', '52.0', '13.0', '60.0', '35.0', '30.0', '54.0', '56.0', '27.0', '11.0']
Percent_of_the_population_from_students__zip_codes_that_4 (numeric, 101 distinct): ['-1.0', '37.0', '82.0', '98.0', '33.0', '7.0', '73.0', '49.0', '86.0', '40.0']
Median_household_income (numeric, 101 distinct): ['-1.0', '29.0', '85.0', '60.0', '73.0', '7.0', '94.0', '53.0', '98.0', '91.0']
Poverty_rate__via_Census_data (numeric, 101 distinct): ['-1.0', '17.0', '46.0', '51.0', '40.0', '9.0', '26.0', '20.0', '67.0', '53.0']
Unemployment_rate__via_Census_data (numeric, 101 distinct): ['-1.0', '40.0', '58.0', '54.0', '20.0', '16.0', '65.0', '72.0', '13.0', '60.0']
State_postcode (numeric, 59 distinct): ['5', '39', '43', '50', '10', '40', '17', '28', '22', '26']
Accreditor_for_institution (numeric, 91 distinct): ['89.0', '66.0', '76.0', '82.0', '11.0', '59.0', '46.0', '24.0', '72.0', '78.0']
Flag_for_main_campus (numeric, 2 distinct): ['0', '1']
Highest_degree_awarded (string, 5 distinct): ['Associate degree', "Bachelor's degree", 'Non-degree-granting', 'Certificate degree', 'Graduate degree']
Control_of_institution (numeric, 4 distinct): ['0', '2', '1', '3']
Region__IPEDS (numeric, 11 distinct): ['7', '2', '1', '0', '8', '5', '3', '6', '4', '9']
Locale_of_institution (numeric, 13 distinct): ['12', '6', '0', '2', '1', '9', '11', '4', '7', '8']
Carnegie_Classification____size_and_setting (numeric, 19 distinct): ['18', '9', '14', '4', '10', '16', '0', '11', '7', '6']
Flag_for_Historically_Black_College_and_University (numeric, 3 distinct): ['2', '0', '1']
Flag_for_distance_education_only_education (numeric, 3 distinct): ['2', '1', '0']
'''

CONTEXT = "Academic Institution Completion Rate"
TARGET = CuratedTarget(raw_name="Completion_rate_for_first_time_full_time_target",
                       new_name="Completion for First Time Full Time Students", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []