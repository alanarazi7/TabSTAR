from typing import List

from tabular.benchmarks.utils import get_suite_datasets
from tabular.datasets.tabular_datasets import OpenMLDatasetID

# The exact datasets IDs where not necessarily included, as they are duplicates of AMLB which is more well known
# Specifically, we excluded one of the wine datasets present in this benchmark due to this deduplication
CTR23_REG = [OpenMLDatasetID.REG_ANONYM_GEOGRAPHICAL_ORIGIN_OF_MUSIC,
             OpenMLDatasetID.REG_COMPUTERS_CPU_ACTIVITY,
             OpenMLDatasetID.REG_COMPUTERS_GAMING_FRAMES_FPS_BENCHMARK,
             OpenMLDatasetID.REG_COMPUTERS_PUMA_ROBOT_ARM,
             OpenMLDatasetID.REG_COMPUTERS_ROBOT_KIN8NM,
             OpenMLDatasetID.REG_COMPUTERS_ROBOT_SARCOS,
             OpenMLDatasetID.REG_COMPUTERS_YOUTUBE_VIDEO_TRANSCODING,
             OpenMLDatasetID.REG_CONSUMER_DIAMONDS_PRICES,
             OpenMLDatasetID.REG_FINANCIAL_INSURANCE_HEALTH_HOURS_WORKED_WIFE,
             OpenMLDatasetID.REG_HOUSES_BRAZILIAN_HOUSES,
             OpenMLDatasetID.REG_HOUSES_CALIFORNIA_HOUSES,
             OpenMLDatasetID.REG_HOUSES_MIAMI,
             OpenMLDatasetID.REG_HOUSES_SEATTLE_KINGS_COUNTY,
             OpenMLDatasetID.REG_PROFESSIONAL_CPS88_WAGES,
             OpenMLDatasetID.REG_PROFESSIONAL_STUDENT_PERFORMANCE_PORTUGAL,
             OpenMLDatasetID.REG_NATURE_ABALONE_FISH_RINGS,
             OpenMLDatasetID.REG_NATURE_FISH_TOXICITY,
             OpenMLDatasetID.REG_NATURE_FOREST_FIRES,
             OpenMLDatasetID.MUL_NATURE_SOLAR_FLARES,
             OpenMLDatasetID.REG_SCIENCE_AIRFOIL_SELF_NOISE,
             OpenMLDatasetID.REG_SCIENCE_AUCTION_VERIFICATION,
             OpenMLDatasetID.REG_SCIENCE_CONCRETE_COMPRESSIVE_STRENGTH,
             OpenMLDatasetID.REG_SCIENCE_ENERGY_EFFICIENCY,
             OpenMLDatasetID.REG_SCIENCE_GRID_STABILITY,
             OpenMLDatasetID.REG_SCIENCE_PHYSIOCHEMICAL_PROTEIN,
             OpenMLDatasetID.REG_SCIENCE_SUPERCONDUCTIVITY,
             OpenMLDatasetID.REG_SCIENCE_WAVE_ENERGY,
             OpenMLDatasetID.REG_SOCIAL_OCCUPATION_MOBILITY_SOCMOB,
             OpenMLDatasetID.REG_SOCIAL_VOTING_SPACE_GEOGRAPHIC_ANALYSIS,
             OpenMLDatasetID.REG_SPORTS_FIFA22_WAGES,
             OpenMLDatasetID.REG_SPORTS_MONEYBALL,
             OpenMLDatasetID.REG_TRANSPORTATION_CAR_GM_PRICE,
             OpenMLDatasetID.REG_TRANSPORTATION_NAVAL_PROPULSION_PLANT]

def get_ctr23_datasets() -> List[OpenMLDatasetID]:
    datasets = get_suite_datasets(sid=353, name='OpenML-CTR23 - A curated tabular regression benchmarking suite',
                                  n_datasets=35)
    openml_datasets = [OpenMLDatasetID(d) for d in datasets]
    return openml_datasets