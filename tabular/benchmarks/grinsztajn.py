from typing import List

from tabular.benchmarks.utils import get_suite_datasets
from tabular.datasets.tabular_datasets import OpenMLDatasetID


# Why do tree-based models still outperform deep learning on tabular data?
# https://arxiv.org/abs/2207.08815
GRINSZTAJN = [OpenMLDatasetID.BIN_ANONYM_ALBERT,
              OpenMLDatasetID.BIN_ANONYM_BIORESPONSE,
              OpenMLDatasetID.BIN_CONSUMER_ELECTRICITY_PRICE_TREND,
              OpenMLDatasetID.BIN_FINANCIAL_BANK_MARKETING,
              OpenMLDatasetID.BIN_FINANCIAL_CC_TAIWAN_CREDIT_DEFAULT,
              OpenMLDatasetID.BIN_FINANCIAL_CREDIT_FICO_HELOC,
              OpenMLDatasetID.BIN_FINANCIAL_CREDIT_GIVE_ME_SOME,
              OpenMLDatasetID.BIN_SCIENCE_EYE_MOVEMENT,
              OpenMLDatasetID.BIN_SCIENCE_MAGIC_TELESCOPE,
              OpenMLDatasetID.BIN_SCIENCE_PARTICLE_HIGGS,
              OpenMLDatasetID.BIN_SCIENCE_PARTICLE_MINIBOONE,
              OpenMLDatasetID.BIN_SOCIAL_COMPASS_TWO_YEARS_OFFEND,
              OpenMLDatasetID.BIN_TRANSPORTATION_ROAD_SAFETY_GENDER,
              OpenMLDatasetID.REG_ANONYM_HOUSE_16H,
              OpenMLDatasetID.MUL_ANONYM_POL,
              OpenMLDatasetID.REG_ANONYM_YPROP,
              OpenMLDatasetID.REG_COMPUTERS_CPU_ACTIVITY,
              OpenMLDatasetID.REG_CONSUMER_DIAMONDS_PRICES,
              OpenMLDatasetID.REG_CONSUMER_MEDICAL_CHARGES,
              OpenMLDatasetID.REG_HOUSES_BRAZILIAN_HOUSES,
              OpenMLDatasetID.REG_HOUSES_CALIFORNIA_HOUSES,
              OpenMLDatasetID.REG_HOUSES_MIAMI,
              OpenMLDatasetID.REG_HOUSES_SEATTLE_KINGS_COUNTY,
              OpenMLDatasetID.REG_NATURE_ABALONE_FISH_RINGS,
              OpenMLDatasetID.REG_SCIENCE_SULFUR,
              OpenMLDatasetID.REG_SCIENCE_SUPERCONDUCTIVITY,
              OpenMLDatasetID.REG_TRANSPORTATION_FLIGHT_SIMULATION_ELEVATORS,
              OpenMLDatasetID.REG_TRANSPORTATION_US_BIKE_SHARING_DEMAND,
              OpenMLDatasetID.REG_TRANSPORTATION_NYC_TAXI_TIP,
              OpenMLDatasetID.REG_TRANSPORTATION_ZURICH_PUBLIC_TRANSPORT_DELAY,
              OpenMLDatasetID.MUL_ANONYM_JANNIS,
              OpenMLDatasetID.MUL_HEALTHCARE_DIABETES_US130,
              OpenMLDatasetID.MUL_NATURE_FOREST_COVERTYPE]


def get_grinsztajn_ids() -> List[int]:
    num_cls = get_suite_datasets(337)
    num_reg = get_suite_datasets(336)
    both_reg = get_suite_datasets(335)
    both_cls = get_suite_datasets(334)
    all_datasets = num_cls + num_reg + both_reg + both_cls
    return sorted(set(all_datasets))