from tabstar_paper.datasets.curation_objects import CuratedFeature, CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
'''

CONTEXT = "CRM data from French Telecom company for Upselling"
TARGET = CuratedTarget(raw_name="upselling", new_name="Propensity to Upsell", task_type=SupervisedTask.BINARY,
                       label_mapping={"-1": "Won't upsell", "1": "Will upsell"})
COLS_TO_DROP = []
FEATURES = []