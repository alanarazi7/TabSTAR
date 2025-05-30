from tabular.datasets.tabular_datasets import KaggleDatasetID, UrlDatasetID, OpenMLDatasetID


DATA2EXAMPLES = {
OpenMLDatasetID.BIN_ANONYM_ADA: 4147,
OpenMLDatasetID.BIN_ANONYM_ALBERT: 425240,
OpenMLDatasetID.BIN_ANONYM_APS_FAILURE: 76000,
OpenMLDatasetID.BIN_ANONYM_ARCENE: 100,
OpenMLDatasetID.BIN_ANONYM_AUSTRALIAN_CREDIT_APPROVAL: 690,
OpenMLDatasetID.BIN_ANONYM_AUTOUNIV_AU1: 1000,
OpenMLDatasetID.BIN_ANONYM_BIORESPONSE: 3751,
OpenMLDatasetID.BIN_ANONYM_CHRISTINE: 5418,
OpenMLDatasetID.BIN_ANONYM_GAMETES: 1600,
OpenMLDatasetID.BIN_ANONYM_GINA: 3153,
OpenMLDatasetID.BIN_ANONYM_GUILLERMO: 20000,
OpenMLDatasetID.BIN_ANONYM_JASMINE: 2984,
OpenMLDatasetID.BIN_ANONYM_KDDCUP_98_DIRECT_MAIL: 82318,
OpenMLDatasetID.BIN_ANONYM_KDDCUP_09_APPETENCY: 50000,
OpenMLDatasetID.BIN_ANONYM_KDDCUP_09_UPSELLING: 50000,
OpenMLDatasetID.BIN_ANONYM_MADELINE: 3140,
OpenMLDatasetID.BIN_ANONYM_MADELONE: 2600,
OpenMLDatasetID.BIN_ANONYM_MAMMOGRAPHY: 11183,
OpenMLDatasetID.BIN_ANONYM_MONKS_PROBLEM_2: 601,
OpenMLDatasetID.BIN_ANONYM_NUMERAI_28_6: 96320,
OpenMLDatasetID.BIN_ANONYM_OIL_SPILL: 937,
OpenMLDatasetID.BIN_ANONYM_PHILIPPINE: 5832,
OpenMLDatasetID.BIN_ANONYM_PORTO_SEGURO: 595212,
OpenMLDatasetID.BIN_ANONYM_RICARDO: 20000,
OpenMLDatasetID.BIN_ANONYM_SATELLITE: 5100,
OpenMLDatasetID.BIN_ANONYM_SYLVINE: 5124,
OpenMLDatasetID.BIN_ANONYM_TWONORM: 7400,
OpenMLDatasetID.BIN_ANONYM_XD6: 973,
OpenMLDatasetID.BIN_COMPUTERS_CPU_TOKYO1: 959,
OpenMLDatasetID.BIN_COMPUTERS_IMAGE_BANK_NOTE_AUTHENTICATION: 1372,
OpenMLDatasetID.BIN_COMPUTERS_IMAGE_SKIN_SEGMENTATION: 245057,
OpenMLDatasetID.BIN_COMPUTERS_MOZILLA: 15545,
OpenMLDatasetID.BIN_COMPUTERS_KC1_CODE_DEFECTIONS: 2109,
OpenMLDatasetID.BIN_COMPUTERS_PC4_CODE_DEFECTIONS: 1458,
OpenMLDatasetID.BIN_COMPUTERS_PHISHING_URL_WEBSITES: 11000,
OpenMLDatasetID.BIN_COMPUTERS_PHISHING_WEBSITE_HUDDERSFIELD: 11055,
OpenMLDatasetID.BIN_CONSUMER_BLASTCHAR_TELCOM_CHURN: 7043,
OpenMLDatasetID.BIN_CONSUMER_CHURN_TELCO_EUROPA: 190776,
OpenMLDatasetID.BIN_CONSUMER_CHURN_TELCO_NIGERIA: 1401,
OpenMLDatasetID.BIN_CONSUMER_CHURN_TELCO_SOUTH_ASIA: 2000,
OpenMLDatasetID.BIN_CONSUMER_CHURN_TELEPHONY: 5000,
OpenMLDatasetID.BIN_CONSUMER_CLICK_PREDICTION: 39948,
OpenMLDatasetID.BIN_CONSUMER_DRESS_DALES: 500,
OpenMLDatasetID.BIN_CONSUMER_ELECTRICITY_PRICE_TREND: 38474,
OpenMLDatasetID.BIN_CONSUMER_HOTEL_REVIEW: 38932,
OpenMLDatasetID.BIN_CONSUMER_INTERNET_ADVERTISEMENTS: 3279,
OpenMLDatasetID.BIN_CONSUMER_MOBILE_CHURN: 66469,
OpenMLDatasetID.BIN_CONSUMER_NEWSPAPER_CHURN: 15855,
OpenMLDatasetID.BIN_CONSUMER_ONLINE_SHOPPERS_PURCHASE_INTENTION: 12330,
OpenMLDatasetID.BIN_CONSUMER_TOUR_TRAVEL_CHURN: 954,
OpenMLDatasetID.BIN_CONSUMER_WHOLESALE_CUSTOMERS_PORTUGAL: 440,
OpenMLDatasetID.BIN_FINANCIAL_ACCOUNT_DEBT_SOUTH_AFRICA: 138509,
OpenMLDatasetID.BIN_FINANCIAL_ADULT_INCOME: 48842,
OpenMLDatasetID.BIN_FINANCIAL_AUDIT_RISK: 1552,
OpenMLDatasetID.BIN_FINANCIAL_BANK_ACCOUNT_FRAUD_BAF: 1000000,
OpenMLDatasetID.BIN_FINANCIAL_BANK_CUSTOMER_CHURN_SHRUTIME: 10000,
OpenMLDatasetID.BIN_FINANCIAL_BANK_MARKETING: 45211,
OpenMLDatasetID.BIN_FINANCIAL_BANK_PERSONAL_LOAN_MODELING: 5000,
OpenMLDatasetID.BIN_FINANCIAL_BANKRUPTCY_QUALITATIVE: 250,
OpenMLDatasetID.BIN_FINANCIAL_CC_FRAUD: 1000000,
OpenMLDatasetID.BIN_FINANCIAL_CC_FRAUD_TRANSACTION: 5227,
OpenMLDatasetID.BIN_FINANCIAL_CC_TAIWAN_CREDIT_DEFAULT: 30000,
OpenMLDatasetID.BIN_FINANCIAL_CREDIT_FICO_HELOC: 9871,
OpenMLDatasetID.BIN_FINANCIAL_CREDIT_GERMAN: 1000,
OpenMLDatasetID.BIN_FINANCIAL_CREDIT_GIVE_ME_SOME: 150000,
OpenMLDatasetID.BIN_FINANCIAL_CREDIT_RISK_DEFAULT: 32581,
OpenMLDatasetID.BIN_FINANCIAL_FRAUD_DETECTION: 4156,
OpenMLDatasetID.BIN_FINANCIAL_HOME_CREDIT_DEFAULT_RISK: 307511,
OpenMLDatasetID.BIN_FINANCIAL_HOME_EQUITY_CREDIT: 5960,
OpenMLDatasetID.BIN_FINANCIAL_LOAN_STATUS_PREDICTION: 614,
OpenMLDatasetID.BIN_FINANCIAL_LT_VEHICLE_LOAN: 233154,
OpenMLDatasetID.BIN_FINANCIAL_TVS_CREDIT_LOAN: 119528,
OpenMLDatasetID.BIN_GENETICS_MUSK_MOLECULES: 6598,
OpenMLDatasetID.BIN_GENETICS_OVA_BREAST: 1545,
OpenMLDatasetID.BIN_GEOGRAPHY_NOMAO_SEARCH_ENGINE: 34465,
OpenMLDatasetID.BIN_HEALTHCARE_ALZHEIMER_HANDWRITE_DARWIN: 174,
OpenMLDatasetID.BIN_HEALTHCARE_BLOOD_TRANSFUSION: 748,
OpenMLDatasetID.BIN_HEALTHCARE_BONE_MARROW_TRANSPLANT_CHILDREN: 187,
OpenMLDatasetID.BIN_HEALTHCARE_BREAST_CANCER_MAMMOGRAM: 39998,
OpenMLDatasetID.BIN_HEALTHCARE_BREAST_CANCER_WISCONSIN: 699,
OpenMLDatasetID.BIN_HEALTHCARE_BREAST_CANCER_YUGOSLAVIA: 286,
OpenMLDatasetID.BIN_HEALTHCARE_CELLS_WDBC_WISCONSIN_BREAST_CANCER: 569,
OpenMLDatasetID.BIN_HEALTHCARE_CARDIOVASCULAR_DISEASE: 70000,
OpenMLDatasetID.BIN_HEALTHCARE_DIABETES_CDC_DEMOGRAPHICS: 253680,
OpenMLDatasetID.BIN_HEALTHCARE_DIABETES_EARLY_STAGE: 520,
OpenMLDatasetID.BIN_HEALTHCARE_DIABETES_RISK_FACTORS: 768,
OpenMLDatasetID.BIN_HEALTHCARE_GLIOMA_BRAIN_TUMOR: 839,
OpenMLDatasetID.BIN_HEALTHCARE_HEART_DISEASE: 1190,
OpenMLDatasetID.BIN_HEALTHCARE_HEART_FAILURE: 5000,
OpenMLDatasetID.BIN_HEALTHCARE_INSURANCE_LEAD_PREDICTION: 23548,
OpenMLDatasetID.BIN_HEALTHCARE_KIDNEY_CHRONIC_DISEASE: 400,
OpenMLDatasetID.BIN_HEALTHCARE_LIVER_INDIAN_ILPD: 583,
OpenMLDatasetID.BIN_HEALTHCARE_LUNG_MALE_ARSENIC: 559,
OpenMLDatasetID.BIN_HEALTHCARE_MENTAL_HEALTH_TECH: 1259,
OpenMLDatasetID.BIN_HEALTHCARE_SEPSIS_PHYSIONET: 1552210,
OpenMLDatasetID.BIN_HEALTHCARE_THYROID_CANCER: 383,
OpenMLDatasetID.BIN_INDUSTRIAL_CYLINDER_BANDING_PRINTING: 540,
OpenMLDatasetID.BIN_NATURE_COLIC_HORSES_SURGICAL_LESION: 368,
OpenMLDatasetID.BIN_NATURE_DISEASED_TREES_WILT: 4839,
OpenMLDatasetID.BIN_NATURE_OZONE_LEVEL: 2534,
OpenMLDatasetID.BIN_NATURE_MUSHROOM_POISONOUS: 8124,
OpenMLDatasetID.BIN_NATURE_PHONEME_SOUNDS: 5404,
OpenMLDatasetID.BIN_PROFESSIONAL_ACADEMIC_COLLEGE_SCORECARD: 124699,
OpenMLDatasetID.BIN_PROFESSIONAL_AMAZON_EMPLOYEE_ACCESS: 32769,
OpenMLDatasetID.BIN_PROFESSIONAL_EMPLOYEE_IBM_ATTRITION: 1470,
OpenMLDatasetID.BIN_PROFESSIONAL_EMPLOYEE_TURNOVER_TECHCO: 34452,
OpenMLDatasetID.BIN_PROFESSIONAL_FAKE_JOB_POSTING: 12725,
OpenMLDatasetID.BIN_PROFESSIONAL_KICKSTARTER_FUNDING: 86502,
OpenMLDatasetID.BIN_PROFESSIONAL_LICD_LABOR_RIGHTS: 63634,
OpenMLDatasetID.BIN_SCIENCE_ANTIGEN_AVIDA_HIL6: 573891,
OpenMLDatasetID.BIN_SCIENCE_CLIMATE_MODEL_SIMULATION_CRASHES: 540,
OpenMLDatasetID.BIN_SCIENCE_EYE_MOVEMENT: 7608,
OpenMLDatasetID.BIN_SCIENCE_EYE_STATE_EEG: 14980,
OpenMLDatasetID.BIN_SCIENCE_HIV_QSAR: 4229,
OpenMLDatasetID.BIN_SCIENCE_MAGIC_TELESCOPE: 13376,
OpenMLDatasetID.BIN_SCIENCE_PARTICLE_HIGGS: 1000000,
OpenMLDatasetID.BIN_SCIENCE_PARTICLE_MINIBOONE: 130064,
OpenMLDatasetID.BIN_SCIENCE_QSAR_BIODEG: 1055,
OpenMLDatasetID.BIN_SCIENCE_SIMULATION_HILL_VALLEY: 1212,
OpenMLDatasetID.BIN_SCIENCE_WATER_TREATMENT: 527,
OpenMLDatasetID.BIN_SOCIAL_COMPASS_TWO_YEARS_OFFEND: 4966,
OpenMLDatasetID.BIN_SOCIAL_EDUCATIONAL_TRANSITIONS_IRISH: 500,
OpenMLDatasetID.BIN_SOCIAL_IMDB_GENRE_PREDICTION: 800,
OpenMLDatasetID.BIN_SOCIAL_GAMING_LEAGUE_OF_LEGENDS_DIAMOND: 48651,
OpenMLDatasetID.BIN_SOCIAL_HATE_SPEECH_DATASET_DYNAMICALLY_GENERATED: 41144,
OpenMLDatasetID.BIN_SOCIAL_JIGSAW_TOXICITY: 100000,
OpenMLDatasetID.BIN_SOCIAL_POLICE_INCIDENTS_SAN_FRANCISCO: 2215023,
OpenMLDatasetID.BIN_SOCIAL_POLITICS_US_CONGRESS_VOTES: 435,
OpenMLDatasetID.BIN_SOCIAL_SPAM_EMAILS_SPAMBASE: 4601,
OpenMLDatasetID.BIN_SOCIAL_SPEED_DATING: 8378,
OpenMLDatasetID.BIN_SOCIAL_TIC_TAC_TOE: 958,
OpenMLDatasetID.BIN_SOCIAL_TWITTER_DISASTER: 11370,
OpenMLDatasetID.BIN_SPORTS_CHESS_KR_VS_KP: 3196,
OpenMLDatasetID.BIN_SPORTS_NBA_SHOTS: 128069,
OpenMLDatasetID.BIN_SPORTS_PRO_FOOTBALL_BETS_PROFB: 672,
OpenMLDatasetID.BIN_SPORTS_RUN_OR_WALK: 88588,
OpenMLDatasetID.BIN_TRANSPORTATION_AIRLINES_DEPARTURE_DELAY: 539383,
OpenMLDatasetID.BIN_TRANSPORTATION_CAR_BAD_BUY_KICK: 72983,
OpenMLDatasetID.BIN_TRANSPORTATION_ESTONIA_DISASTER_PASSENGERS: 989,
OpenMLDatasetID.BIN_TRANSPORTATION_ROAD_SAFETY_GENDER: 111762,
OpenMLDatasetID.BIN_TRANSPORTATION_TITANIC_SURVIVAL: 1309,
OpenMLDatasetID.MUL_ANONYM_AMAZON_COMMERCE_REVIEWS: 1500,
OpenMLDatasetID.MUL_ANONYM_CNAE: 1080,
OpenMLDatasetID.MUL_ANONYM_DILBERT: 10000,
OpenMLDatasetID.MUL_ANONYM_DNA: 3186,
OpenMLDatasetID.MUL_ANONYM_FABERT: 8237,
OpenMLDatasetID.MUL_ANONYM_FIRST_ORDER_THEOREM_PROVING: 6118,
OpenMLDatasetID.MUL_ANONYM_FOURIER_IMAGE_COEFFICIENT: 2000,
OpenMLDatasetID.MUL_ANONYM_HELENA: 65196,
OpenMLDatasetID.MUL_ANONYM_ISOLET_LETTER_SPEECH_RECOGNITION: 7797,
OpenMLDatasetID.MUL_ANONYM_JANNIS: 83733,
OpenMLDatasetID.MUL_ANONYM_LED24: 3200,
OpenMLDatasetID.MUL_ANONYM_MFEAT_FACTORS: 2000,
OpenMLDatasetID.MUL_ANONYM_MICRO_MASS: 571,
OpenMLDatasetID.MUL_ANONYM_PENDIGITS: 10992,
OpenMLDatasetID.MUL_ANONYM_POL: 15000,
OpenMLDatasetID.MUL_ANONYM_PLANTS_TEXTURE: 1599,
OpenMLDatasetID.MUL_ANONYM_ROBERT: 10000,
OpenMLDatasetID.MUL_ANONYM_SPOKEN_ARABIC_DIGIT: 263256,
OpenMLDatasetID.MUL_ANONYM_SYNTHETIC_CONTROL: 600,
OpenMLDatasetID.MUL_ANONYM_VOLCANOES_VENUS: 1183,
OpenMLDatasetID.MUL_ANONYM_VOLKERT: 58310,
OpenMLDatasetID.MUL_ANONYM_ZERNIKE_MFEAT: 2000,
OpenMLDatasetID.MUL_COMPUTERS_IMAGE_ARTIFICIAL_CHARACTERS: 10218,
OpenMLDatasetID.MUL_COMPUTERS_IMAGE_CIFAR10: 60000,
OpenMLDatasetID.MUL_COMPUTERS_IMAGE_GESTURE_PHASE_SEGMENTATION: 9873,
OpenMLDatasetID.MUL_COMPUTERS_IMAGE_GTSRB_GERMAN_TRAFFIC_SIGN: 51839,
OpenMLDatasetID.MUL_COMPUTERS_IMAGE_INDIAN_PINES: 9144,
OpenMLDatasetID.MUL_COMPUTERS_IMAGE_LETTER_RECOGNITION: 20000,
OpenMLDatasetID.MUL_COMPUTERS_IMAGE_MNIST_DIGITS: 70000,
OpenMLDatasetID.MUL_COMPUTERS_IMAGE_MNIST_FASHION: 70000,
OpenMLDatasetID.MUL_COMPUTERS_IMAGE_MNIST_JAPANESE_KUZUSHIJI_49: 270912,
OpenMLDatasetID.MUL_COMPUTERS_IMAGE_OPT_DIGITS: 5620,
OpenMLDatasetID.MUL_COMPUTERS_IMAGE_SEGMENTATION: 2310,
OpenMLDatasetID.MUL_COMPUTERS_INTRUSION_DETECTION_KDD: 4898431,
OpenMLDatasetID.MUL_COMPUTERS_META_STREAM_INTERVALS: 45164,
OpenMLDatasetID.MUL_COMPUTERS_PAGE_BLOCK_PARSING: 5473,
OpenMLDatasetID.MUL_COMPUTERS_ROBOT_WALL_NAVIGATION: 5456,
OpenMLDatasetID.MUL_CONSUMER_ELECTRICITY_TAMILNADU: 45781,
OpenMLDatasetID.MUL_CONSUMER_INTERNET_USAGE_PROFESSION: 10108,
OpenMLDatasetID.MUL_CONSUMER_PRODUCT_SENTIMENT: 5091,
OpenMLDatasetID.MUL_CONSUMER_WOMEN_ECOMMERCE_CLOTHING_REVIEW: 18788,
OpenMLDatasetID.MUL_FINANCIAL_CORPORATE_CREDIT_RATING_CAT: 5000,
OpenMLDatasetID.MUL_FINANCIAL_CORPORATE_CREDIT_RATING_SYMBOL: 2029,
OpenMLDatasetID.MUL_FINANCIAL_CREDIT_RISK_CHINA: 27522,
OpenMLDatasetID.MUL_FINANCIAL_CREDIT_SCORE_HZL: 100000,
OpenMLDatasetID.MUL_FINANCIAL_LOAN_IBRD: 9217,
OpenMLDatasetID.MUL_FINANCIAL_P2P_LOAN: 2875146,
OpenMLDatasetID.MUL_FOOD_WINE_QUALITY_CAT: 4898,
OpenMLDatasetID.MUL_FOOD_WINE_REVIEW: 84123,
OpenMLDatasetID.MUL_GENETICS_SPLICE_DNA: 3190,
OpenMLDatasetID.MUL_GENETICS_ACP_BREAST_CANCER: 949,
OpenMLDatasetID.MUL_HEALTHCARE_APPENDICITIS_REGENSBURG_PEDIATRIC: 782,
OpenMLDatasetID.MUL_HEALTHCARE_AUDIOLOGY_DIAGNOSTIC: 226,
OpenMLDatasetID.MUL_HEALTHCARE_BABY_SLEEP_STATE: 1024,
OpenMLDatasetID.MUL_HEALTHCARE_CANCER_CERVICAL: 858,
OpenMLDatasetID.MUL_HEALTHCARE_CMC_CONTRACEPTIVE_WOMEN: 1473,
OpenMLDatasetID.MUL_HEALTHCARE_DIABETES_US130: 101766,
OpenMLDatasetID.MUL_HEALTHCARE_EGYPTIAN_SKULLS: 150,
OpenMLDatasetID.MUL_HEALTHCARE_HEART_ARRHYTMIA: 452,
OpenMLDatasetID.MUL_HEALTHCARE_HEART_CARDIOTOCOGRAPHY: 2126,
OpenMLDatasetID.MUL_HEALTHCARE_HEPATITIS_C_EGYPT: 1385,
OpenMLDatasetID.MUL_HEALTHCARE_HYPOTHYROID: 3772,
OpenMLDatasetID.MUL_HEALTHCARE_LYMPH_YUGOSLAVIA: 148,
OpenMLDatasetID.MUL_HEALTHCARE_MENTAL_HEALTH_OCCUPATION: 292364,
OpenMLDatasetID.MUL_HEALTHCARE_OBESITY_LEVELS: 2111,
OpenMLDatasetID.MUL_HEALTHCARE_PBCSEQ_BILIARY_CIRRHOSIS: 1945,
OpenMLDatasetID.MUL_HEALTHCARE_TEETH_DMFT_ANALCATDATA: 797,
OpenMLDatasetID.MUL_HEALTHCARE_WHITE_BLOOD_CELLS_WBC: 10298,
OpenMLDatasetID.MUL_HOUSES_MELBOURNE_AIRBNB: 18316,
OpenMLDatasetID.MUL_INDUSTRIAL_STEEL_PLATES_FAULT: 1941,
OpenMLDatasetID.MUL_NATURE_AIR_QUALITY_POLLUTION: 5000,
OpenMLDatasetID.MUL_NATURE_CJS_MAPLE_TREES: 2796,
OpenMLDatasetID.MUL_NATURE_EUCALYPTUS_SEED: 736,
OpenMLDatasetID.MUL_NATURE_FOREST_COVERTYPE: 581012,
OpenMLDatasetID.MUL_NATURE_SOLAR_FLARES: 1066,
OpenMLDatasetID.MUL_NATURE_SOYBEAN_DISEASE: 683,
OpenMLDatasetID.MUL_NATURE_YEAST_PROTEIN: 1484,
OpenMLDatasetID.MUL_PROFESSIONAL_DATA_SCIENTIST_SALARY: 15841,
OpenMLDatasetID.MUL_PROFESSIONAL_NURSERY_APPLICATIONS_SLOVENIA: 12960,
OpenMLDatasetID.MUL_PROFESSIONAL_STUDENT_PERFORMANCE_ENTRANCE_EXAM: 666,
OpenMLDatasetID.MUL_PROFESSIONAL_TEACHER_ASSISTANT_EVALUATION_TAE: 151,
OpenMLDatasetID.MUL_SCIENCE_ANNEAL_CHEMICAL: 898,
OpenMLDatasetID.MUL_SCIENCE_BALANCE_SCALE: 625,
OpenMLDatasetID.MUL_SCIENCE_BATCH_CHORAL_HARMONY: 5665,
OpenMLDatasetID.MUL_SCIENCE_DRUG_DIRECTORY_FDA_UNAPPROVED: 120215,
OpenMLDatasetID.MUL_SCIENCE_MICE_PROTEIN: 1080,
OpenMLDatasetID.MUL_SCIENCE_MIMIC_ROUTE: 4156450,
OpenMLDatasetID.MUL_SCIENCE_QSAR_BIOCONCENTRATION: 779,
OpenMLDatasetID.MUL_SCIENCE_VOWEL_JAPANESE: 9961,
OpenMLDatasetID.MUL_SCIENCE_VOWEL_SPEAKER_RECOGNITION: 990,
OpenMLDatasetID.MUL_SCIENCE_WAVEFORM_500: 5000,
OpenMLDatasetID.MUL_SOCIAL_BIAS_FRAMES: 147139,
OpenMLDatasetID.MUL_SOCIAL_BOOKS_AUTHORSHIP_ANALCATDATA: 841,
OpenMLDatasetID.MUL_SOCIAL_BROWN_FROWN_CORPORA_COLLINS: 1000,
OpenMLDatasetID.MUL_SOCIAL_DPBEDIA: 342781,
OpenMLDatasetID.MUL_SOCIAL_FUNPEDIA: 29819,
OpenMLDatasetID.MUL_SOCIAL_GOOGLE_QA_TYPE_REASON: 4863,
OpenMLDatasetID.MUL_SOCIAL_HARRY_POTTER_FAN_FICTION: 648493,
OpenMLDatasetID.MUL_SOCIAL_HOLISTIC_BIAS: 472991,
OpenMLDatasetID.MUL_SOCIAL_NEWS_CHANNEL_CATEGORY: 20284,
OpenMLDatasetID.MUL_SOCIAL_OKCUPID_DATING_JOB_STEM: 50789,
OpenMLDatasetID.MUL_SOCIAL_SOCC_OPINION_COMMENTS_CORPUS_TOXICITY: 1043,
OpenMLDatasetID.MUL_SOCIAL_STACKOVERFLOW_POLARITY: 4423,
OpenMLDatasetID.MUL_SOCIAL_SUPREME_COURT_ANALCATDATA: 4052,
OpenMLDatasetID.MUL_SOCIAL_WIKIPEDIA_TALK_LABELS_ATTACKS: 855514,
OpenMLDatasetID.MUL_SPORTS_ACTIVITY_LOCALIZED_LDPA: 164860,
OpenMLDatasetID.MUL_SPORTS_BASEBALL_HALL_OF_FAME: 1340,
OpenMLDatasetID.MUL_SPORTS_CHESS_JUNGLE: 44819,
OpenMLDatasetID.MUL_SPORTS_CONNECT4_GAME: 67557,
OpenMLDatasetID.MUL_SPORTS_POKER_HAND: 1025009,
OpenMLDatasetID.MUL_SPORTS_WALKING_ACTIVITY: 149332,
OpenMLDatasetID.MUL_TRANSPORTATION_AIRLINES_TWITTER_SENTIMENT: 1097,
OpenMLDatasetID.MUL_TRANSPORTATION_CAR_ACCEPTABILITY: 1728,
OpenMLDatasetID.MUL_TRANSPORTATION_CAR_ORIGIN_COUNTRY: 406,
OpenMLDatasetID.MUL_TRANSPORTATION_CAR_STUDENT_INSURANCE_RISK: 20000,
OpenMLDatasetID.MUL_TRANSPORTATION_SHUTTLE_SPACE: 58000,
OpenMLDatasetID.MUL_TRANSPORTATION_TRAFFIC_ACCIDENTS_FARS: 100968,
OpenMLDatasetID.MUL_TRANSPORTATION_TRAFFIC_VIOLATION: 70340,
OpenMLDatasetID.MUL_TRANSPORTATION_VEHICLE_SILHOUETTE: 846,
OpenMLDatasetID.REG_ANONYM_ALLSTATE_CLAIM_SEVERITY: 188318,
OpenMLDatasetID.REG_ANONYM_BANK_32NH: 8192,
OpenMLDatasetID.REG_ANONYM_BUZZ_IN_SOCIAL_MEDIA_TWITTER: 583250,
OpenMLDatasetID.REG_ANONYM_DIONIS: 416188,
OpenMLDatasetID.REG_ANONYM_FAT_MEAT_TECATOR: 240,
OpenMLDatasetID.REG_ANONYM_GEOGRAPHICAL_ORIGIN_OF_MUSIC: 1059,
OpenMLDatasetID.REG_ANONYM_HOUSE_16H: 22784,
OpenMLDatasetID.REG_ANONYM_MERCEDES_BENZ_GREENER_MANUFACTURING: 4209,
OpenMLDatasetID.REG_ANONYM_SANTANDER_TRANSACTION_VALUE: 4459,
OpenMLDatasetID.REG_ANONYM_TOPO: 8885,
OpenMLDatasetID.REG_ANONYM_YOLANDA: 400000,
OpenMLDatasetID.REG_ANONYM_YPROP: 8885,
OpenMLDatasetID.REG_COMPUTERS_ALGO_RUNTIME_MIP2016: 1090,
OpenMLDatasetID.REG_COMPUTERS_ALGO_RUNTIME_SAT11: 4440,
OpenMLDatasetID.REG_COMPUTERS_CPU_ACTIVITY: 8192,
OpenMLDatasetID.REG_COMPUTERS_GAMING_FRAMES_FPS_BENCHMARK: 24624,
OpenMLDatasetID.REG_COMPUTERS_META_LEVEL_LEARNING: 528,
OpenMLDatasetID.REG_COMPUTERS_PUMA_ROBOT_ARM: 8192,
OpenMLDatasetID.REG_COMPUTERS_ROBOT_KIN8NM: 8192,
OpenMLDatasetID.REG_COMPUTERS_ROBOT_SARCOS: 48933,
OpenMLDatasetID.REG_COMPUTERS_YOUTUBE_VIDEO_TRANSCODING: 68784,
OpenMLDatasetID.REG_CONSUMER_AMERICAN_EAGLE_PRICES: 22662,
OpenMLDatasetID.REG_CONSUMER_AVOCADO_SALES: 18249,
OpenMLDatasetID.REG_CONSUMER_BLACK_FRIDAY: 166821,
OpenMLDatasetID.REG_CONSUMER_BOOK_PRICE_PREDICTION: 4989,
OpenMLDatasetID.REG_CONSUMER_BOOKING_HOTEL_REVIEW: 515738,
OpenMLDatasetID.REG_CONSUMER_DIAMONDS_PRICES: 53940,
OpenMLDatasetID.REG_CONSUMER_JC_PENNEY_PRODUCT_PRICE: 10860,
OpenMLDatasetID.REG_CONSUMER_MEDICAL_CHARGES: 163065,
OpenMLDatasetID.REG_CONSUMER_MERCARI_ONLINE_MARKETPLACE: 100000,
OpenMLDatasetID.REG_CONSUMER_ONLINE_NEWS_POPULARITY: 24007,
OpenMLDatasetID.BIN_CONSUMER_REASONER_RECOMMENDATION: 58497,
OpenMLDatasetID.REG_FINANCIAL_CC_USER_SEGMENTATION: 8950,
OpenMLDatasetID.REG_FINANCIAL_INSURANCE_HEALTH_HOURS_WORKED_WIFE: 22272,
OpenMLDatasetID.REG_FINANCIAL_INSURANCE_PREMIUM_DATA: 1338,
OpenMLDatasetID.REG_FINANCIAL_STOCK_AEROSPACE: 950,
OpenMLDatasetID.REG_FOOD_WINE_JUDGEMENT_SENSORY: 576,
OpenMLDatasetID.REG_HEALTHCARE_BODY_FAT: 252,
OpenMLDatasetID.REG_HEALTHCARE_DIABETES_SKLEARN: 442,
OpenMLDatasetID.REG_HEALTHCARE_POLLUTION_LA_MORTALITY_RMFTSA: 508,
OpenMLDatasetID.REG_HOUSES_BOSTON_HOUSE: 506,
OpenMLDatasetID.REG_HOUSES_BOSTON_PRICE_NOMINAL: 1460,
OpenMLDatasetID.REG_HOUSES_BRAZILIAN_HOUSES: 10692,
OpenMLDatasetID.REG_HOUSES_CALIFORNIA_HOUSES: 20640,
OpenMLDatasetID.REG_HOUSES_CALIFORNIA_PRICES_2020: 37951,
OpenMLDatasetID.REG_HOUSES_GERMAN_HOUSES: 10552,
OpenMLDatasetID.REG_HOUSES_HOUSEHOLD_MONTHLY_ELECTRICITY: 1000,
OpenMLDatasetID.REG_HOUSES_IOWA_HOUSES_PRICES: 1460,
OpenMLDatasetID.REG_HOUSES_LISBON: 246,
OpenMLDatasetID.REG_HOUSES_MIAMI: 13932,
OpenMLDatasetID.REG_HOUSES_PERTH: 33656,
OpenMLDatasetID.REG_HOUSES_SEATTLE_KINGS_COUNTY: 21613,
OpenMLDatasetID.REG_NATURE_ABALONE_FISH_RINGS: 4177,
OpenMLDatasetID.REG_NATURE_CLIMATE_CHANGE_AGRICULTURE_FINANCIAL: 10000,
OpenMLDatasetID.REG_NATURE_EL_NINO_KDD: 782,
OpenMLDatasetID.REG_NATURE_FISH_TOXICITY: 908,
OpenMLDatasetID.REG_NATURE_FOREST_FIRES: 517,
OpenMLDatasetID.REG_NATURE_MYANMAR_AIR_QUALITY: 5122,
OpenMLDatasetID.REG_NATURE_NO2_POLLUTION_NORWAY: 500,
OpenMLDatasetID.REG_NATURE_POLLEN_GRAINS_SYNTH: 3848,
OpenMLDatasetID.REG_NATURE_POLLEN_LUXEMBOURG: 7784,
OpenMLDatasetID.REG_NATURE_QUAKE_RICHTER: 2178,
OpenMLDatasetID.REG_NATURE_WIND_SPEED_IRELAND: 6574,
OpenMLDatasetID.REG_PROFESSIONAL_CPS88_WAGES: 28155,
OpenMLDatasetID.REG_PROFESSIONAL_EMPLOYEE_SALARY_MONTGOMERY: 9228,
OpenMLDatasetID.REG_PROFESSIONAL_STUDENT_PERFORMANCE_PORTUGAL: 649,
OpenMLDatasetID.REG_SCIENCE_AIRFOIL_SELF_NOISE: 1503,
OpenMLDatasetID.REG_SCIENCE_AUCTION_VERIFICATION: 2043,
OpenMLDatasetID.REG_SCIENCE_CONCRETE_COMPRESSIVE_STRENGTH: 1030,
OpenMLDatasetID.REG_SCIENCE_ENERGY_EFFICIENCY: 768,
OpenMLDatasetID.REG_SCIENCE_GRID_STABILITY: 10000,
OpenMLDatasetID.REG_SCIENCE_PHYSIOCHEMICAL_PROTEIN: 45730,
OpenMLDatasetID.REG_SCIENCE_QSAR_TID_11: 5742,
OpenMLDatasetID.REG_SCIENCE_QSAR_TID_10980: 5766,
OpenMLDatasetID.REG_SCIENCE_SIMULATION_2D_PLANES: 40768,
OpenMLDatasetID.REG_SCIENCE_SIMULATION_FRIED_FORMULA_PREDICT: 40768,
OpenMLDatasetID.REG_SCIENCE_SIMULATION_MV: 40768,
OpenMLDatasetID.REG_SCIENCE_SULFUR: 10081,
OpenMLDatasetID.REG_SCIENCE_SUPERCONDUCTIVITY: 21263,
OpenMLDatasetID.REG_SCIENCE_WAVE_ENERGY: 72000,
OpenMLDatasetID.REG_SOCIAL_COLLEGES_GRANTS: 7063,
OpenMLDatasetID.REG_SOCIAL_OCCUPATION_MOBILITY_SOCMOB: 1156,
OpenMLDatasetID.REG_SOCIAL_OKCUPID_DATING_PROFILE_AGE: 59946,
OpenMLDatasetID.REG_SOCIAL_STRIKES_PER_COUNTRY: 625,
OpenMLDatasetID.REG_SOCIAL_US_CRIME: 1994,
OpenMLDatasetID.REG_SOCIAL_VOTING_SPACE_GEOGRAPHIC_ANALYSIS: 3107,
OpenMLDatasetID.REG_SPORTS_BASEBALL_HITTER_SALARY: 322,
OpenMLDatasetID.REG_SPORTS_FIFA20_PLAYERS_VALUE: 14999,
OpenMLDatasetID.REG_SPORTS_FIFA22_WAGES: 19178,
OpenMLDatasetID.REG_SPORTS_FIFA_PLAYERS_STATS: 183978,
OpenMLDatasetID.REG_SPORTS_MONEYBALL: 1232,
OpenMLDatasetID.REG_SPORTS_NBA_2K20_PLAYERS_RATING: 439,
OpenMLDatasetID.REG_SPORTS_NBA_ALL_STAR: 1408,
OpenMLDatasetID.REG_TRANSPORTATION_CAR_GM_PRICE: 804,
OpenMLDatasetID.REG_TRANSPORTATION_FLIGHT_SIMULATION_ELEVATORS: 16599,
OpenMLDatasetID.REG_TRANSPORTATION_NAVAL_PROPULSION_PLANT: 11934,
OpenMLDatasetID.REG_TRANSPORTATION_NYC_BIKE_TRIP_DURATION: 4500000,
OpenMLDatasetID.REG_TRANSPORTATION_NYC_TAXI_TIP: 581835,
OpenMLDatasetID.REG_TRANSPORTATION_NYC_TAXI_TRIP_DURATION: 2083778,
OpenMLDatasetID.REG_TRANSPORTATION_SEOUL_BIKE_SHARING_DEMAND: 8760,
OpenMLDatasetID.REG_TRANSPORTATION_US_AIRPORT_PASSENGERS: 3606803,
OpenMLDatasetID.REG_TRANSPORTATION_US_BIKE_SHARING_DEMAND: 17379,
OpenMLDatasetID.REG_TRANSPORTATION_ZURICH_PUBLIC_TRANSPORT_DELAY: 5465575,
KaggleDatasetID.BIN_SOCIAL_HUMAN_CHOICE_PREDICTION_LM_GAMES: 71579,
KaggleDatasetID.MUL_FOOD_MICHELIN_GUIDE_RESTAURANTS: 17735,
KaggleDatasetID.REG_FOOD_RAMEN_RATINGS_2022: 4120,
KaggleDatasetID.MUL_FOOD_YELP_REVIEWS: 10000,
KaggleDatasetID.MUL_SOCIAL_GOT_SCRIPTS: 23911,
KaggleDatasetID.MUL_SOCIAL_US_ELECTIONS_SPEECHES: 269,
KaggleDatasetID.MUL_TRANSPORTATION_US_ACCIDENTS_MARCH23: 7728394,
KaggleDatasetID.REG_CONSUMER_CAR_PRICE_CARDEKHO: 37814,
KaggleDatasetID.REG_FOOD_ALCOHOL_WIKILIQ_PRICES: 12869,
KaggleDatasetID.REG_FOOD_BEER_RATINGS: 3197,
KaggleDatasetID.REG_FOOD_CHOCOLATE_BAR_RATINGS: 1795,
KaggleDatasetID.REG_FOOD_COFFEE_REVIEW: 2440,
KaggleDatasetID.REG_FOOD_WHISKY_SCOTCH_REVIEWS: 2247,
KaggleDatasetID.REG_FOOD_WINE_POLISH_MARKET_PRICES: 2247,
KaggleDatasetID.REG_FOOD_WINE_VIVINO_SPAIN: 8650,
KaggleDatasetID.REG_FOOD_ZOMATO_RESTAURANTS: 51717,
KaggleDatasetID.REG_PROFESSIONAL_COMPANY_EMPLOYEES_SIZE: 7173426,
KaggleDatasetID.REG_SOCIAL_ANIME_PLANET_RATING: 16621,
KaggleDatasetID.REG_SOCIAL_BOOK_READABILITY_CLEAR: 4726,
KaggleDatasetID.REG_SOCIAL_FILMTV_MOVIE_RATING_ITALY: 41399,
KaggleDatasetID.REG_SOCIAL_KOREAN_DRAMA: 1647,
KaggleDatasetID.REG_SOCIAL_MOVIES_DATASET_REVENUE: 45466,
KaggleDatasetID.REG_SOCIAL_MUSEUMS_US_REVENUES: 33072,
KaggleDatasetID.REG_SOCIAL_SPOTIFY_POPULARITY: 114000,
KaggleDatasetID.REG_SOCIAL_VIDEO_GAMES_SALES: 16598,
KaggleDatasetID.REG_SPORTS_FOOTBALL_MANAGER_IMPORTANT_MATCHES: 159541,
KaggleDatasetID.REG_SPORTS_NBA_DRAFT_VALUE_OVER_REPLACEMENT: 1922,
KaggleDatasetID.REG_TRANSPORTATION_USED_CAR_MERCEDES_BENZ_ITALY: 16392,
KaggleDatasetID.REG_TRANSPORTATION_USED_CAR_PAKISTAN: 72655,
KaggleDatasetID.REG_TRANSPORTATION_USED_CAR_SAUDI_ARABIA: 8035,
UrlDatasetID.REG_CONSUMER_BABIES_R_US_PRICES: 5085,
UrlDatasetID.REG_CONSUMER_BIKE_PRICE_BIKEWALE: 9003,
UrlDatasetID.REG_PROFESSIONAL_EMPLOYEE_RENUMERATION_VANCOUBER: 44574,
UrlDatasetID.REG_PROFESSIONAL_ML_DS_AI_JOBS_SALARIES: 105434,
UrlDatasetID.REG_PROFESSIONAL_SCIMAGOJR_ACADEMIC_IMPACT: 31136,
UrlDatasetID.REG_SOCIAL_BOOKS_GOODREADS: 3967,
UrlDatasetID.REG_SOCIAL_MOVIES_ROTTEN_TOMATOES: 7390,
}