from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Moneyball
====
Examples: 1232
====
URL: https://www.openml.org/search?type=data&id=41021
====
Description: **Author**: MITx  
**Source**: [Kaggle](https://www.kaggle.com/wduckett/moneyball-mlb-stats-19622012/data), originally from [The Analytics Edge course on EdX](https://www.edx.org/course/analytics-edge-mitx-15-071x-3). Data collected from [baseball-reference.com](baseball-reference.com)  
**Please cite**:   

**Moneyball**  
In the early 2000s, Billy Beane and Paul DePodesta worked for the Oakland Athletics. While there, they literally changed the game of baseball. They didn't do it using a bat or glove, and they certainly didn't do it by throwing money at the issue; in fact, money was the issue. They didn't have enough of it, but they were still expected to keep up with teams that had much deeper pockets. This is where Statistics came riding down the hillside on a white horse to save the day. This data set contains some of the information that was available to Beane and DePodesta in the early 2000s, and it can be used to better understand their methods.

### Attributes  
This data set contains a set of variables that Beane and DePodesta focused heavily on. They determined that stats like on-base percentage (OBP) and slugging percentage (SLG) were very important when it came to scoring runs, however they were largely undervalued by most scouts at the time. This translated to a gold mine for Beane and DePodesta. Since these players weren't being looked at by other teams, they could recruit these players on a small budget. The variables are as follows:

Team  
League  
Year  
Runs Scored (RS)  
Runs Allowed (RA)  
Wins (W)  
On-Base Percentage (OBP)  
Slugging Percentage (SLG)  
Batting Average (BA)  
Playoffs (binary)  
RankSeason  
RankPlayoffs  
Games Played (G)  
Opponent On-Base Percentage (OOBP)  
Opponent Slugging Percentage (OSLG)  

### Acknowledgements  
This data set is referenced in The Analytics Edge course on EdX during the lecture regarding the story of Moneyball. The data itself is gathered from baseball-reference.com. Sports-reference.com is one of the most comprehensive sports statistics resource available, and I highly recommend checking it out.

Inspiration
It is such an important skill in today's world to be able to see the "truth" in a data set. That is what DePodesta was able to do with this data, and it unsettled the entire system of baseball recruitment. Beane and DePodesta defined their season goal as making it to playoffs. With that in mind, consider these questions:

How does a team make the playoffs?
How does a team win more games?
How does a team score more runs?
They are all simple questions with simple answers, but now it is time to use the data to find the "truth" hidden in the numbers.
====
Target Variable: RS (numeric, 374 distinct): ['682', '691', '707', '735', '758', '708', '673', '714', '654', '731']
====
Features:

Team (nominal, 39 distinct): ['HOU', 'DET', 'NYY', 'NYM', 'MIN', 'LAD', 'SFG', 'PIT', 'PHI', 'STL']
League (nominal, 2 distinct): ['AL', 'NL']
Year (numeric, 47 distinct): ['2012', '2004', '2011', '1998', '1999', '2001', '2002', '2003', '2000', '2005']
RA (numeric, 381 distinct): ['717', '657', '744', '731', '648', '680', '611', '643', '698', '745']
W (numeric, 63 distinct): ['83', '86', '76', '79', '88', '90', '89', '75', '85', '80']
OBP (numeric, 87 distinct): ['0.322', '0.32', '0.325', '0.321', '0.333', '0.324', '0.33', '0.331', '0.323', '0.339']
SLG (numeric, 162 distinct): ['0.401', '0.395', '0.381', '0.409', '0.391', '0.388', '0.403', '0.396', '0.387', '0.385']
BA (numeric, 75 distinct): ['0.263', '0.261', '0.258', '0.264', '0.256', '0.26', '0.259', '0.267', '0.262', '0.27']
Playoffs (nominal, 2 distinct): ['0', '1']
RankSeason (nominal, 9 distinct): ['2', '1', '3', '4', '5', '6', '7', '8']
RankPlayoffs (nominal, 6 distinct): ['3', '4', '1', '2', '5']
G (nominal, 8 distinct): ['162', '161', '163', '160', '159', '164', '165', '158']
OOBP (numeric, 73 distinct): ['0.314', '0.329', '0.327', '0.336', '0.342', '0.33', '0.319', '0.348', '0.328', '0.338']
OSLG (numeric, 113 distinct): ['0.431', '0.423', '0.422', '0.398', '0.408', '0.407', '0.415', '0.405', '0.404', '0.401']
'''

CONTEXT = "Moneyball Dataset for Baseball Team Performance"
TARGET = CuratedTarget(raw_name="RS", new_name="Baseball Team Runs Scored", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [
    CuratedFeature(raw_name='League', new_name='Baseball League',
                   value_mapping={'AL': 'American League', 'NL': 'National League'}),
    CuratedFeature(raw_name='Year', new_name='Season Year'),
    CuratedFeature(raw_name='RA', new_name='Runs Allowed'),
    CuratedFeature(raw_name='W', new_name='Wins'),
    CuratedFeature(raw_name='OBP', new_name='On-Base Percentage'),
    CuratedFeature(raw_name='SLG', new_name='Slugging Percentage'),
    CuratedFeature(raw_name='BA', new_name='Batting Average'),
    CuratedFeature(raw_name='Playoffs', value_mapping={'0': 'Not Qualified', '1': 'Qualified'}),
    CuratedFeature(raw_name='RankSeason', new_name='Season Rank'),
    CuratedFeature(raw_name='RankPlayoffs', new_name='Playoff Rank'),
    CuratedFeature(raw_name='G', new_name='Games Played in Season'),
    CuratedFeature(raw_name='OOBP', new_name='Opponent On-Base Percentage'),
    CuratedFeature(raw_name='OSLG', new_name='Opponent Slugging Percentage'),
]
