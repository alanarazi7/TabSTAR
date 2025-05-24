from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: SOCC
====
Examples: 1043
====
URL: https://www.openml.org/search?type=data&id=46709
====
Description: SOCC
SFU Opinion and Comments Corpus

The SFU Opinion and Comments Corpus (SOCC) is a corpus for the analysis of online news comments. Our corpus contains comments and the articles from which the comments originated. The articles are all opinion articles, not hard news articles. The corpus is larger than any other currently available comments corpora, and has been collected with attention to preserving reply structures and other metadata. In addition to the raw corpus, we also present annotations for four different phenomena: constructiveness, toxicity, negation and its scope, and appraisal.

For more information about this work, please see our papers.

Kolhatkar, V., N. Thain, J. Sorensen, L. Dixon and M. Taboada (2023) Classifying constructive comments. First Monday 28(4). https://doi.org/10.5210/fm.v28i4.13163

Kolhatkar, V.,H. Wu, L. Cavasso, E. Francis, K. Shukla and M. Taboada (2020) The SFU Opinion and Comments Corpus: A corpus for the analysis of online news comments. Corpus Pragmatics 4(2), 155-190.

Kolhatkar. V. and M. Taboada (2017) Using New York Times Picks to identify constructive comments. Proceedings of the Workshop Natural Language Processing Meets Journalism, Conference on Empirical Methods in Natural Language Processing. Copenhagen. September 2017.

Kolhatkar, V. and M. Taboada (2017) Constructive language in news comments. Proceedings of the 1st Abusive Language Online Workshop, 55th Annual Meeting of the Association for Computational Linguistics. Vancouver. August 2017, pp. 11-17.

SFU constructiveness and toxicity corpus
Authours annotated a subset of SOCC for constructiveness and toxicity. The annotated corpus is organized as a CSV and contains 1,043 annotated comments in responses to 10 different articles covering a variety of subjects: technology, immigration, terrorism, politics, budget, social issues, religion, property, and refugees. For half of the articles, they included only top-level comments. For the other half, we included both top-level comments and responses. They used CrowdFlower (then Figure Eight, now Appen) as our crowdsourcing annotation platform and annotated the comments for constructiveness. They asked the annotators to first read the articles, and then to tell us whether the displayed comment was constructive or not.

For toxicity, we asked annotators a multiple-choice question, How toxic is the comment? Four answers were possible:

Very toxic (4)
Toxic (3)
Mildly toxic (2)
Not toxic (1)
More information on the annotation, and the instructions to annotators, is available in the CrowdFlower_instructions file.

Our target column is "toxicity_level":

Crowd's annotation on the toxicity level of the comment. Each comment was annotated by at least three annotators and so we are providing the first two popular answers and their associated confidence scores for toxicity level. If you want ground truth go with the first one, as it is the most popular answer.

paper_url = "https://arxiv.org/pdf/2004.05476"

original_data_url = "https://www.kaggle.com/datasets/mtaboada/sfu-opinion-and-comments-corpus-socc"
====
Target Variable: toxicity_level (string, 14 distinct): ['1', '1\n2', '2\n1', '1\n3', '1\n4', '2\n3', '2\n4', '3\n1', '3\n2', '2']
====
Features:

article_id (numeric, 13 distinct): ['26373964', '32803341', '32746841', '29425530', '25343745', '32013536', '29265806', '23462276', '20144737', '31284492']
comment_counter (string, 1043 distinct): ['source1_23462276_4', 'source1_26373964_22', 'source1_26373964_8', 'source1_26373964_16', 'source1_26373964_106', 'source1_26373964_109', 'source1_26373964_37', 'source1_26373964_69', 'source1_26373964_147', 'source1_26373964_96']
title (string, 18 distinct): ['Why Belgium is ground zero for jihadi terrorism', "Don't be fooled by the (surprise!) budget surplus - The Globe and Mail", 'Yes to Uber. No to taxi cartels', "What Donald Trump's victory means for Canada and the world - The Globe and Mail", 'Thank you, Hillary. Now women know retreat is not an option - The Globe and Mail', "Is the B.C. property levy on 'foreign buyers' a new head tax? - The Globe and Mail", "Europe may be failing Syrian refugees, but Canada shouldn't boast yet - The Globe and Mail", 'Enough is enough: Time to address epidemic of violence against native women', "Apple Watch: It's the precise opposite of a labour-saving device", "The driving force behind Beijing's moves in the South China Sea"]
globe_url (string, 13 distinct): ['http://www.theglobeandmail.com/opinion/dont-be-fooled-by-the-surprise-budget-surplus/article26373964/', 'http://www.theglobeandmail.com/opinion/thank-you-hillary-women-now-know-retreat-is-not-an-option/article32803341/', 'http://www.theglobeandmail.com/opinion/editorials/what-donald-trumps-victory-means-for-canada-and-the-world/article32746841/', 'http://www.theglobeandmail.com/opinion/why-belgium-is-ground-zero-for-jihadi-terrorism/article29425530/', 'http://www.theglobeandmail.com/opinion/editorials/yes-to-uber-no-to-taxi-cartels/article25343745/', 'http://www.theglobeandmail.com/opinion/is-the-bc-property-levy-on-foreign-buyers-a-new-head-tax/article32013536/', 'http://www.theglobeandmail.com/opinion/europe-may-be-failing-syrian-refugees-but-canada-shouldnt-boast-yet/article29265806/', 'http://www.theglobeandmail.com/opinion/as-technology-marches-inexorably-oninvention-becomes-the-mother-of-necessity/article23462276/', 'http://www.theglobeandmail.com/opinion/put-native-women-on-the-agenda/article20144737/', 'http://www.theglobeandmail.com/opinion/the-driving-force-behind-beijings-moves-in-the-south-china-sea/article31284492/']
url (string, 13 distinct): ['http://www.sfu.ca/content/dam/sfu/discourse-lab/globe/budget.html', 'http://www.sfu.ca/content/dam/sfu/discourse-lab/globe/hillary.html', 'http://www.sfu.ca/content/dam/sfu/discourse-lab/globe/trump.html', 'http://www.sfu.ca/content/dam/sfu/discourse-lab/globe/belgium.html', 'http://www.sfu.ca/content/dam/sfu/discourse-lab/globe/uber.html', 'http://www.sfu.ca/content/dam/sfu/discourse-lab/globe/property.html', 'http://www.sfu.ca/content/dam/sfu/discourse-lab/globe/refugees.html', 'http://www.sfu.ca/content/dam/sfu/discourse-lab/globe/watch.html', 'http://www.sfu.ca/content/dam/sfu/discourse-lab/globe/aboriginal.html', 'http://www.sfu.ca/content/dam/sfu/discourse-lab/globe/china.html']
comment_text (string, 1043 distinct): ["While technology does march on, sometimes it takes a few backwards steps towards true progress. Case in point are the ipads and tablets which I see as little more than toys for the easily distracted. While tablets are today's rage that have led to the nosediving of the laptop, anyone who has used these new devices knows all to well their limitations. Trying to compose an email on a ipad is like communicating in morse code or sending a telegram circa 1913. The same goes for the smartwatch which does not even have GPS, which every cheap digital camera has, big mistake. But what do I know, I'm a geezer and still tell the time by a watch with a face.", "There's an approach that gets too little publicity. The Cons pass a budget. But much of that consists of individual projects that need Ministerial approval. So the Minister stalls and by year-end lots of funds are unspent. Hence the underfunding in Vet Affairs, refugee processing, foreign aid projects and the like. They try to pass this off as sound financial management but they're being devious. had they announced 10-15% cuts in various programs up front, there's be plenty of debate and controversy. Instead they announce no cuts but deliver them anyway and no one knows until 6 months after the books have closed. It's quite deliberate.", "Mr. Flaherty remonstrated that he could easily show a balanced budget for 2014-15, even a surplus. No, instructed his boss, always thinking politically. It would be better to save the news – the surprise! – until the fall of 2015, that is during an election. Well, I knew it, half of Ottawa knew it, lets see if the Canadian voters know it.Harper and his motley crew of Reformers are once again caught in on of their numerous lies. After drastic cuts to Veterans Affairs, Immigration (350 million dollars), Aboriginal Affairs, healthcare, the CBC and many other government departments, the fact the budget was balanced in 2014/15 was an afterthought. Thus the question we'll be facing in October; do we want a government that will actually govern and make the hard decisions that are called for in these challenging times, or do we want a Harper government that spends 90% of its time campaigning instead of governing. I for one have seen enough off these far-right Harper Reformers. Lets hope the vast majority of Canadians agree with my 'sentiments'..", "Well look a surprise! If this guy running the country says it's a surprise then he's simply not running the country. Oh and by the way look squirrel!", 'So this article is saying that Harper should had a deficit this year with election in one month instead of last year with no election! This is the problem with liberal thinking it is not realistic, it is idealistic', "All parties play accounting tricks to make the numbers look better. The crazy thing about the Ontario Liberals, and I suspect federal, is that their numbers look horrible even after a massage and then they try and pull a 'stupid pet trick' by telling us the numbers look great. It's like calling a plane crash a thing of beauty.", 'The OECD says that in 2016 Canada will have 2.1% GDP growth, Oliver used 4.9% for 2016 in his Budget. So we need an honest fiscal update from the Con government. Which we will not get. Not that we are being fooled Mr. Simpson. We are being misled, to put it politely.', "Not only shouldn't you be fooled by the surprise budget surplus but you should be fairly disgusted by it. Harper is cooking the books with our money in order to get himself re-elected. Think of it. Thirty-five million of us; one of him. And who is more important?", "During Martin's stewardship of the economy he ran a grand total of $50 billion in surplus budgets. That is exactly the grand total that he took from working Canadians by cutting their EI benefits during the same period. Martin's hard kick in the groin to working Canadians was an unnecessary rounding error.", "Great piece. Most Canadians who have been keeping close watch on government decisions and issues over the past year (gee, I wonder what that percentage of Canadians is...15-20%, I suspect, thinking optimistically) weighed in early on this 'surprise' surplus with many 'decreased spending' stories. Now, the details that include the TIMING manipulation as well as the obfuscation of the real reasons for this 'tiny' (yup, that's the word!) surplus, blown all out of proportion. Thanks to Mr. Simpson for his last paragraph, too, about the irresponsibility of the NDP and Liberal candidates for pandering to seniors with the OAS roll-back. This is fiscally nuts. See populationpyramid. net/canada/2015 to see why."]
is_constructive (string, 2 distinct): ['yes', 'no']
is_constructive:confidence (numeric, 235 distinct): ['1.0', '0.6667', '0.68', '0.6641', '0.6538', '0.6685', '0.6631', '0.6947', '0.6754', '0.6567']
toxicity_level:confidence (string, 268 distinct): ['1', '0.6562\n0.3438', '0.68\n0.32', '0.6538\n0.3462', '0.6667\n0.3333', '0.6552\n0.3448', '0.675\n0.325', '0.6651\n0.3349', '0.6632\n0.3368', '0.6481\n0.3519']
did_you_read_the_article (numeric, 1 distinct): ['1']
did_you_read_the_article:confidence (numeric, 2 distinct): ['1.0', '0.6612']
annotator_comments (string, 26 distinct): ['\n\n', '\n\n\n', '\n\n\n\n\n\n\n\n\n\n\n', '\n\n\n\n\n\n\n\n\n\n\n\n', '\n\n\n\n\n\n\n\n\n', '\n\n\n\n\n\n\n\n\n\n', '\n', '\n\n\n\n\n\n\n\n\n\n\n\n\n', '\n\n\n\n\n\n\n\n\n\n\n\n\n\n', '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n']
expert_is_constructive (string, 2 distinct): ['no', 'yes']
expert_toxicity_level (numeric, 4 distinct): ['1.0', '2.0', '3.0', '4.0']
expert_comments (string, 60 distinct): ["I don't see any solutions being offered here, nor evidence presented to support this statement of why children are driven to the cities.", "This is really insulting, and I don't see any solutions being offered here, rather I see blame.", 'Perhaps if there was evidence presented here, this comment could be seen as somehow constructive.', 'Intelligent, informed, relevant to the article.', 'Lots of generalities used here "nobody really cares", "people just don\'t care" etc. without appropriate evidence.', "I can sense the writer's frustration here.", 'I found this comment quite powerful in terms of creating a contrast between Clinton and the rural woman.', 'This comment started off as constructive but then came the comment about "the previous guy\'s wife" and nepotism that is just opinion.', 'Evidence was presented.', "I'm not sure how this comment was viewed as constructive, as it is merely about the author vs the content of the piece Henein wrote."]
'''

def parse_score(s) -> int:
    if s.isnumeric():
        return int(s)
    return int(s.split()[0])

CONTEXT = "SOCC Corpus Toxicity Labeling"
TARGET = CuratedTarget(raw_name="toxicity_level", task_type=SupervisedTask.MULTICLASS, processing_func=parse_score)
COLS_TO_DROP = ['comment_counter']
FEATURES = []