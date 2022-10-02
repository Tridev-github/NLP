
Title : Rule based comment evaluation and hybrid comment summarisation

ABSTRACT:
Ranking of comments has an important role in determining whether to buy a product or not. Giving equal weightages to every comment believing all of them to be true is not helpful. Most online business sites evaluate their comments’ quality using criteria such as overall rating or helpfulness. Helpfulness is typically a score measured as the total votes given by consumers, which is an interesting way of defining a comment's relevance and quality. Research in the comments ranking area also uses this type of helpfulness score as their comments’ evaluation score. However, this method fails to identify these most recent comments with few votes. The major problem in customer reviews is not doubting them for giving a false review but what all products are the customers taking into account when said that it is the best or worst product.The proposed model starts by making some hypotheses on what might be the characteristics that are significant predictors for what it takes for a comment to get ranked high. These hypotheses are now tested with relevant techniques to know if they succeed in differentiating the comments. 

KEYWORDS: 
Relativity, point-of-interest, rule-based, Rule identification, Rule generalisation, Rule evaluation, Rule implementation, Comment summarisation.


INTRODUCTION: 
In the recent decade, online shopping has seen tremendous increase in customers. With the exponential growth in internet and infrastructure, it was no surprise that the e-commerce market experienced a corresponding boost. The market size of the online retail industry in India amounted to approximately 60 billion U.S. dollars in 2020.[1]

While buying products online has its ups and down. One of the major parts of online purchasing is product reviews. Gathering information based on other peoples is an essential part of the purchasing decision process.[2]

Simultaneously, it has been observed that not all reviews and corresponding product ratings contribute to helpful recommendations. Moreover, spammers exploit these review platforms illegally because of incentives involved in writing fake reviews. Currently, fake reviews and reviewers form a bulk of the review opus making review spamming an open research challenge. These spam reviews must be detected to nullify their contribution towards product ratings. Reviewer credibility analysis can be used to quickly address deceptive online reviews and reviewers. Pertinent research studies suggest various parameters that can affect reviewer credibility. These include: linguistic styles, review clarity and comprehensiveness, word count, sentence count in reviews, helpful vote count, evidence (speaker’s degree of certainty using certain propositional attitudes for example, certainly, surely) [3] [4]

Gathering information based on other people’s opinions is an essential part of the purchasing decision process [5]. With the rapid growth of the Internet, these conversations in online markets provide a large amount of product information. So when doing online shopping, consumers rely on online product comments, posted by other consumers, for their purchase decisions.

The fundamental question is how to judge a comment’s quality. Most online business sites evaluate their comments’ quality using criteria such as overall rating or helpfulness. Helpfulness is typically a score measured as the total votes given by consumers, which is an interesting way of defining a comment's relevance and quality. Research in the comments ranking area also uses this type of helpfulness score as their comments’ evaluation score [6]. However, this method fails to identify these most recent comments with few votes. For example, we may always observe that only a few comments published a long time ago have a high helpfulness score in a product site, and most other comments have no votes. The reason for the phenomenon is that most people only read the first few pages of comments before making their purchase decisions. A new comment that has just appeared on the product site and has not received any votes until recently may remain at the bottom of the comment list. This comment may contain important information about this product, thus has the potential to rise to the top of the list.

The major problem in customer reviews is not doubting them for giving a false review but what all products are the customers taking into account when said that it is the best or worst product. Let us say, a particular buyer must have used some limited number of products and the product he is reviewing may be the best among what he used and another buyer who has used a vast number of products might not feel the same way about it.

Most of the present-day recommender systems tend to use rating-information to quantify user-user, user-item and item-item similarities. Several techniques like clustering, nearest-neighbour methods, matrix manipulations, point-of-interest modelling have been used to model user interest patterns so as to maximise purchase satisfaction. But user-ratings are biassed by certain hidden factors like brand-adherence and product-prejudice. So, sole considerations of rating-oriented similarity and user-interest analysis are rendered useless in the complex modern-day setting. The limited capabilities of user-ratings have given way to heuristic-driven, context-driven, sentiment and emotion-driven user-interest profiling.[7][8]

Ranking comments in the runtime is time taking if machine learning algorithms are used rather than depending on rankings given by users, so many of the e-commerce sites do not rank comments dynamically or prefer depending on users to rank them.[9]

However, a large number of comments for a single product may make it harder for people to evaluate the true underlying quality of a product. So, machine learning techniques should be made use of while maintaining the runtime efficiency for the proposed solution. [10]

Hence, the proposed rule-based model can be used for ranking of comments. The flow of the model starts with making hypotheses of what might the rules be for a comment that could possibly be  ranked high. Following the making of hypotheses, the support and confidence of each rule needs to be calculated by implementing all the relevant rules depending on what a rule is, to be applied on test data. There by getting the support of a particular rule we decide if the support is viable to be taken into account or not and the confidence is, as for the percentage the rule correctly classifies the given data. 

The model has 3 stages namely RULE IDENTIFICATION, RULE GENERALISATION, RULE IMPLEMENTATION, COMMENT SUMMARISATION.

RULE IDENTIFICATION:
In this stage, we search for all possible rules which have high confidence. This is a recursive stage which  includes, making hypotheses, implementing techniques such as text clustering, n-gram model, pagerank algorithm, naive bayes approach etc.. The techniques are generally dependent on the rule, the relevant techniques are applied and the rules are selected based on the only factor, confidence of the rule.

RULE GENERALISATION:
This stage focuses on improving the support of rules that is being generated by generalising the rules, the approaches followed are SVD, Sentiment analysis, bag of words, clustering, K-L divergence, maximum general entropy etc.. This is a step where the contradictions of rules are removed and the support of the generated rules is increased.

RULE IMPLEMENTATION:
An API call is made to retrieve the comments in run time, the rules generated and generalised are implemented on the retrieved comments to rank them. A function with parameters of support and confidence of each rule that falls under a particular rule is used and the summation of all the values is used to get the final rank of every comment.  

COMMENT SUMMARISATION:
This step consists of summarisation of comments, the only difference in this summarisation and previously existing techniques is that this approach gives a weight to each comment and each word by the above mentioned function. Thereby countering spamming and the relativity of a comment.
  
So forth achieving the following goals, resolving relativity of a comment, making it interdependent to user’s experience, countering spamming, generating a clean database for further rule generation, decreasing runtime and other cross platform applications which could make use of users credibility.









ARCHITECTURE:





CONTRIBUTION:

An Entropy-based Method with Word Embedding Clustering to rank the comments:

Unlike general comment classification based on the words used in the comment, which is way simpler and not of particular significance, a comment ranking method is efficient in particular cases, not just the classification whether it is positive or negative but a score is required to specify how far it is from an ideal comment.A large number of comments for a single product may make it harder for people to evaluate the true underlying quality of a product. In this situation, consumers tend to focus on the average rating of a product, like the number of stars on Amazon.com. But in reality, some products can easily obtain high average ratings by cheating while some other products may get unfair low ratings. Therefore, it is very important to extract these relevant and high-quality comments from the product site, which can help consumers obtain accurate information about this product. Online shopping has seen a drastic rise through years, Canadian online shoppers reached $57.4 billion in 2018, being such a diverse market has its disadvantages.  The most obvious one is the lack of interactivity. Collecting the opinions of other people who used the same products is an essential part in the purchasing process. While in this case just classification of the comments into positive and negative alone may not be sufficient. As we cant afford customers wasting their time to go through each of the comments to make sure they get the quality product. So ranking comments with a reasonable criterion should be employed. Unlike general comment classification based on the words used in the comment, this paper has considered an “ideal” comment and calculated its entropy by using vivid equations and measured how farther are other comments from the ideal one inorder to rank the comments,  the bag of word clusters model is used instead of bag-of-words model that treats each word as an independent item, we group semantic-related words as clusters using pre-trained word2vec word embeddings and represent each comment as a distribution of word clusters.
 Kullback–Leibler (K-L) divergence:

(Maximum General Entropy Comment):

(General Entropy) :

Normalised discounted cumulative gain is used to evaluate the results.
nDCGm = DCGm/IDCGm ,
It is often used to measure the effectiveness of web search engine algorithms, but it can also be applied to text ranking applications. If a comment with high graded relevance appears lower in the ranking result, it will be penalised as the graded relevance value is reduced logarithmically proportional to the position of the ranking result. To achieve high DCG value, the algorithm should rank a high relevance comment higher than low relevance one. Pre-trained semantic related clusters are compared to a bag of words, saying that it treats each word as an independent word.KL divergence algorithm to rank comments is compared to vivid classifiers, as in here you find how far the considered comment is to an ideal comment. This method has low computational cost since it only requires statistical information from the text. This method can be applied to most of the product’s comments ranking applications since only the comments texts are needed.A sufficient amount of computational power can be saved by using our text representation method. Another advantage of the bag of word clusters model is that it considers the semantic relationship between words, where words with the same semantic meaning will be treated as the same item. So, this  model can perform better when making a comparison between two texts.Using KL divergence to maximum general entropy by using a comment which is widely used by people and calculating the general entropy to check by what it differs from the considered comment. 

An approach to generate an abstract summary from extractive summary using WordNet ontology:

This paper presents a unique approach to generating abstractive summary from extractive summary by utilising WordNet ontology in order to generate abstractive summary. Experimental results have shown that the generated summary is well-compressed, grammatically correct, and easily readable by humans. The popularity of the internet in this era is increasing tremendously and reading entire texts for necessary relevant information is clearly a highly time consuming and tedious task. In today’s life, getting correct information is very important. In the traditional system, the summarization process was time consuming and hectic. Performance metrics were always a concern. To improvise and overcome the traditional system, a new method called auto-summarization came into effect.In earlier systems, summarization was a time-consuming and stressful task.
Metrics for performance were always an issue. To improve and surpass the previous methodology, a new method known as auto-summarization was implemented. This auto summary tool is essential in today's social workforce. Automatic summarization has emerged as one of the most appealing and intriguing research subjects for extracting important information from a large collection of documents. The primary goal of automated summarization is to provide a solution for users to obtain their data in the shortest amount of time possible. 
Term Frequency: 

Description: Where ni, j is the number of occurrences of term ti
in sentence sj and ∑k nkj is the sum of number of occurrences of all the terms in sentence sj.
Equation 2: 

Description: Where, maxn= maximum frequency of any term in the document
N= no of sentences in the document
ni = total no of occurrences of term i in the document
The evaluation method is conducted by giving five articles to evaluators. Then, the evaluators read the document and generate a summary. Then there is a comparison between the system and humans. Then a graph is generated. The statics system generated 10 sentences from each document, and humans generated random no of sentences from each document. But there is a good ratio of common sentences between human systems.The paper does not highlight previous attempts at exploring this problem or there have not been proper attempts at exploring summarization through abstraction using hybrid summarization techniques.The hybrid technique can be seen to be effective for summarization of many documents when compared to other online extractive tools and general abstractive summary is very well compressed and is in a readable format for humans.The biggest and most obvious disadvantage is that the time to summarisation exponentially increases as the size of the document increases. The summarisation accuracy is also not consistent in the current form of the proposed solution.

Reviewer Credibility and Sentiment Analysis Based User Profile Modeling for Online Product Recommendation:

Recommendations based on the user's interest have a higher click ratio to that of generic recommendations. Deciphering the user’s interest is a complex task. While considering the credibility of the one who writes reviews is as important as considering a review to make a decision to buy a product. So, the other problem being, constructing a model that is robust to fake and unworthy reviews.The present-day recommender systems tend to use the user's rating-information to quantify user-user, user-item and item-item similarities. Several techniques like clustering, nearest-neighbour methods, matrix manipulations, point-of-interest modelling have been used to model user interest patterns so as to maximise purchase satisfaction. But user ratings are biassed by certain hidden factors like brand adherence and product-prejudice. So, sole considerations of rating-oriented similarity and user-interest analysis are rendered useless in the complex modern-day setting. The limited capabilities of user-ratings have given way to heuristic driven, context-driven, sentiment and emotion-driven user interest profiling.
The coherence of text is contemplated by its syntactic and semantic structure and relations. It measures the context
similarity of review sentences with candidate features, and thereby measures the consistency of context in reviews.

Representative rating (RR) is a quantitative indication of the opinion of masses. It is a quantity that represents
product rating, such that it is inclusive of all the reviews for the product. 

Expertise factor depends on the star-ratings given by users to products. 


MAP is an extension of Average Precision (AP) where we take the average of all AP’s to calculate MAP. Average precision is a measure that combines recall and precision for ranked retrieval results. The mathematical equations for the metrics MAP and AP are listed. Using MAP@N as the metrics, where N is the number of products recommended to each user. shows the variation of Mean Average Precision with Number of Recommendations.
The comparative analysis of various latent factors contributing to reviewer credibility analysis. Using mean product ratings for evaluating expertise factor (ESMean)  does not give appreciable results. Addition of the proposed α factor (representative rating) to mean product rating (ESRR) slightly improves the MAP performance of CISER. Hence, ESRR better models reviewer expertise compared to ESMean. It is evident that trust factor and network importance factor greatly contribute towards recommendation performance. An efficient linear combination of these three factors leads to superior performance.This research counteracted the problem of reducing the product search space for customers. The proposed CISER model recommends products with commendable accuracy, without the need of collecting and processing alternative data from social networks, surveys etc. The model augments heuristic-driven user interest profiling with reviewer credibility analysis and fine-grained feature sentiment analysis to devise a robust recommendation methodology.Recommendations based on the user's interests outperform generic recommendations in terms of click ratio. It's challenging to determine the user's interests. In order to decide whether to purchase a product, it is just as vital to take into account a reviewer's credibility as it is to do so. The other issue is creating a model that is resistant to fraudulent and untrustworthy reviews.In order to provide a reliable recommendation approach, this research effort enhances heuristic-driven user interest profiling with analysis of reviewer credibility and fine-grained feature sentime.So, the proposed model is CISER, Candidate feature extraction module, which extracts important features based on context and sentiment confidence. Reviewer credibility analysis gives a method for making our model resistant to fraudulent and unreliable reviews and reviewers. Reviewer credibility analysis proposes a method of associating knowledge, trust, and influence scores with reviewers in order to weigh their opinions based on their credibility. The user interest mining module mines interest patterns using the aesthetics of review authoring as heuristics. The candidate feature sentiment assignment module assigns ratings to candidate features in review depending on the polarity of their fastText sentiment. Finally, for purchase recommendations, the recommendation module employs credibility weighted sentiment rating of user desired characteristics For quantitative examination of many alternative items, the suggested recommendation approach employs not only numeric ratings, but also sentiment expressions connected with features, customer preference profile, and reviewer credibility.




Multiple Text Document Summarization System using hybrid Summarization technique:

Critical Analysis:
This work proposes a novel method for creating abstractive summary from extractive summary by exploiting the WordNet ontology. The resulting summary is well-compressed, grammatically accurate, and easily readable by humans, according to experimental data. Summarization was a time-consuming and demanding process in previous systems. Performance metrics were always a concern. A new approach known as auto-summarization was introduced to improve and surpass the prior methods. In today's social workforce, this auto summary feature is crucial. One of the most interesting and exciting research topics for extracting significant information from a big collection of texts is automatic summarization. The fundamental purpose of automated summarization is to give consumers a solution for obtaining their data in the least amount of time feasible. The study uses the related work section to highlight the work done in extractive summarization and claims that the development of abstractive summaries has been less challenging, citing it as a tough problem.
This method uses the Hyperlink-Induced Topic Search algorithm via link analysis to rate web pages based on their content.

Equations Implemented:
Term Frequency - 

Where ni, j is the number of occurrences of term ti
in sentence sj and ∑k nkj is the sum of number of occurrences of all the terms in sentence sj.
Equation 2 - 

Where, maxn= maximum frequency of any term in the document
N= no of sentences in the document
ni = total no of occurrences of term i in the document

Comparison to other methods:

Sr. No
Name of Tool
No of documents supported
Futures
1
Auto Summarizer (T1)
Single (DOC, txt)
No browse option, select only upto 2000 sentences
2
Free Summarizer (T2)
Single (DOC, txt)
No button for clear text, no browse option, select upto 2500 sentences
3
Text Comparator (T3)
Single (DOC, txt)
No button for Clear text, no browse option, ask for how much % of summary should be generated. Select upto 1000 sentences
4
Open Text Summarizer (T4)
Single (DOC, txt)
Button for clear text, no browse option, ask for how much % of summary should be generated, select upto 4000 sentences
5
Tools4noobs (T5)
Single (DOC, txt)
Button for clear text, select more than 4000 sentences
6
SMMR (T6)
Single (PDF, DOC, txt)
Option for browser file, copy paste, select more than 5000 sentences, accepts pdf, doc, txt file
7
Proposed System (Multi-Document Summerization System)
Multiple (PDF, DOC, txt)
Option for browse file, accepts pdf/doc/txt file for multiple files, select more than 10000 sentences


When compared to other online extractive tools, the hybrid approach can be demonstrated to be successful for summarising multiple documents, and the overall abstractive summary is extremely well compressed and in a legible manner for humans. The most apparent downside is that the time required to summarise grows exponentially with the size of the material. The precision of summarisation is also inconsistent in the existing form of the suggested solution.


Comment Ranking Diversification in Forum Discussions:

Critical Analysis:
This study proposes a novel method for ranking comments within a forum. Rather than simply ordering the comments by a specific score, they propose re-ranking the comments using a top K comment diversification re-ranking model based on Maximal Marginal Relevance and evaluating the impact of this rank by semantic diversity, inclusion of lower-ranked comments' semantics, and redundancy within a specific forum. This subject was picked since traditional ranking systems such as Facebook, Amazon, and Reddit forums are widely used, but many people confront a similar issue. Posts are largely rated based on their content or rating, such as the number of responses, individuals who thought the remark useful, upvotes, and so on. As a result, the higher ranked posts tend to solely represent the majority perspective because the majority is interacting with and promoting material. Text grading systems are ubiquitous, but many of them have a flaw. When postings are rated largely on their linguistic content, comparable posts usually obtain similar scores. Furthermore, because the dominant group is more likely to upvote articles expressing their point of view, higher ranking postings often only represent the consensus. Viewers who just look at the top posts in forums with thousands of entries may only be exposed to the consensus perspective. If the ground truth semantics of each comment were known as a priority, the issue might be avoided by normalising comment scores by the number of comments with equivalent semantics. Within the framework of a HarvardX course discussion forum, we use a top K comment diversification re-ranking model and analyse its influence in three categories: (1) semantic variety, (2) incorporation of the semantics of lower-ranked comments, and (3) repetition.

Equations Implemented: 

Maximal-Marginal Relevance (MMR) - 

To obtain the updated score, s', a single parameter modifies the trade-off between the original comment score, s, and its maximum cosine similarity across all comments that have already been added to the new ranking, c. For example, = 1 ranks fully by score, but = 0 chooses the most diverse comments regardless of score. 

Comparison to other methods:
On the gold dataset, word-level comment embedding approaches (word2vec, Gated CNN, LDA) fared worse than a basic TFIDF vector representation alone, with a traditional application of dimensionality reduction using PCA achieving the maximum accuracy.


Embedding Method
Median Quantile Difference
Logistic Regression Accuracy
TCIDF
0.338
0.841
PCA + TFIDF
0.434
0.867
LSA + TFIDF
0.431
0.867
NMF + TFIDF
0.416
0.861
LDA + TFIDF
0.129
0.815
Word2Vec + TFIDF
0.205
0.815
Word2Vec + nBOW
0.167
0.815
Gated CNN + TFIDF
0.116
0.786


When the TFIDF embedding is compared to PCA and LSA, it is clear that dense embeddings have advantages. Surprisingly, when paired with TFIDF, word2vec and Gated CNN performed worse than TFIDF.
Our word2vec model was most likely trained on the Google News corpus, which is a semantically distinct and considerably larger corpus than student comments in an online course. As a consequence, word embeddings linked to course content were compressed into a smaller space in comparison to the model's broader embeddings.

This is a novel way for categorising comments, and while fresh ideas may appear unsuitable at first, this solution might be tweaked and employed in other situations. In MMR assessments, comments ranked higher by diversity did not perform substantially better or worse than the gold baseline. When comments with modest diversity were considered, the MMR ranking was picked much more than the golden baseline. The remarks picked in the case of severe diversity were virtually wholly random and considerably different from the baseline.


Feature based Summarizing and Ranking from Customer Reviews:

Critical Analysis:
Customer feedback on particular aspects of a tourism offering is analysed. Based on this study, the ultimate objective is to develop a feature-based summary of a product. Users often choose characteristics for a variety of elements while submitting thoughts to travel websites (cleanliness, location, etc.). It is becoming increasingly apparent that educational big data can be an invaluable educational asset when it comes to advancing academic reform in a variety of ways. As a result of the rapid development of academic data, a big data approach to education is based on the analysis of student learning behaviors, attributes, and psychological states as a result of big data. It has been found that a great deal of educational big data can be applied to teaching innovation, academic administration, and research management. As an example, some of the applications of this type include predicting student academic performance, recommending students for employment, and providing financial assistance to students from low-income households. Different empirical studies have demonstrated that it is possible to predict the performance of students in the following term based on the results of empirical studies. The goal of the study topic of data mining is to automate the process of obtaining such valuable knowledge. There are many real-world applications of this technology, including e-commerce, corporate intelligence, information monitoring, and public polls, which have all benefited greatly from it. It is the process of identifying the feelings, attitudes, and opinions that an author has conveyed in their writing about a certain subject that is known as opinion mining. In recent years, there has been a noticeable increase in the number of review websites on the internet, where customers have the ability to leave comments on a service (such as a hotel or restaurant) and give it a positive or negative rating. In addition to offering new users guidance and assisting them in making travel arrangements, these websites are valuable tools that assist them with their travel planning.

Equations Implemented: 

selecting the class y with the highest conditional some linguistic context xX and predicting the correct probability p in the context x: XY with a conditional probability model by simply selecting the class y with the highest conditional some linguistic context xX and predicting the correct probability p in the context x.

Definition of conditional probability of p(y/x).

y denotes the result, x the history (or context), k the number of features, and Z(x) a normalising factor.

Comparison to other methods:



Subjective Class
Objective Class 
Precision (%)
77.98
87.34
Recall (%)
93.63
62.44
F-score (%)
85
72.82

This table represents the performance of the classifier when using the IR metrics.



Precision (%)
Recall (%)
F-score(%)
Adjacent Based
68.65
57.93
62.69
Pattern Based
59.65
59.95
59.72
Paper’s Approach
75.65
82.77
78.45

 This table shows prediction features and opinion pairs based on different IR metrics and comparative methods.

The most significant benefit that can be recognised in this study is a new and creative manner of capturing the link between the product description and the experience that buyers or customers have when purchasing or experiencing the goods. This strategy appears to be far more successful. However, the method described does not appear to be capable of handling sentences when noun phrases are compared to one another. The model's performance has dropped dramatically, which is not ideal because things are constantly compared in assessments of experiences or items. Nonetheless, the strategy may most likely be enhanced with more model training and data categorization.

EVALUATION METHODS:
Most online business sites evaluate their comments’ quality using criteria such as overall rating or helpfulness. Helpfulness is typically a score measured as the total votes given by consumers, which is an interesting way of defining a comment's relevance and quality. Research in the comments ranking area also uses this type of helpfulness score as their comments’ evaluation score. However, this method fails to identify these most recent comments with few votes. For example, we may always observe that only a few comments published a long time ago have a high helpfulness score in a product site, and most other comments have no votes. The reason for the phenomenon is that most people only read the first few pages of comments before making their purchase decisions. A new comment that has just appeared on the product site and has not received any votes until recently may remain at the bottom of the comment list. This comment may contain important information about this product, thus has the potential to rise to the top of the list.
 Mean Average Precision (MAP) is an extension of Average Precision (AP) where we take the average of all AP’s to calculate MAP. Average precision is a measure that combines recall and precision for ranked retrieval results.  
PSO for tuning coefficients for linear combination of feature scores and reviewer credibility factor scores. We empirically evaluate the tuning performances of PSO and other swarm optimization algorithms like Cuckoo Search, BAT and Artificial Bee Colony Optimisation.
Normalised Discounted Cumulative Gain (nDCG)) Given a ranked list with m comments and its DCG value, the normalised Discounted Cumulative Gain is computed as, nDCGm =DCGm/IDCGm, where IDCGm is the Ideal Discounted Cumulative Gain. Assigning each comment a relevance score, to achieve higher nDCG value, original comments should rank higher than other comments.




COMPARISON OF BASE PAPERS:  

Datasets/corpora
data sources
performance
scope
applications
approaches used
evaluation metrics
Amazon Product dataset 
Amazon developers account
This method has low computational cost since it only requires statistical information from the text. This method can be applied to most
of the product’s comments ranking applications since only the comments texts are needed.

A sufficient amount of computational power can be saved by using our text representation method. Another advantage of the bag of word clusters model is that it considers the semantic relationship between words, where words with the same semantic meaning will be treated as the same item. So, this  model can perform better when making a comparison between two texts.
Using KL divergence to maximum general entropy by using a comment which is widely used by people and calculating the general entropy to check by what it differs to the considered comment.



A large number of comments for a single product may make it harder for people to evaluate the true underlying quality of a product. In this situation, consumers tend to focus on the average rating of a product, like the number of stars on Amazon.com. But in reality, some products can easily obtain high average ratings by cheating while some other products may get unfair low ratings. Therefore, it is very important to extract these relevant and high-quality
comments from the product site, which can help consumers obtain accurate information about this product.



Unlike general comment classification based on the words used in the comment, which is way simpler and not of particular significance, a comment ranking method is efficient in particular cases, not just the classification whether it is positive or negative but a score is required to specify how far it is from an ideal comment.
An Entropy-based Method with Word Embedding Clustering for comment ranking
K-L Divergence, K-means, nDCG, Word2Vec


Normalised discounted cumulative gain is used to evaluate the results.
nDCGm = DCGm/IDCGm ,
It is often used to measure the effectiveness of web search engine algorithms, but it can also be applied to text ranking applications. If a comment with high graded relevance appears lower in the ranking result, it will be penalized as the graded relevance value is reduced logarithmically proportional to the position of the ranking result. To achieve high DCG value, the algorithm should rank a high relevance comment higher than low relevance one. 



DUC and TAC datasets
Official website:
tac.nist.gov, duc.nist.gov
The paper does not highlight previous attempts at exploring this problem or there have not been proper attempts at exploring summarization through abstraction using hybrid summarization techniques.The hybrid technique can be seen to be effective for summarization of many documents when compared to other online extractive tools and general abstractive summary is very well compressed and is in a readable format for humans.
The popularity of the internet in this era is increasing tremendously and reading entire texts for necessary relevant information is clearly a highly time consuming and tedious task. In today’s life, getting correct information is very important. In the traditional system, the summarization process was time consuming and hectic. Performance metrics were always a concern. To improvise and overcome the traditional system, a new method called auto-summarization came into effect.



This paper presents a unique approach to generating abstractive summary from extractive summary by utilising WordNet ontology in order to generate abstractive summary. Experimental results have shown that the generated summary is well-compressed, grammatically correct, and easily readable by humans.
HITS algorithm
Hyperlink-Induced Topic Search is an algorithm via link analysis used to rate web pages based on their content.



The evaluation method is conducted by giving five articles to evaluators. Then, the evaluators read the document and generate a summary. Then there is a comparison between the system and humans. Then a graph is generated. The statics system generated 10 sentences from each document, and humans generated random no of sentences from each document. But there is a good ratio of common sentences between human systems.
Amazon camera review dataset
Amazon developers account
Advantages/merits of proposed solution in your view.
This research counteracted the problem of reducing the product search space for customers. The proposed CISER model recommends products with commendable accuracy, without the need of collecting and processing alternative data from social networks, surveys etc. The model augments heuristic-driven user interest profiling with reviewer credibility analysis and fine-grained feature sentiment analysis to devise a robust recommendation methodology.



The present-day recommender systems tend to user’s rating-information to quantify user-user, user-item and item-item similarities. Several techniques like clustering, nearest-neighbour methods, matrix manipulations, point-of-interest modelling have been used to model user interest patterns so as to maximise purchase satisfaction. But user ratings are biassed by certain hidden factors like brand adherence and product-prejudice. So, sole considerations of rating-oriented similarity and user-interest analysis are rendered useless in the complex modern-day setting. The limited capabilities of user-ratings have given way to heuristic driven, context-driven, sentiment and emotion-driven user interest profiling.



Recommendations based on the user's interest have a higher click ratio to that of generic recommendations. Deciphering the user’s interest is a complex task. While considering the credibility of the one who writes reviews is as important as considering a review to make a decision to buy a product. So, the other problem being, constructing a model that is robust to fake and unworthy reviews.
REVIEWER CREDIBILITY ANALYSIS:The credibility aspects related to review utility are extracted from the data-set and described.
USER INTEREST MINING,  CANDIDATE FEATURE EXTRACTION, CANDIDATE FEATURE SENTIMENT SCORING.
The comparative analysis of various latent factors contributing to reviewer credibility analysis is demonstrated in Figure 13. Using mean product ratings for evaluating expertise factor (ESMean) does not give appreciable results. Addition of the proposed α factor (representative rating) to mean product rating (ESRR) slightly improves the MAP performance of CISER. Hence, ESRR better models reviewer expertise compared to ESMean. It is evident that trust factor and network importance factor greatly contribute towards recommendation performance. An efficient linear combination of these three factors leads to superior performance.
MAP is an extension of Average Precision (AP)
where we take the average of all AP’s to calculate MAP. Average
precision is a measure that combines recall and precision for
ranked retrieval results. The mathematical equations for the
metrics MAP and AP are listed.
Using MAP@N as the metrics, where N is the number of products recommended to each user. shows the variation of Mean Average Precision with Number
of Recommendations.



Dataset of doc, docx, txt, pdf files
Author’s institutes thesis material  
To evaluate performance, the authors take 5 documents all from the same domain. From comparing with other online extractive tools it is noted that this is one of the only systems to accept PDF files and accepts all 5 of the files provided. It also shows us that this system supports the most number of lines. When we compare the proposed system to that of human generated summarisation we see that the number of sentences generated were more and the number of common sentences were much lower. 
The internet's popularity is skyrocketing in this day and age; reading complete texts and papers for vital relevant information is plainly a time-consuming and laborious activity. In today's world, having accurate information is critical. The summarising procedure was time consuming and stressful in the prior manner.
Metrics for performance were always an issue. To improve and surpass the previous methodology, a new method known as auto-summarization was implemented. This auto summary tool is essential in today's social workforce.
The primary goal of automated summarization is to provide a solution for users to obtain their data in the shortest amount of time possible. Also, readers may disregard beneficial information as irrelevant, because it is common for readers to want to read the entire and lengthy document in search of relevant language, but sadly, he/she loses patience until the conclusion and some useful information may be lost.
Linguistic Analysis, Redundancy Detection, Stop Word, Stemming, Sentence Representation, Term Frequency, Word Graph Generation, Domain Ontology, Meronymy, Holonymy
Accuracy, Error, Precision, Specificity, Fbeta-measure
Forum with comments
forum discussions of an edX course, HarvardX: HDS3221.2x
Christianity Through Its Scriptures


While this project only considers the comments for the online edX course, “Harvardx Cristiantity Through Its Scriptures”, the increase visibility of diversity of comments between the many students can aid in the debunking of misconceptions that are held by the majority of the forum respondents.
Large-scale commenting platforms, e.g.
Facebook, edX, Reddit, etc., can consider the importance of
ranking diversification on learning and user experience, and
This new method can provide a new way to serve comments to the user such that they are not only served the most popular content or opinions.
Automated Gold Data Generation, Maximal-Marginal Relevance (MMR), Comment Embedding Model Selection
Inclusion, diversification, Redundancy, Reliability, Agreement, Cohen Kappa
Tourism reviews
Reviews from TripAdvisor
In terms of precision when compared to Adjacent based approaches it is 7% better. At recall, the proposed approach is better by a whopping 24.84%. When considering F-score the applied approach is 15.76% better.

When pattern based approach is considered for precision, it is better by 16%, for recall method it is better by 22.82% and for F-score it is better by 18.73%.
Opinion mining approaches have grown in popularity for automatically processing customer reviews in order to extract product characteristics and user opinions stated about them.
This method can be used for e-commerce, business intelligence, information monitoring and public polls.
Extraction of features, Opinion phrases extraction, feature and opinion pair prediction
Precision, Recall, F-score



CODE:
import pandas as pd
df = pd.read_csv("C:/Users/methu/Downloads/fake reviews dataset.csv")
df = df[['category','label','text_']]
df
groupDf = df.groupby(df['category'])
groupDf
setOfCat = set()
temp = df[['category']].values.tolist()
for x in temp:
    for y in x:
        setOfCat.add(y)
print(setOfCat)
setOflab = set()
temp = df[['label']].values.tolist()
for x in temp:
    for y in x:
        setOflab.add(y)
print()
print()
print(setOflab)
trueDf = df[df['label']=='OR']
cat1TrueDfListOfText = trueDf[trueDf['category']=='Tools_and_Home_Improvement_5']['text_'].values.tolist()
print(cat1TrueDfListOfText)
fDf = df[df['label']=='CG']
cat1FDfListOfText = fDf[fDf['category']=='Tools_and_Home_Improvement_5']['text_'].values.tolist()
print(cat1FDfListOfText)
punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
preprocesscat1TrueDfListOfText = []

for sentence in cat1TrueDfListOfText:
    sent = ""
    for letter in sentence:
        if letter not in punc:
               sent += letter.lower()
    preprocesscat1TrueDfListOfText.append(sent)
print(preprocesscat1TrueDfListOfText)

preprocesscat1FDfListOfText = []

for sentence in cat1FDfListOfText:
    sent = ""
    for letter in sentence:
        if letter not in punc:
               sent += letter.lower()
    preprocesscat1FDfListOfText.append(sent)
print(preprocesscat1FDfListOfText)
import nltk
from nltk.corpus import stopwords
stp = stopwords.words('english')

temp = []

for sentence in preprocesscat1TrueDfListOfText:
    subTemp = []
    for word in sentence.split(" "):
        if word not in stp:
            subTemp.append(word)
    temp.append(subTemp)
preprocesscat1TrueDfListOfText = []
for lis in temp:
    preprocesscat1TrueDfListOfText.append(" ".join(lis))
print(preprocesscat1TrueDfListOfText)

temp = []

for sentence in preprocesscat1FDfListOfText:
    subTemp = []
    for word in sentence.split(" "):
        if word not in stp:
            subTemp.append(word)
    temp.append(subTemp)
preprocesscat1FDfListOfText = []
for lis in temp:
    preprocesscat1FDfListOfText.append(" ".join(lis))
print(preprocesscat1FDfListOfText)
import matplotlib.pyplot as plt
import numpy as np
from apyori import apriori
from nltk.sentiment import SentimentIntensityAnalyzer
SA = SentimentIntensityAnalyzer()
posNnegInCat1True = []
total=0
for sentence in preprocesscat1TrueDfListOfText:
    total+=1
    neg = []
    pos = []
    for word in sentence.split(" "):
        if SA.polarity_scores(word)["pos"]==1:
            pos.append(word)
        elif SA.polarity_scores(word)["neg"]==1:
            neg.append(word)
    
    if len(neg)!=0 and len(pos)!=0 and len(pos)>len(neg):
        posNnegInCat1True.append([pos,neg])
        
print(len(posNnegInCat1True),posNnegInCat1True)
posNnegInCat1F = []
for sentence in preprocesscat1FDfListOfText:
    neg = []
    pos = []
    for word in sentence.split(" "):
        if SA.polarity_scores(word)["pos"]==1:
            pos.append(word)
        elif SA.polarity_scores(word)["neg"]==1:
            neg.append(word)
    if len(neg)!=0 and len(pos)!=0 and len(pos)>len(neg):
        posNnegInCat1F.append([pos,neg])
        
print(len(posNnegInCat1F),posNnegInCat1F)
joinedCat1True = (" ".join(preprocesscat1TrueDfListOfText)).split(" ")
joinedCat1True = [i for i in joinedCat1True if i!='']
print(joinedCat1True)
n3GramT = nltk.collocations.TrigramCollocationFinder.from_words(joinedCat1True)
n2GramT = nltk.collocations.BigramCollocationFinder.from_words(joinedCat1True)

joinedCat1F = (" ".join(preprocesscat1FDfListOfText)).split(" ")
joinedCat1F = [i for i in joinedCat1F if i!='']
print(joinedCat1F)
n3GramF = nltk.collocations.TrigramCollocationFinder.from_words(joinedCat1F)
n2GramF = nltk.collocations.BigramCollocationFinder.from_words(joinedCat1F)
n3GramT.ngram_fd.most_common(9)
n2GramT.ngram_fd.most_common(9)
n3GramF.ngram_fd.most_common(9)
n2GramF.ngram_fd.most_common(9)
n3GramWords = []
for i in n3GramT.ngram_fd.most_common(9):
    for j in i[0]:
        n3GramWords.append(j)
print(n3GramWords)

posTagN3 = nltk.pos_tag(n3GramWords)
print()
print(posTagN3)
n2GramWords = []
for i in n2GramT.ngram_fd.most_common(9):
    for j in i[0]:
        n2GramWords.append(j)
print(n2GramWords)

posTagN2 = nltk.pos_tag(n2GramWords)
print()
print(posTagN2)
from sematch.semantic.similarity import WordNetSimilarity
wns = WordNetSimilarity()
import mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
data = df
data.columns
posNnegInCat1True = []
posNum = 0
negNum = 0
total = 0
posNnegNum = 0
for sentence in preprocesscat1TrueDfListOfText+preprocesscat1FDfListOfText:
    total+=1
    neg = []
    pos = []
    for word in sentence.split(" "):
        if SA.polarity_scores(word)["pos"]==1:
            pos.append(word)
        elif SA.polarity_scores(word)["neg"]==1:
            neg.append(word)
    if len(neg)!=0:
        negNum+=1
    if len(pos)!=0:
        posNum+=1
    if len(neg)!=0 and len(pos)!=0:
        posNnegInCat1True.append([pos,neg])
        posNnegNum+=1
t=posNnegNum*2/3
print("Support :",posNnegNum/total)

posNnegInCat1True = []
posNum = 0
negNum = 0
total = 0
posNnegNum = 0
SUPP = t
for sentence in preprocesscat1TrueDfListOfText:
    total+=1
    neg = []
    pos = []
    for word in sentence.split(" "):
        if SA.polarity_scores(word)["pos"]==1:
            pos.append(word)
        elif SA.polarity_scores(word)["neg"]==1:
            neg.append(word)
    if len(neg)!=0:
        negNum+=1
    if len(pos)!=0:
        posNum+=1
    if len(neg)!=0 and len(pos)!=0:
        posNnegInCat1True.append([pos,neg])
        posNnegNum+=1
print("Confidence :",posNnegNum/SUPP)

CONCLUSION AND FUTURE DIRECTIONS:
The proposed rule based model can be used for ranking of comments. The flow of the model starts with making hypotheses of what might the rules be for a comment that could possibly be  ranked high. Following the making of hypotheses, the support and confidence of each rule needs to be calculated by implementing all the relevant rules depending on what a rule is, to be applied on test data. There by getting the support of a particular rule we decide if the support is viable to be taken into account or not and the confidence is, as for the percentage the rule correctly classifies the given data. 
The rule started with an assumption that if a part of the comment is true then there is a higher probability that the other information present in the same comment might also be true. Thereby applying the relevant techniques to test this hypothesis, we got significant confidence. The support of this was relatively low, hence the rule is generalised by calculating polarity of a comment. The rule was modified for  making the  rule in compliance with that, though we got a spike in suppor,t the trade-off was, increase in run time, as polarity of each word has to be calculated in the run time.
So forth achieving the following goals, resolving relativity of a comment, making it interdependent to user’s experience, countering spamming, generating a clean database for further rule generation, decreasing runtime and other cross platform applications which could make use of users credibility.
Our future direction includes extensive search of other rules and generalising them. When a large number of rules are applied they often conflict or overlap with each other, so a new module needs to be added to handle these inconsistencies.



REFERENCES:
R. Dong, M. P. O’Mahony, M. Schaal, K. Mccarthy and B. Smyth, "Combining similarity and sentiment in opinion mining for product recommendation", J. Intell. Inf. Syst., vol. 46, no. 2, pp. 285-312, Apr. 2016.
 Alberto, T. C., Lochter, J. V., and Almeida, T. A. Tubespam: Comment spam filtering on youtube. In 2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA) (2015), pp. 138–143
He, R., and McAuley, J. Ups and downs: Modelling the visual evolution of fashion trends with one-class collaborative filtering. In proceedings of the 25th international conference on world wide web (2016), pp. 507–517
Chen, C. C., and Tseng, Y. Quality evaluation of product reviews using an information quality framework. Decision Support Systems 50, 4 (2011), 755 – 768.
olov, T., Sutskever, I., Chen, K., Corrado, G. S., and Dean, J. Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems 26. Curran Associates, Inc., 2013, pp. 3111–3119
Nikfarjam, A., Sarker, A., Oconnor, K., Ginn, R., and Gonzalez, G. Pharmacovigilance from social media: Mining adverse drug reaction mentions using sequence labelling with word embedding cluster features. Journal of the American Medical Informatics Association : JAMIA 22 (03 2015).
Pragnya Addala, Text Summarization A Literature Survey, https://www.scribd.com/doc/235008952/Text-Summarization-Literature-Survey - scribd.com on Jul 24, 2014
Elena Baralis , Luca Cagliero , Saima Jabeen, Alessandro Fiori,Sajid Shah, Multi-document summarization based on the Yago ontology, Expert Systems with Applications 40 (2013) ;69766984,Elsevier
Lei Li and Tao Li, An Empirical Study of Ontology-Based Multi-Document Summarization in Disaster Management,In IEEE Transactions on Systems, Man, cybernetics: Systems
A. Kumar and A. Jaiswal, "Particle swarm optimised ensemble learning for enhanced predictive sentiment accuracy of tweets", Proc. ICETIT, pp. 633-646, 2020.
A. Kumar and A. Joshi, "Ontology driven sentiment analysis on social Web for government intelligence", Proc. Special Collection EGovernment Innov. India, pp. 134-139, 2017.
A. Kumar and A. Jaiswal, "Empirical study of twitter and tumblr for sentiment analysis using soft computing techniques", Proc. World Congr. Eng. Comput. Sci., vol. 1, pp. 1-5, 2017.
H. H. Do, P. Prasad, A. Maag and A. Alsadoon, "Deep Learning for Aspect-Based Sentiment Analysis: A Comparative Review", Expert Syst. Appl., vol. 118, pp. 272-299, Mar. 2019, [online] Available: http://www.sciencedirect.com/science/article/pii/S0957417418306456.
A. Dridi, M. Atzeni and D. Reforgiato Recupero, "FineNews: Fine-grained semantic sentiment analysis on financial microblogs and news", Int. J. Mach. Learn. Cyber., vol. 10, no. 8, pp. 2199-2207, Aug. 2019.
R. Dong, M. P. O’Mahony, M. Schaal, K. Mccarthy and B. Smyth, "Combining similarity and sentiment in opinion mining for product recommendation", J. Intell. Inf. Syst., vol. 46, no. 2, pp. 285-312, Apr. 2016.
Jee-Uk Heu, Iqbal Qasim, Dong-Ho Lee, FoDoSu : Multi-document summarization exploiting semantic analysis based on so-cial Folksonomy,In Information Processing and Management Elsevier
J.I. Sheeba and Dr.K. Vivekanandan, “Improved Sentiment Classification From Meeting Transcripts”, IJCSI International Journal of Computer Science Issues, Vol. 9, Issue 5, September 2012.
A. Da’u and N. Salim, "Sentiment-Aware Deep Recommender System With Neural Attention Networks", IEEE Access, vol. 7, pp. 45472-45484, 2019.
A. Said, E. W. De Luca and S. Albayrak, "Inferring contextual user profiles-improving recommender performance", Proc. 3rd RecSys Workshop Context-Aware Recommender Syst., 2011.
F. García-Sánchez, R. Colomo-Palacios and R. Valencia-García, "A social-semantic recommender system for advertisements", Inf. Process. Manage., vol. 57, no. 2, Mar. 2020.
J. Wei, F. Meng and N. Arunkumar, "A personalised authoritative user-based recommendation for social tagging", Future Gener. Comput. Syst., vol. 86, pp. 355-361, Sep. 2018, [online] Available: http://www.sciencedirect.com/science/article/pii/S0167739X18303078.
M. Xu and S. Liu, "Semantic-enhanced and context-aware hybrid collaborative filtering for event recommendation in event-based social networks", IEEE Access, vol. 7, pp. 17493-17502, 2019.
L. U, Y. Chai and J. Chen, "Improved personalised recommendation based on user attributes clustering and score matrix filling", Comput. Standards Inter., vol. 57, pp. 59-67, Mar. 2018.
H. Tahmasbi, M. Jalali and H. Shakeri, "Modelling Temporal Dynamics of User Preferences in Movie Recommendation", Proc. 8th Int. Conf. Comput. Knowl. Eng. (ICCKE), pp. 194-199, Oct. 2018.
S. E. Middleton, D. C. De Roure and N. R. Shadbolt, "Capturing knowledge of user preferences: Ontologies in recommender systems", Proc. 1st Int. Conf. Knowl. Capture, pp. 100-107, 2001.
X. Xie and B. Wang, "Web page recommendation via twofold clustering: Considering user behaviour and topic relation", Neural Comput. Appl., vol. 29, no. 1, pp. 235-243, 2018.
J. Al-Sharawneh and M.-A. Williams, "Credibility-based social network recommendation: Follow the leader", Proc. 21st Australas. Conf. Inf. Syst., pp. 1-11, 2010.
T. Ahmad and M.N Doja, “Rule Based System for Enhancing Recall for Feature Mining from Short Sentences in Customer Review Documents”, International Journal on Computer Science and Engineering (IJCSE), ISSN : 0975-3397, Vol. 4 No. 06, June 2012.
F. Wogenstein, J. Drescher, D. Reinel, S. Rill and J. Scheidt, “Evaluation of an Algorithm for Aspect-Based Opinion Mining Using a Lexicon-Based Approach”, WISDOM ’13, Chicago, USA, August 2013.
Carbonell, J.; and Goldstein, J. 1998. The use of MMR,
diversity-based reranking for reordering documents and producing summaries. In Proceedings of the 21st annual international ACM SIGIR conference on Research and development in information retrieval, 335–336.
Fevry, T.; and Phang, J. 2018. Unsupervised Sentence Compression using Denoising Auto-Encoders. In Proceedings of the 22nd Conference on Computational Natural Language Learning, 413–422
Gurevych, I.; and Strube, M. 2004. Semantic similarity applied to spoken dialogue summarization. In Proceedings of the 20th international conference on Computational Linguistics, 764. Association for Computational Linguistics
Kingma, D. P.; and Ba, J. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 
Liu, P. J.; Chung, Y.-A.; Ren, J.; et al. 2019b. SummAE: Zero-shot abstractive text summarization using length agnostic auto-encoders
B. Liu, Sentiment Analysis and Opinion Mining, San Mateo, CA, USA:Morgan & Claypool Publishers, 2012.
Amazon Customer Reviews Dataset, Dec. 2019, [online] Available: https://s3.amazonaws.com/amazon-reviews-pds/readme.html.
S. Raza and C. Ding, "Progress in context-aware recommender systems—An overview", Comput. Sci. Rev., vol. 31, pp. 84-97, Feb. 2019, [online] Available: http://www.sciencedirect.com/science/article/pii/S1574013718302120.
L. Chen, G. Chen and F. Wang, "Recommender systems based on user reviews: The state of the art", User Model. User-Adapted Interact., vol. 25, no. 2, pp. 99-154, June. 2015.
C. He, D. Parra and K. Verbert, "Interactive recommender systems: A survey of the state of the art and future research challenges and opportunities", Expert Syst. Appl., vol. 56, pp. 9-27, Sep. 2016, [online] Available: http://www.sciencedirect.com/science/article/pii/S0957417416300367.
S. Reddy, S. Nalluri, S. Kunisetti, S. Ashok and B. Venkatesh, "Content-based movie recommendation system using genre correlation" in Smart Intelligent Computing and Applications, Singapore:Springer, pp. 391-397, 2019.
X. Li, W. Jiang, W. Chen, J. Wu and G. Wang, "Haes: A new hybrid approach for movie recommendation with elastic serendipity", Proc. 28th ACM Int. Conf. Inf. Knowl. Manage., pp. 1503-1512, 2019.
 Woloszyn, V., dos Santos, H. D. P., Wives, L. K., and Becker, K. Mrr: An unsupervised algorithm to rank reviews by relevance. In Proceedings of the International Conference on Web Intelligence (New York, NY, USA, 2017), WI ’17, Association for Computing Machinery, p. 877–883.
tone, B., Dennis, S., and Kwantes, P. J. Comparing methods for single paragraph similarity analysis. Topics in Cognitive Science 3, 1 (2011), 92–122.
Shannon, C. E. A mathematical theory of communication. The Bell System Technical Journal 27, 3 (1948), 379–423.
Mikolov, T., Chen, K., Corrado, G. S., and Dean, J. Efficient estimation of word representations in vector space. CoRR abs/1301.3781 (2013)
Kaufmann, L. Clustering by means of medoids. Proc. Statistical Data Analysis Based on the L1 Norm Conference, Neuchatel, 1987 (1987), 405–416.
X. Li, W. Jiang, W. Chen, J. Wu, and G. Wang, ‘‘Haes: A new hybrid approach for movie recommendation with elastic serendipity,’’ in Proc. 28th ACM Int. Conf. Inf. Knowl. Manage., New York, NY, USA, 2019, pp. 1503–1512
A. Said, E. W. De Luca, and S. Albayrak, ‘‘Inferring contextual user profiles-improving recommender performance,’’ in Proc. 3rd RecSys Workshop Context-Aware Recommender System., Chicago, IL, USA, 2011. F. García-Sánchez, R. Colomo-Palacios, and R. Valencia-García, ‘‘A social-semantic recommender system for advertisements,’’ Inf. Process. Manage., vol. 57, no. 2, Mar. 2020, Art. no. 102153
Harsha Dave and Shree Jaswal, "Multi-Document Abstractive Summarization Based on Ontology", International Journal Of Engineering Research and Technology (IJERT) ICNTE-2015 Conference Proceedings, vol. 3, no. 1.
 B. Liu, “Sentiment Analysis and Subjectivity. Handbook of Natural Language Processing”, Second Edition, (editors: N. Indurkhya and F. J. Damerau), 2010.
 A. Kamal, “Subjectivity Classification using Machine Learning Techniques for Mining Feature-Opinion Pairs from Web Opinion Sources”, New Delhi – 110025, India.
J. Wiebe and E. Riloff, “Creating Subjective and Objective Sentence Classifiers from Unannotated Texts”.
T. Ahmad and M.N Doja, “Rule Based System for Enhancing Recall for Feature Mining from Short Sentences in Customer Review Documents”, International Journal on Computer Science and  Engineering (IJCSE), ISSN : 0975-3397, Vol. 4 No. 06, June 2012.
S.S. Htay and K.T. Lynn, “Extracting Product Features and Opinion Words Using Pattern Knowledge in Customer Reviews”, The Scientific World Journal, Volume 2013 (2013), Article ID 394758, September 2013.
G. Somprasertsri and P. Lalitrojwong, “Mining Feature-Opinion in Online Customer Reviews for Opinion Summarization”, Journal of Universal Computer Science, vol. 16, no. 6 (2010), 938-955, March 2010.
M. Hu and B. Liu, “Mining and Summarising Customer Reviews”, KDD’04, Seattle, Washington, USA, August 22–25, 2004.
F. Wogenstein, J. Drescher, D. Reinel, S. Rill and J. Scheidt, “Evaluation of an Algorithm for Aspect-Based Opinion Mining Using a Lexicon-Based Approach”, WISDOM ’13, Chicago, USA, August 2013.
B.Liu, “Web Data Mining”, Springer, 2008.
X. Ding, B. Liu, P.S. Yu, “A Holistic Lexicon-Based Approach to Opinion Mining”, WSDM’08, Palo Alto, California, USA, February, 2008.
J.I. Sheeba and Dr.K. Vivekanandan, “Improved Sentiment Classification From Meeting Transcripts”, IJCSI International Journal of Computer Science Issues, Vol. 9, Issue 5, September 2012.
S. Rill, S. Adolph, J. Drescher, N. Korfiatis, D. Reinel, J. Scheidt, O. Sch¨ utz, F. Wogenstein, R. V. Zicari, “A Phrase-Based Opinion List for the German Language”.
A. Esuli and F. Sebastiani. SentiWordNet: a publicly available lexical resource for opinion mining. In Proc. of LREC 2006 - 5th Conf. on Language Resources and Evaluation, Volume 6, 2006.
F. Batista, R. Ribeiro, “Sentiment Analysis and Topic Classification based on Binary Maximum Entropy Classifiers”, Procesamiento del Lenguaje Natural, Revista nº 50 marzo de 2013, pp 77-84.
K. Fragos, Y. Maistros and C. Skourlas, “A Weighted Maximum Entropy Language Model for Text Classification”
Multi-document Summarization with Maximal Marginal Relevance-guided Reinforcement Learning arXiv:2010.00117
 Learning Maximal Marginal Relevance Model via Directly Optimizing Diversity Evaluation Measures August 2015 pp113-122
Adaptive Maximum Marginal Relevance Based Multi-email Summarization AICI 2009: Artificial Intelligence and Computational Intelligence pp 417–424
F. Batista, R. Ribeiro, “Sentiment Analysis and Topic Classification based on Binary Maximum Entropy Classifiers”, Procesamiento del Lenguaje Natural, Revista nº 50 marzo de 2013, pp 77-84.
S.S. Htay and K.T. Lynn, “Extracting Product Features and Opinion Words Using Pattern Knowledge in Customer Reviews”, The Scientific World Journal, Volume 2013 (2013), Article ID 394758, September 2013.


