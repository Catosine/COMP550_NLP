# COMP550_19Fall
This is a repository for [COMP 550: Natural Language Processing](https://www.mcgill.ca/study/2019-2020/courses/comp-550) provied by McGill University in fall 2019.

# Projects
- [**Sentiment_Analysis**](https://github.com/Catosine/COMP550_NLP/tree/master/Sentiment_Analysis)  
This is the folder for the third question of the first assignment. I 
planned to implement a sentiment analysis model using naive bayes, 
logistic regression, and support vector machine to classify a binary 
distributed dataset (either positive or negative). The dataset is the 
same one used in [_Seeing starts: Exploiting class relationships for sentiment 
categorization with respect to rating scale_](http://www.cs.cornell.edu/home/llee/papers/pang-lee-stars.pdf) 
by Bo Pang and Lillian Lee in 2005. You may access the data from [here](http://www.cs.cornell.edu/people/pabo/movie-review-data/).
Also, the pretrained word embeddings, GloVe, is used in the experiment.

- [**Decipherment_with_HMM**](https://github.com/Catosine/COMP550_NLP/tree/master/Decipherment_with_HMMs)  
This is the folder for the third question of the second assignment. I 
planned to implement a standard hidden Markov model to solve a sets of 
ciphers encrypted in three different ways. The dataset is retrived from
professor, which may not be able to be uploaded here. In addition to
improve the tagging accuracy, I used nltk built-in corpus
[Penn Treebank](https://www.nltk.org/_modules/nltk/corpus.html)
to count the transition probilities.

# References
1. Pang, Bo, and Lillian Lee. "Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales." Proceedings of the 43rd annual meeting on association for computational linguistics. Association for Computational Linguistics, 2005.
2. Pennington, Jeffrey, Richard Socher, and Christopher Manning. "Glove: Global vectors for word representation." Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014.
3. Marcus, Mitchell, et al. "The Penn Treebank: annotating predicate argument structure." Proceedings of the workshop on Human Language Technology. Association for Computational Linguistics, 1994.
# License
This project is under the MIT license.
