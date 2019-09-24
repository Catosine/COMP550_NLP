# Sentiment Analysis
This is directory is used for question 3 in assignment 1.

## What does this code do?
1. Train a model;
2. Keep the record;
3. Generate the result;

## Required Package
1. numpy
2. nltk
3. sklearn
4. tqdm
5. matplotlib
6. seaborn
7. pandas

## Required Dataset
1. [rt-polaritydata.tar.gz](http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz)
2. [glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)

## How to run the code?  
~~~~
# Suppose the positive dat has been kept at 
# path/to/the/data/rt-polaritydata/rt-polarity.pos
# path/to/the/data/rt-polaritydata/rt-polarity.neg

python3 a1q3.py --model BNB \
                --data_dir path/to/the/data \
                --log_dir name_of_directory_where_keeps_logs \
                --pos_data rt-polaritydata/rt-polarity.pos \
                --neg_data rt-polaritydata/rt-polarity.neg \
                --feature_extract regex_tokenize \
~~~~

## Details about parameters  
Please refers to code, or try:
`python3 a1q3.py -h` or `python3 a1q3.py --help`

## Reference
1. Pang, Bo, and Lillian Lee. "Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales." Proceedings of the 43rd annual meeting on association for computational linguistics. Association for Computational Linguistics, 2005.  
2. Pennington, Jeffrey, Richard Socher, and Christopher Manning. "Glove: Global vectors for word representation." Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014.
