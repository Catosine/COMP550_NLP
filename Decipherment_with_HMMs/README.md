# Decipherment with HMMs
This is directory is used for question 3 in assignment 2.

## What does this code do?
This code trains a hidden Markov model to decrypt three kinds of ciphers:
1. The first cipher is quite archaic, first used by Julius Caesar. Each letter in the plain text is shifted
by a fixed amount to produce the cipher text.
2. The second cipher is a more complex cipher, in which there are two letter substitution ciphers. When
encrypting each letter, one of the two is randomly chosen, and that cipher is used to generate the
ciphertext letter.
3. The third cipher was invented by Gilbert Vernam. The cipher text is produced by adding the
integer values of the a key and the plain text. We also know that the key is produced by shifting
characters in plain text by 3 places from right to left. For example if you need to encrypt the
plain text `nlp is awesome.` the key you will use is `is awesome.nlp`. To generate the cipher,
you need to add the integer values of the two strings character by character. Integer values for
characters from a-z are 0-25 and for `{blank}`, `,`, `.` are `26`, `27` and `28` respectively. Thus the cipher
text will be "ktexilbshqwnzpo".

## Required Package
1. nltk
2. sklearn
3. tqdm

## Required Dataset
1. Dataset provided by professor
2. Penn Treebank (Accessed by `nltk.corpus.treebank`)

## How to run the code?  
~~~~
# Suppose the positive dat has been kept at 
# absolute/path/to/the/data/

# To train a HMM for cipher1
python3 decipher.py --data_dir absolute/path/to/the/data \
                    cipher1

# To train a HMM for cipher2 using external corpus
python3 decipher.py --data_dir absolute/path/to/the/data \
                    -lm \
                    cipher2
                    
# To train a HMM for cipher3 using Laplace smoothing
python3 decipher.py --data_dir absolute/path/to/the/data \
                    -laplace \
                    cipher3
                    
# To train a HMM for cipher3 using external corpus and Laplace smoothing
python3 decipher.py --data_dir absolute/path/to/the/data \
                    -lm \
                    -laplace \
                    cipher3
~~~~

## Details about parameters  
Please refers to code, or try:
`python3 a1q3.py -h` or `python3 a1q3.py --help`

## Reference
1. Marcus, Mitchell, et al. "The Penn Treebank: annotating predicate argument structure." Proceedings of the workshop on Human Language Technology. Association for Computational Linguistics, 1994.
