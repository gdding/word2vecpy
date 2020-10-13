A Python implementation of the Continuous Bag of Words (CBOW) and skip-gram neural network architectures, and the hierarchical softmax and negative sampling learning algorithms for efficient learning of word vectors (Mikolov, et al., 2013a, b, c; http://code.google.com/p/word2vec/).

Usage
-----
To train word vectors:
```
word2vec_train.py [-h] -train FI -model FO [-cbow CBOW] [-negative NEG]
                  [-dim DIM] [-alpha ALPHA] [-window WIN]
                  [-min-count MIN_COUNT] [-processes NUM_PROCESSES]
                  [-binary BINARY]
                  
required arguments:
  -train FI                 Training file
  -model FO                 Output file to save the model

optional arguments:
  -h, --help                show this help message and exit
  -cbow CBOW                1 for CBOW, 0 for skip-gram, default=1
  -negative NEG             Number of negative examples (>0) for negative sampling, 
                            0 for hierarchical softmax, default=5
  -dim DIM                  Dimensionality of word embeddings, default=100
  -alpha ALPHA              Starting learning rate, default=0.025
  -window WIN               Max window length, default=5
  -min-count MIN_COUNT      Min count for words used to learn <unk>, default=5
  -processes NUM_PROCESSES  Number of processes, default=1
  -binary BINARY            1 for output model in binary format, 0 otherwise, default=0
  -epoch EPOCH              Number of traing epochs, default=1
  
for example:
  python word2vec_train.py -train train_corpus.txt -model model-cbow-5-200-3.txt -window 5 -processes 8 -dim 200 -epoch 3
```
Each sentence in the training file is expected to be newline separated. 

To test word vectors:
```
word2vec_test.py [-h] -model FI -et ET -test TEST

required arguments:
  -model FI                 model file
  -et ET                    evaluation type, 0 for wordsim353, 1 for Google analogy
  -test TEST                test file to be evaluated, when et is 0, must be wordsim353 file, otherwise analogy file.

for example:
  python word2vec_test.py -model model-cbow-5-200-3.txt -et 0 -test test/test_wordsim-353.txt
  python word2vec_test.py -model model-cbow-5-200-3.txt -et 1 -test test/test_analogy.txt
```

Evaluation Results
----------------------
```
1. Download training and testing data: https://cloud.tsinghua.edu.cn/f/18ae0a962c4042889044/?dl=1
2. Unzip the downloaded data to word2vec_data
3. Train word2vec model using word2vec_train.py as following:
    $python word2vec_train.py -train train_corpus.txt -model model-cbow-5-200-5.txt -window 5 -processes 4 -dim 200 -epoch 5
4. After the training finished, use the following command to evaluate the model:
(1) $python word2vec_test.py -model model-cbow-5-200-3.txt -et 0 -test word2vec_data/test_wordsim-353.txt
    The evaluation result for wordsim-353 is 0.6459
(2) $python word2vec_test.py -model model-cbow-5-2-00-3.txt -et 1 -test word2vec_data/test_analogy.txt
    The evaluation result for Google analogy is 0.7138
```

Implementation Details
----------------------
Written in Python 2.7.17, NumPy 1.16.6, and Pandas 0.24.2.

References
----------
Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013a). Distributed representations of words and phrases and their compositionality. Advances in Neural Information Processing Systems. http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013b). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781. http://arxiv.org/pdf/1301.3781.pdf

Mikolov, T., Yih, W., & Zweig, G. (2013c). Linguistic Regularities in Continuous Space Word Representations. HLT-NAACL. http://msr-waypoint.com/en-us/um/people/gzweig/Pubs/NAACL2013Regularities.pdf
