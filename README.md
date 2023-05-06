# Word2vec Semantic Comparison between Subject Answers
Natural language processing (NLP) is a branch of Artificial Intelligence to help computers understand human languages (both text and spoken words) [1] Word representation techniques help NLP tasks to perform better by grouping together vectors of similar words [2]. Mikolov et al. [3] proposed the vector offset method to capture meaningful syntactic and semantic regularities [1, 2, 3, 4]. The Word2Vec, a word embedding technique in NLP, was introduced by Mikolov et al. [3].  Continuous skip-gram and continuous bag of words (CBOW) are two architectures of the word2vec model . The Skip-gram model was developed by Mikolov et al. [3] to learn high-quality distributed vector representation. This is useful for predicting the surrounding words in a sentence or a document.  However, the CBOW [3] is useful for predicting the current word based on the context. 

The value of Word2Vec is its ability to create vector representations of word semantics on which simple algebraic operations can be run. An example from Mikolov et al. [3] is that the “vector(”King”) - vector(”Man”) + vector(”Woman”) is very similar to the vector “Queen” [3]. In research, these word2vec models have been employed in study the relationship between words in documents and revealed biases based on gender and race, e.g. “Man is to computer programmer as woman is to homemaker?” [5], and racial biases “Black is to criminal as caucasian is to police” [6]. In this study, Word2Vec was employed in order to score the semantic similarities between the ground truth image labels (words for the target drawing) and subject predictions (the children’s responses). Word2Vec allows us to reduce the sources bias when manually scoring semantic similarities. It also creates a continuous scale which is valuable for regression analysis.

In order to score similarity between the ground truth labels and subject predictions (Italian), all answers were manually translated into english. We then filtered words by their part of speech to create a Bag of Words including only Nouns, Proper Nouns, Adjectives, and Verbs. Words were then embedded using a word2vec model trained using the following corpus: OntoNotes 5, ClearNLP Constituent-to-Dependency Conversion, WordNet 3.0 (using spaCy [7], an open-source library for NLP, source: https://spacy.io/models/en#en_core_web_lg). The ground truth labels consisting of individual words were also vectorized, then each word in the subject’s prediction was compared against the ground truth and the shortest distance was reported as the semantic distance of the subject’s prediction. Code to reproduce the exact analysis can be found in the supplementary resources. (Supplementary resources: https://github.com/A-Telfer/neuropsychology-study-semantic-similarity)


References
1. Khurana, D., Koli, A., Khatter, K. et al. Natural language processing, state of the art, current trends and challenges. Multimed Tools Appl 82, 3713–3744 (2023).
2. Tomas Mikolov and Ilya Sutskever and Kai Chen and Greg Corrado and Jeffrey Dean , Distributed Representations of Words and Phrases and their Compositionality, arXiv: 1310.4546 (2013)
3. Tomas Mikolov and Kai Chen and Greg Corrado and Jeffrey. Efficient Estimation of Word Representations in Vector Space. arXiv: 1301.3781 (2013)
4. Mikolov, W.T. Yih, G. Zweig. Linguistic Regularities in Continuous Space Word Representations. NAACL HLT (2013).
5. Bolukbasi, Tolga, et al. "Man is to computer programmer as woman is to homemaker? debiasing word embeddings." Advances in neural information processing systems 29 (2016).
6. Manzini, Thomas, et al. "Black is to criminal as caucasian is to police: Detecting and removing multiclass bias in word embeddings." arXiv preprint arXiv:1904.04047 (2019).
7. Jugran S, Kumar A, Tyagi BS, Anand V. Extractive automatic text summarization using SpaCy in Python & NLP. In: 2021 International conference on advance computing and innovative technologies in engineering (ICACITE); 2021. p. 582–5.




## Download Model


```python
pip install -r requirements.txt
```

    Requirement already satisfied: spacy==3.5.2 in /opt/anaconda3/lib/python3.9/site-packages (from -r requirements.txt (line 1)) (3.5.2)
    Requirement already satisfied: spacy-transformers==1.2.3 in /opt/anaconda3/lib/python3.9/site-packages (from -r requirements.txt (line 2)) (1.2.3)
    Requirement already satisfied: numpy==1.22.4 in /opt/anaconda3/lib/python3.9/site-packages (from -r requirements.txt (line 3)) (1.22.4)
    Requirement already satisfied: pandas==1.5.3 in /opt/anaconda3/lib/python3.9/site-packages (from -r requirements.txt (line 4)) (1.5.3)
    Requirement already satisfied: spacy-legacy<3.1.0,>=3.0.11 in /opt/anaconda3/lib/python3.9/site-packages (from spacy==3.5.2->-r requirements.txt (line 1)) (3.0.12)
    Requirement already satisfied: spacy-loggers<2.0.0,>=1.0.0 in /opt/anaconda3/lib/python3.9/site-packages (from spacy==3.5.2->-r requirements.txt (line 1)) (1.0.4)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/anaconda3/lib/python3.9/site-packages (from spacy==3.5.2->-r requirements.txt (line 1)) (1.0.9)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/anaconda3/lib/python3.9/site-packages (from spacy==3.5.2->-r requirements.txt (line 1)) (2.0.7)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/anaconda3/lib/python3.9/site-packages (from spacy==3.5.2->-r requirements.txt (line 1)) (3.0.8)
    Requirement already satisfied: thinc<8.2.0,>=8.1.8 in /opt/anaconda3/lib/python3.9/site-packages (from spacy==3.5.2->-r requirements.txt (line 1)) (8.1.10)
    Requirement already satisfied: wasabi<1.2.0,>=0.9.1 in /opt/anaconda3/lib/python3.9/site-packages (from spacy==3.5.2->-r requirements.txt (line 1)) (1.1.1)
    Requirement already satisfied: srsly<3.0.0,>=2.4.3 in /opt/anaconda3/lib/python3.9/site-packages (from spacy==3.5.2->-r requirements.txt (line 1)) (2.4.6)
    Requirement already satisfied: catalogue<2.1.0,>=2.0.6 in /opt/anaconda3/lib/python3.9/site-packages (from spacy==3.5.2->-r requirements.txt (line 1)) (2.0.8)
    Requirement already satisfied: typer<0.8.0,>=0.3.0 in /opt/anaconda3/lib/python3.9/site-packages (from spacy==3.5.2->-r requirements.txt (line 1)) (0.7.0)
    Requirement already satisfied: pathy>=0.10.0 in /opt/anaconda3/lib/python3.9/site-packages (from spacy==3.5.2->-r requirements.txt (line 1)) (0.10.1)
    Requirement already satisfied: smart-open<7.0.0,>=5.2.1 in /opt/anaconda3/lib/python3.9/site-packages (from spacy==3.5.2->-r requirements.txt (line 1)) (5.2.1)
    Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/anaconda3/lib/python3.9/site-packages (from spacy==3.5.2->-r requirements.txt (line 1)) (4.64.1)
    Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/anaconda3/lib/python3.9/site-packages (from spacy==3.5.2->-r requirements.txt (line 1)) (2.28.1)
    Requirement already satisfied: pydantic!=1.8,!=1.8.1,<1.11.0,>=1.7.4 in /opt/anaconda3/lib/python3.9/site-packages (from spacy==3.5.2->-r requirements.txt (line 1)) (1.10.7)
    Requirement already satisfied: jinja2 in /opt/anaconda3/lib/python3.9/site-packages (from spacy==3.5.2->-r requirements.txt (line 1)) (2.11.3)
    Requirement already satisfied: setuptools in /opt/anaconda3/lib/python3.9/site-packages (from spacy==3.5.2->-r requirements.txt (line 1)) (63.4.1)
    Requirement already satisfied: packaging>=20.0 in /opt/anaconda3/lib/python3.9/site-packages (from spacy==3.5.2->-r requirements.txt (line 1)) (21.3)
    Requirement already satisfied: langcodes<4.0.0,>=3.2.0 in /opt/anaconda3/lib/python3.9/site-packages (from spacy==3.5.2->-r requirements.txt (line 1)) (3.3.0)
    Requirement already satisfied: transformers<4.29.0,>=3.4.0 in /opt/anaconda3/lib/python3.9/site-packages (from spacy-transformers==1.2.3->-r requirements.txt (line 2)) (4.28.1)
    Requirement already satisfied: torch>=1.8.0 in /opt/anaconda3/lib/python3.9/site-packages (from spacy-transformers==1.2.3->-r requirements.txt (line 2)) (2.0.0)
    Requirement already satisfied: spacy-alignments<1.0.0,>=0.7.2 in /opt/anaconda3/lib/python3.9/site-packages (from spacy-transformers==1.2.3->-r requirements.txt (line 2)) (0.9.0)
    Requirement already satisfied: python-dateutil>=2.8.1 in /opt/anaconda3/lib/python3.9/site-packages (from pandas==1.5.3->-r requirements.txt (line 4)) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/lib/python3.9/site-packages (from pandas==1.5.3->-r requirements.txt (line 4)) (2022.1)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /opt/anaconda3/lib/python3.9/site-packages (from packaging>=20.0->spacy==3.5.2->-r requirements.txt (line 1)) (3.0.9)
    Requirement already satisfied: typing-extensions>=4.2.0 in /opt/anaconda3/lib/python3.9/site-packages (from pydantic!=1.8,!=1.8.1,<1.11.0,>=1.7.4->spacy==3.5.2->-r requirements.txt (line 1)) (4.3.0)
    Requirement already satisfied: six>=1.5 in /opt/anaconda3/lib/python3.9/site-packages (from python-dateutil>=2.8.1->pandas==1.5.3->-r requirements.txt (line 4)) (1.16.0)
    Requirement already satisfied: charset-normalizer<3,>=2 in /opt/anaconda3/lib/python3.9/site-packages (from requests<3.0.0,>=2.13.0->spacy==3.5.2->-r requirements.txt (line 1)) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in /opt/anaconda3/lib/python3.9/site-packages (from requests<3.0.0,>=2.13.0->spacy==3.5.2->-r requirements.txt (line 1)) (3.3)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /opt/anaconda3/lib/python3.9/site-packages (from requests<3.0.0,>=2.13.0->spacy==3.5.2->-r requirements.txt (line 1)) (1.26.11)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/anaconda3/lib/python3.9/site-packages (from requests<3.0.0,>=2.13.0->spacy==3.5.2->-r requirements.txt (line 1)) (2022.9.24)
    Requirement already satisfied: blis<0.8.0,>=0.7.8 in /opt/anaconda3/lib/python3.9/site-packages (from thinc<8.2.0,>=8.1.8->spacy==3.5.2->-r requirements.txt (line 1)) (0.7.9)
    Requirement already satisfied: confection<1.0.0,>=0.0.1 in /opt/anaconda3/lib/python3.9/site-packages (from thinc<8.2.0,>=8.1.8->spacy==3.5.2->-r requirements.txt (line 1)) (0.0.4)
    Requirement already satisfied: filelock in /opt/anaconda3/lib/python3.9/site-packages (from torch>=1.8.0->spacy-transformers==1.2.3->-r requirements.txt (line 2)) (3.12.0)
    Requirement already satisfied: sympy in /opt/anaconda3/lib/python3.9/site-packages (from torch>=1.8.0->spacy-transformers==1.2.3->-r requirements.txt (line 2)) (1.10.1)
    Requirement already satisfied: networkx in /opt/anaconda3/lib/python3.9/site-packages (from torch>=1.8.0->spacy-transformers==1.2.3->-r requirements.txt (line 2)) (2.8.4)
    Requirement already satisfied: huggingface-hub<1.0,>=0.11.0 in /opt/anaconda3/lib/python3.9/site-packages (from transformers<4.29.0,>=3.4.0->spacy-transformers==1.2.3->-r requirements.txt (line 2)) (0.14.1)
    Requirement already satisfied: pyyaml>=5.1 in /opt/anaconda3/lib/python3.9/site-packages (from transformers<4.29.0,>=3.4.0->spacy-transformers==1.2.3->-r requirements.txt (line 2)) (5.4.1)
    Requirement already satisfied: regex!=2019.12.17 in /opt/anaconda3/lib/python3.9/site-packages (from transformers<4.29.0,>=3.4.0->spacy-transformers==1.2.3->-r requirements.txt (line 2)) (2022.7.9)
    Requirement already satisfied: tokenizers!=0.11.3,<0.14,>=0.11.1 in /opt/anaconda3/lib/python3.9/site-packages (from transformers<4.29.0,>=3.4.0->spacy-transformers==1.2.3->-r requirements.txt (line 2)) (0.13.3)
    Requirement already satisfied: click<9.0.0,>=7.1.1 in /opt/anaconda3/lib/python3.9/site-packages (from typer<0.8.0,>=0.3.0->spacy==3.5.2->-r requirements.txt (line 1)) (8.0.4)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/anaconda3/lib/python3.9/site-packages (from jinja2->spacy==3.5.2->-r requirements.txt (line 1)) (2.0.1)
    Requirement already satisfied: fsspec in /opt/anaconda3/lib/python3.9/site-packages (from huggingface-hub<1.0,>=0.11.0->transformers<4.29.0,>=3.4.0->spacy-transformers==1.2.3->-r requirements.txt (line 2)) (2022.7.1)
    Requirement already satisfied: mpmath>=0.19 in /opt/anaconda3/lib/python3.9/site-packages (from sympy->torch>=1.8.0->spacy-transformers==1.2.3->-r requirements.txt (line 2)) (1.2.1)
    Note: you may need to restart the kernel to use updated packages.



```python
import locale
locale.getpreferredencoding = lambda: "UTF-8"
! python -m spacy download en_core_web_lg
```

    Collecting en-core-web-lg==3.5.0
      Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.5.0/en_core_web_lg-3.5.0-py3-none-any.whl (587.7 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m587.7/587.7 MB[0m [31m1.9 MB/s[0m eta [36m0:00:00[0m00:01[0m00:04[0m
    [?25hRequirement already satisfied: spacy<3.6.0,>=3.5.0 in /opt/anaconda3/lib/python3.9/site-packages (from en-core-web-lg==3.5.0) (3.5.2)
    Requirement already satisfied: spacy-legacy<3.1.0,>=3.0.11 in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (3.0.12)
    Requirement already satisfied: spacy-loggers<2.0.0,>=1.0.0 in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (1.0.4)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (1.0.9)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (2.0.7)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (3.0.8)
    Requirement already satisfied: thinc<8.2.0,>=8.1.8 in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (8.1.10)
    Requirement already satisfied: wasabi<1.2.0,>=0.9.1 in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (1.1.1)
    Requirement already satisfied: srsly<3.0.0,>=2.4.3 in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (2.4.6)
    Requirement already satisfied: catalogue<2.1.0,>=2.0.6 in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (2.0.8)
    Requirement already satisfied: typer<0.8.0,>=0.3.0 in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (0.7.0)
    Requirement already satisfied: pathy>=0.10.0 in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (0.10.1)
    Requirement already satisfied: smart-open<7.0.0,>=5.2.1 in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (5.2.1)
    Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (4.64.1)
    Requirement already satisfied: numpy>=1.15.0 in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (1.22.4)
    Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (2.28.1)
    Requirement already satisfied: pydantic!=1.8,!=1.8.1,<1.11.0,>=1.7.4 in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (1.10.7)
    Requirement already satisfied: jinja2 in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (2.11.3)
    Requirement already satisfied: setuptools in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (63.4.1)
    Requirement already satisfied: packaging>=20.0 in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (21.3)
    Requirement already satisfied: langcodes<4.0.0,>=3.2.0 in /opt/anaconda3/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (3.3.0)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /opt/anaconda3/lib/python3.9/site-packages (from packaging>=20.0->spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (3.0.9)
    Requirement already satisfied: typing-extensions>=4.2.0 in /opt/anaconda3/lib/python3.9/site-packages (from pydantic!=1.8,!=1.8.1,<1.11.0,>=1.7.4->spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (4.3.0)
    Requirement already satisfied: charset-normalizer<3,>=2 in /opt/anaconda3/lib/python3.9/site-packages (from requests<3.0.0,>=2.13.0->spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in /opt/anaconda3/lib/python3.9/site-packages (from requests<3.0.0,>=2.13.0->spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (3.3)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /opt/anaconda3/lib/python3.9/site-packages (from requests<3.0.0,>=2.13.0->spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (1.26.11)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/anaconda3/lib/python3.9/site-packages (from requests<3.0.0,>=2.13.0->spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (2022.9.24)
    Requirement already satisfied: blis<0.8.0,>=0.7.8 in /opt/anaconda3/lib/python3.9/site-packages (from thinc<8.2.0,>=8.1.8->spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (0.7.9)
    Requirement already satisfied: confection<1.0.0,>=0.0.1 in /opt/anaconda3/lib/python3.9/site-packages (from thinc<8.2.0,>=8.1.8->spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (0.0.4)
    Requirement already satisfied: click<9.0.0,>=7.1.1 in /opt/anaconda3/lib/python3.9/site-packages (from typer<0.8.0,>=0.3.0->spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (8.0.4)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/anaconda3/lib/python3.9/site-packages (from jinja2->spacy<3.6.0,>=3.5.0->en-core-web-lg==3.5.0) (2.0.1)
    [38;5;2m✔ Download and installation successful[0m
    You can now load the package via spacy.load('en_core_web_lg')



```python
import pandas as pd
import numpy as np
import spacy
```

## Data Cleaning




```python
df = pd.read_excel("Rating Scale - English.xlsx", skiprows=3)
df = df.loc[:129]
df.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Group</th>
      <th>Subject</th>
      <th>Stimuli</th>
      <th>Unnamed: 3</th>
      <th>Answer given (expl 1)</th>
      <th>Answer given 2 (expl 2)</th>
      <th>expl.1</th>
      <th>Unnamed: 7</th>
      <th>expl.2</th>
      <th>Unnamed: 9</th>
      <th>Unnamed: 10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>127</th>
      <td>B</td>
      <td>13</td>
      <td>8</td>
      <td>key</td>
      <td>racket</td>
      <td>racket</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>128</th>
      <td>B</td>
      <td>13</td>
      <td>9</td>
      <td>lamp</td>
      <td>home/house</td>
      <td>home/house</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>129</th>
      <td>B</td>
      <td>13</td>
      <td>10</td>
      <td>leaf</td>
      <td>racket</td>
      <td>racket</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Rename Columns



```python
df = df.rename(columns={
    'Group': 'group',
    'Subject': 'subject',
    'Stimuli': 'stimuli',
    'Unnamed: 3': 'ground_truth',
    'Answer given (expl 1)': 'prediction1',
    'Answer given 2 (expl 2)': 'prediction2',
    'expl.1 ': 'manual_similarity_score1',
    'expl.2': 'manual_similarity_score2'
})
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group</th>
      <th>subject</th>
      <th>stimuli</th>
      <th>ground_truth</th>
      <th>prediction1</th>
      <th>prediction2</th>
      <th>manual_similarity_score1</th>
      <th>Unnamed: 7</th>
      <th>manual_similarity_score2</th>
      <th>Unnamed: 9</th>
      <th>Unnamed: 10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>VI</td>
      <td>1</td>
      <td>1</td>
      <td>face</td>
      <td>face</td>
      <td>face</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>VI</td>
      <td>1</td>
      <td>4</td>
      <td>person/figure</td>
      <td>little person</td>
      <td>little child</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df[df.columns.drop(['Unnamed: 7', 'Unnamed: 9', 'Unnamed: 10'])]
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group</th>
      <th>subject</th>
      <th>stimuli</th>
      <th>ground_truth</th>
      <th>prediction1</th>
      <th>prediction2</th>
      <th>manual_similarity_score1</th>
      <th>manual_similarity_score2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>VI</td>
      <td>1</td>
      <td>1</td>
      <td>face</td>
      <td>face</td>
      <td>face</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>VI</td>
      <td>1</td>
      <td>4</td>
      <td>person/figure</td>
      <td>little person</td>
      <td>little child</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



### Correct Values


```python
for name, _ in df.groupby(["stimuli", "ground_truth"]):
  print(name)
```

    (1, 'face')
    (2, 'bottle')
    (3, 'cup')
    (3, 'cup/mug')
    (4, 'person')
    (4, 'person/figure')
    (5, 'telephone')
    (6, 'umbrella')
    (7, 'scissors')
    (7, 'scissors ')
    (8, 'key')
    (9, 'lamp')
    (10, 'leaf ')
    (11, 'apple')
    (12, 'shoe')
    (13, 'crutch/cane')
    (15, 'flower')
    (16, 'hand')



```python
df.loc[df.stimuli==2, 'ground_truth'] = 'bottle'
df.loc[df.stimuli==3, 'ground_truth'] = 'cup'
df.loc[df.stimuli==4, 'ground_truth'] = 'person'
df.loc[df.stimuli==7, 'ground_truth'] = 'scissors'

for name, _ in df.groupby(["stimuli", "ground_truth"]):
  print(name)
```

    (1, 'face')
    (2, 'bottle')
    (3, 'cup')
    (4, 'person')
    (5, 'telephone')
    (6, 'umbrella')
    (7, 'scissors')
    (8, 'key')
    (9, 'lamp')
    (10, 'leaf ')
    (11, 'apple')
    (12, 'shoe')
    (13, 'crutch/cane')
    (15, 'flower')
    (16, 'hand')



```python
df.ground_truth.unique()
```




    array(['face', 'person', 'umbrella', 'key', 'lamp', 'shoe', 'crutch/cane',
           'bottle', 'leaf ', 'cup', 'telephone', 'scissors', 'apple',
           'flower', 'hand'], dtype=object)




```python
sorted(df.subject.unique())
```




    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]




```python
df.group.unique()
```




    array(['VI', 'B', nan], dtype=object)




```python
df[df.group.isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group</th>
      <th>subject</th>
      <th>stimuli</th>
      <th>ground_truth</th>
      <th>prediction1</th>
      <th>prediction2</th>
      <th>manual_similarity_score1</th>
      <th>manual_similarity_score2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>80</th>
      <td>NaN</td>
      <td>7</td>
      <td>1</td>
      <td>face</td>
      <td>portrait of a person</td>
      <td>portrait</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.fillna('')
```

## POS Tagging and Filtering


```python
# nlp = spacy.load("en_core_web_trf")
nlp = spacy.load("en_core_web_lg")

# Extract nouns and adjectives
pos_tags = ["NOUN", "ADJ", "VERB", "PROPN"]
df["tokenized_ground_truth"] = df.ground_truth.apply(nlp)
df["tokenized_ground_truth"] = df.tokenized_ground_truth.apply(
    lambda x: [w for w in x if w.pos_ in pos_tags])

df["tokenized_prediction1"] = df.prediction1.apply(nlp)
df["tokenized_prediction1"] = df.tokenized_prediction1.apply(
    lambda x: [w for w in x if w.pos_ in pos_tags])

df["tokenized_prediction2"] = df.prediction2.apply(nlp)
df["tokenized_prediction2"] = df.tokenized_prediction2.apply(
    lambda x: [w for w in x if w.pos_ in pos_tags])

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group</th>
      <th>subject</th>
      <th>stimuli</th>
      <th>ground_truth</th>
      <th>prediction1</th>
      <th>prediction2</th>
      <th>manual_similarity_score1</th>
      <th>manual_similarity_score2</th>
      <th>tokenized_ground_truth</th>
      <th>tokenized_prediction1</th>
      <th>tokenized_prediction2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>VI</td>
      <td>1</td>
      <td>1</td>
      <td>face</td>
      <td>face</td>
      <td>face</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>[face]</td>
      <td>[face]</td>
      <td>[face]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>VI</td>
      <td>1</td>
      <td>4</td>
      <td>person</td>
      <td>little person</td>
      <td>little child</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>[person]</td>
      <td>[little, person]</td>
      <td>[little, child]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>VI</td>
      <td>1</td>
      <td>6</td>
      <td>umbrella</td>
      <td>handle and some kind of bend/curve</td>
      <td>umbrella</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>[umbrella]</td>
      <td>[handle, kind, bend, curve]</td>
      <td>[umbrella]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VI</td>
      <td>1</td>
      <td>8</td>
      <td>key</td>
      <td>I don't know</td>
      <td>key</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>[key]</td>
      <td>[know]</td>
      <td>[key]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VI</td>
      <td>1</td>
      <td>9</td>
      <td>lamp</td>
      <td>I don't know</td>
      <td>container</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>[lamp]</td>
      <td>[know]</td>
      <td>[container]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>125</th>
      <td>B</td>
      <td>13</td>
      <td>6</td>
      <td>umbrella</td>
      <td>umbrella</td>
      <td>umbrella</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>[umbrella]</td>
      <td>[umbrella]</td>
      <td>[umbrella]</td>
    </tr>
    <tr>
      <th>126</th>
      <td>B</td>
      <td>13</td>
      <td>7</td>
      <td>scissors</td>
      <td>tree</td>
      <td>tree</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>[scissors]</td>
      <td>[tree]</td>
      <td>[tree]</td>
    </tr>
    <tr>
      <th>127</th>
      <td>B</td>
      <td>13</td>
      <td>8</td>
      <td>key</td>
      <td>racket</td>
      <td>racket</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>[key]</td>
      <td>[racket]</td>
      <td>[racket]</td>
    </tr>
    <tr>
      <th>128</th>
      <td>B</td>
      <td>13</td>
      <td>9</td>
      <td>lamp</td>
      <td>home/house</td>
      <td>home/house</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>[lamp]</td>
      <td>[home, house]</td>
      <td>[home, house]</td>
    </tr>
    <tr>
      <th>129</th>
      <td>B</td>
      <td>13</td>
      <td>10</td>
      <td>leaf</td>
      <td>racket</td>
      <td>racket</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>[leaf]</td>
      <td>[racket]</td>
      <td>[racket]</td>
    </tr>
  </tbody>
</table>
<p>130 rows × 11 columns</p>
</div>



## Vector Distances


```python
def calculate_closest_distance_vectors(s1, s2):
  nearest = -1
  for w1 in s1:
    v1 = w1.vector
    for w2 in s2:
      v2 = w2.vector
      d = np.linalg.norm(v1-v2, ord=2)
      if nearest == -1 or d < nearest:
        nearest = d
  
  return nearest

for idx, row in df.iterrows():
  df.loc[idx, 'distance_predication1'] = calculate_closest_distance_vectors(
      row.tokenized_ground_truth, row.tokenized_prediction1)
  
  df.loc[idx, 'distance_predication2'] = calculate_closest_distance_vectors(
      row.tokenized_ground_truth, row.tokenized_prediction2)

```


```python
df[['group', 'subject', 'stimuli', 'ground_truth', 'prediction1', 'distance_predication1', 'prediction2', 'distance_predication2']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group</th>
      <th>subject</th>
      <th>stimuli</th>
      <th>ground_truth</th>
      <th>prediction1</th>
      <th>distance_predication1</th>
      <th>prediction2</th>
      <th>distance_predication2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>VI</td>
      <td>1</td>
      <td>1</td>
      <td>face</td>
      <td>face</td>
      <td>0.000000</td>
      <td>face</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>VI</td>
      <td>1</td>
      <td>4</td>
      <td>person</td>
      <td>little person</td>
      <td>0.000000</td>
      <td>little child</td>
      <td>47.189495</td>
    </tr>
    <tr>
      <th>2</th>
      <td>VI</td>
      <td>1</td>
      <td>6</td>
      <td>umbrella</td>
      <td>handle and some kind of bend/curve</td>
      <td>44.875175</td>
      <td>umbrella</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VI</td>
      <td>1</td>
      <td>8</td>
      <td>key</td>
      <td>I don't know</td>
      <td>88.529228</td>
      <td>key</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VI</td>
      <td>1</td>
      <td>9</td>
      <td>lamp</td>
      <td>I don't know</td>
      <td>70.088333</td>
      <td>container</td>
      <td>55.279503</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>125</th>
      <td>B</td>
      <td>13</td>
      <td>6</td>
      <td>umbrella</td>
      <td>umbrella</td>
      <td>0.000000</td>
      <td>umbrella</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>126</th>
      <td>B</td>
      <td>13</td>
      <td>7</td>
      <td>scissors</td>
      <td>tree</td>
      <td>64.222961</td>
      <td>tree</td>
      <td>64.222961</td>
    </tr>
    <tr>
      <th>127</th>
      <td>B</td>
      <td>13</td>
      <td>8</td>
      <td>key</td>
      <td>racket</td>
      <td>82.567841</td>
      <td>racket</td>
      <td>82.567841</td>
    </tr>
    <tr>
      <th>128</th>
      <td>B</td>
      <td>13</td>
      <td>9</td>
      <td>lamp</td>
      <td>home/house</td>
      <td>64.835320</td>
      <td>home/house</td>
      <td>64.835320</td>
    </tr>
    <tr>
      <th>129</th>
      <td>B</td>
      <td>13</td>
      <td>10</td>
      <td>leaf</td>
      <td>racket</td>
      <td>59.010170</td>
      <td>racket</td>
      <td>59.010170</td>
    </tr>
  </tbody>
</table>
<p>130 rows × 8 columns</p>
</div>


