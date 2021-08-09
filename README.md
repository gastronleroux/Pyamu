# Pyamu

Compilation of university tasks and projects done in python.

## Installation and Setup Instructions

1. Clone the repository.
2. Download latest python if you haven't already.
3. Pip install required packages.
4. Import pyamu in a separate file and use its functions.

Required packages:

```
pip install -U numpy
pip install -U pandas
pip install -U pyspellchecker
pip install -U scikit-learn
pip install -U spacy
spacy download fr_core_news_lg
```
For testing: `pip install -U pytest`

## Tests
To test pyamu simply write `pytest` in console when inside the directory containing pyamu.py and test_pyamu.py.  
You can also test selected parts of pyamu, e.g. `pytest -k NaturalLanguageProcessing` will only test Natural Language Processing functions.

## Functions
Detailed descriptions of the functions can be found in the code comments.

### Natural Language Processing
* frLemmatize
* spamClassifier

### Neural Networks
* displayGates
* displayGradientMethod
* displayBackPropagation
* displayBoltzmannMachine

### Combinatorial Algorithms
* lexicalOrder
* bijectionSubsets
* lexicalPosition
* lexicalSubsetFromPosition
* lexicalOrderRecursive
* grayOrder
* grayRank
* graySubsetFromRank
* kSubsetSuccessor
* kSubsetRank
* kSubsetFromRank
* subsetDivision
* rgfFunction
* rgfDivision
* rgfGenerate
* permutationSuccessor
* permutationRank
* permutationFromRank
* numDivision
* conjugateDivision
* allDivisions
* prufer
* fromPrufer
* pruferRank
* pruferFromRank
* prim
