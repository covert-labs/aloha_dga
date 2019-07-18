# ALOHA DGA: Applying Auxiliary Loss Optimization for Hypothesis Augmentation (ALOHA) to DGA Domain Detection

This repo is based on Endgame's [dga_predict](https://github.com/endgameinc/dga_predict) code base, but it contains several [extensions](https://github.com/endgameinc/dga_predict/compare/master...covert-labs:master) for my research on trying out some of the ideas from [ALOHA: Auxiliary Loss Optimization for Hypothesis Augmentation](https://arxiv.org/pdf/1903.05700.pdf) applied to DGA classifiers.

This repo contains 4 classifers + 4 extensions of these that use Auxiliary Loss Optimization for Hypothesis Augmentation.

Baseline Models:
* Bigram - Endgame's Bigram model.
* LSTM - Endgame's LSTM model.
* CNN - CNN adapted from [snowman](https://github.com/keeganhines/snowman).
* LSTM + CNN - CNN adapted from [snowman](https://github.com/keeganhines/snowman), combined with LSTM as defined by [Deep Learning For Realtime Malware Detection (ShmooCon 2018)](https://www.youtube.com/watch?v=99hniQYB6VM)'s LSTM + CNN (see 13:17 for architecture) by Domenic Puzio and Kate Highnam.

ALOHA Extended Models (each simply use the malware family as additional labels)
* ALOHA CNN
* ALOHA Bigram
* ALOHA LSTM
* ALOHA LSTM + CNN

## Installation

```
conda create -n dga_predict python=2.7 scikit-learn keras tensorflow-gpu matplotlib
source activate dga_predict
pip install tldextract
```

## Running the code

```
python run.py
```

will download and generate all the data, train and evaluate the 8 classifiers, and save several PNGs to disk (the ROC curves at various zoom levels).

It defaults to 1 fold to speed things up.  This code will run on your local machine or on a machine with a GPU (GPU will of course
be much faster).

## DGA Algorithms

We have 11 DGA algorithms in our repo.  Some are from the https://github.com/baderj/domain_generation_algorithms
repo.  We noted these in each file and kept the same GNU license.  However, we made some small edits
such as allowing for no TLD and varying the size for some algorithms.

