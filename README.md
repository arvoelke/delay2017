### Requirements

 - Python 2.7.3
 - nengolib ([pre-release](https://github.com/arvoelke/nengolib/tree/aee92b8fc45749f07f663fe696745cf0a33bfa17))
 - nengo==2.3.1.dev0
 - scipy==0.18.0rc1
 - numpy==1.11.3
 - seaborn==0.7.1
 - matplotlib==2.0.0
 - hyperopt>=0.0.2 ([dependencies](https://github.com/hyperopt/hyperopt/pull/246/files) include: nose, pymongo, and networkx)
 - doit>=0.29.0

### Instructions

 - Install all requirements.
 - Run `doit` to generate all of the files required by LaTeX.
 - Then compile the LaTeX paper!

### Source Code

All of the optimization/simulation/analysis code is in `main.py` (and executed by `dodo.py`).
