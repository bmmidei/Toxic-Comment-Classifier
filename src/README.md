# Getting Started

Following these instructions will get you a copy of the project up and running on your local machine for development
and testing purposes.

## Prerequisites

Running the code in this repository requires elementary knowledge of both Jupyter and Anaconda. It is recommended that 
new users create a new virtual environment with Anaconda to ensure that package dependencies match the developer 
versions. If you are unfamiliar with Anaconda, you can find more information and getting started tutorials here:
https://conda.io/docs/user-guide/overview.html

Begin by creating a directory on your local machine and cloning this repository using the ```git clone``` command.
Within the top level directory, you will find a 'req.txt' file, which includes a comprehensive list of dependencies
 necessary to execute the functionality of this repository. Use the following command to create a new conda environment
and install the necessary dependencies.
```
conda create -n new_env_name --file req.txt
```

Note that python version 3.6.7 was used for this project. To create a new Anaconda environment, you may use the terminal
command:
```
conda create -n name_of_myenv python=3.6.7
```

## Dataset

The dataset may be downloaded directly from the kaggle competition page found [here](https://www.kaggle.com/c/quora-insincere-questions-classification/data)
The train.csv and test.csv files should be placed in a directory 'inputs/' to follow the same folder structure as mine. 

## GloVe Word Embeddings

All of the neural network notebooks make use of GloVe word embeddings. These embeddings are posted on the Kaggle competition
page, but may also be downloaded directly from the [GloVe site](https://nlp.stanford.edu/projects/glove/). The embedding used
is the 'glove.840B.300d' embedding and should be downloaded to the directory 'inputs/embeddings/'.
