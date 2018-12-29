# Toxic-Comment-Classifier
## -- In Progress --

This project is a development space for the Kaggle competition to classify questions as sincere or insincere. More
information on the competition can be found [(Here)](https://www.kaggle.com/c/quora-insincere-questions-classification).
This problem will be tackled using both a simple Naive Bayes approach, as well as by multiple deep learning approaches.

Because the competition require that jupyter notebooks be submitted for each solution, each notebook must run from start
to finish and produce a prediction csv. All notebooks are found in the src/ directory. 

## Getting Started

Following these instructions will get you a copy of the project up and running on your local machine for development
and testing purposes.

### Prerequisites

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

### Naive Bayes

The first attempt to tackle this problem uses a Naive Bayes approach. Naive Bayes relies on the assumption that all
word occurrences are independent of one another. This is quite a strong assumption. Therefore, Naive Bayes is often
good for a rough model, but not complex enough to handle real world situations. For text classification specifically,
Naive Bayes loses all notions of context in the text.

Using a Naive Bayes approach, I achieved a training accuracy of 0.926. While this result seems quite good, it's actually
a mediocre accuracy. The vast majority of data samples are labeled as sincere questions. Therefore, a model that
makes predictions of all '0's will fare quite well. Once submitted to the Kaggle competition, this Naive Bayes
model fared in the lowest 20% of all submissions. A stronger model than a simple Naive Bayes model is required
to tackle this problem.

### Simple LSTM

The next notebook uses GloVe word embeddings to preprocess the raw text and then uses a bidirectional LSTM Neural
Network architecture to make predictions. This results in a significantly improved accuracy. The F1-score obtained
from this approach is 0.647.

More information on F1-score can be found [here](https://en.wikipedia.org/wiki/F1_score)

## Built With

* [Pandas](https://pandas.pydata.org/) - Data preparation
* [scikit-learn](https://scikit-learn.org/) - Machine Learning Library
* [Keras](https://keras.io/) - Deep Learning API
* [Tensorflow](https://www.tensorflow.org/) - Deep Learning Backend
* [Glove](https://nlp.stanford.edu/projects/glove/) - Pretrained word embeddings
* [NLTK](https://www.nltk.org/) - Natural Language ToolKit
* [Seaborn](https://seaborn.pydata.org/) - Statistical data visualization
* [WordCloud](https://amueller.github.io/word_cloud/) - WordCloud generator

## Authors

* **Brian Midei** - [bmmidei](https://github.com/bmmidei)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Kaggle competition for posing the original problem and providing data - https://www.kaggle.com/c/quora-insincere-questions-classification
