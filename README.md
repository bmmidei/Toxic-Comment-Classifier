# Toxic-Comment-Classifier

This project is an development space for the Kaggle competition to classify questions as sincere or insincere. More
information on the competition can be found [(Here)](https://www.kaggle.com/c/quora-insincere-questions-classification).
 This problem will be tackled both using a simple Naive Bayes approach, as well as by a deep learning approach.


## Getting Started

Following these instructions will get you a copy of the project up and running on your local machine for development
and testing purposes.

### Prerequisites

Running the code in this repository requires elementary knowledge of both Jupyter and Anaconda. It is recommended that 
new users create a new virtual environment with Anaconda to ensure that package dependencies match the developer 
versions. If you are unfamiliar with Anaconda, you can find more information and getting started tutorials here:
https://conda.io/docs/user-guide/overview.html

Note that python version 3.6.7 was used for this project. To create a new Anaconda environment, you may use the terminal
command:
```
conda create -n name_of_myenv python=3.6.7
```
After creating this environment, you may clone this repository to your local machine. Within the top level directory,
you will find a 'req.txt' file, which includes a comprehensive list of dependencies necessary to execute the functionality
of this repository. With your new environment active, use the following command to install these dependencies:
```
pip install -r /path/to/req.txt
```

## Built With

* [Pandas](https://pandas.pydata.org/) - Data preparation

## Authors

* **Brian Midei** - [bmmidei](https://github.com/bmmidei)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Kaggle competition for posing the original problem and providing data - https://www.kaggle.com/c/quora-insincere-questions-classification
