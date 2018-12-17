import math
import re

class NB_Classifier(object):

    def __init__(self, stopword_file=None):
        self.word_list = set()
        self.label_prior = {}
        self.word_given_label = {}

        if stopword_file is not None:
            self.stopwords = self.extract_stopwords(stopword_file)
        else:
            self.stopwords = None

    def extract_stopwords(self, stopword_file):
        stopwords = set()

        # Read in file line by line and create set of stop words
        with open(stopword_file) as file:
            for line in file:
                stopwords.add(line.strip())
        return stopwords

    def text_to_words(self, text, lowercase=True):
        '''
        This function separates a string into individual words
        :param text: Text as a string
        :return: A list of words
        '''
        # Use RegEx to separate essay into words. Note that special characters
        # are considered individual words.
        if lowercase == True:
            words = re.findall(r"[\w']+|[.,!?;]", text.lower())
            tokens = [word for word in words if word not in self.stopwords]
        else:
            tokens = re.findall(r"[\w']+|[.,!?;]", text)

        return tokens
    def collect_dictionary(self, q_text, min_freq):
        word_dict = {}

        for q in q_text:
            words = self.text_to_words(q, lowercase=True)
            for word in words:
                word_dict[word] = 1 if word not in word_dict else word_dict[word]+1

        frequent_words = [key for key,val in word_dict.items() if val>=min_freq]
        self.word_list = set(frequent_words)

    def train(self, q_text, labels, k=0.08):

        # Initialize occurrences to 0
        for word in self.word_list:
            self.word_given_label[(word,0)] = 0
            self.word_given_label[(word,1)] = 0

        for q, label in zip(q_text, labels):
            words = self.text_to_words(q, lowercase=True)

            # Update count in label prior dictionary
            if label not in self.label_prior:
                self.label_prior[label] = 1
            else:
                self.label_prior[label] += 1

            # Update count in conditional probability dictionary
            for word in words:
                if word not in self.word_list:
                    pass
                else:
                    self.word_given_label[(word,label)] += 1

        # Use dictionary comprehension to compute probabilities from occurrence counts
        total_samples = sum(val for key,val in self.label_prior.items())
        self.label_prior = {key:val/total_samples for (key,val) in self.label_prior.items()}

        # Create 2 new dictionaries, one for each label
        pos_dict = {key:val for (key,val) in self.word_given_label.items() if key[1]==0}
        neg_dict = {key:val for (key,val) in self.word_given_label.items() if key[1]==1}

        # Calculate total number of words in positive and negative comments
        total_pos_words = sum(val for key,val in pos_dict.items())
        total_neg_words = sum(val for key,val in neg_dict.items())

        # Calculate total possible classes
        num_unique_words = len(self.word_list)

        # Calculate word-given-label probabilities for each label according to additive smoothing formula
        pos_dict = {key:(val+k)/(total_pos_words+k*num_unique_words) for (key,val) in pos_dict.items()}
        neg_dict = {key:(val+k)/(total_neg_words+k*num_unique_words) for (key,val) in neg_dict.items()}

        # Merge dictionaries
        self.word_given_label = {**pos_dict, **neg_dict}

    def predict(self, text):
        # Obtain a list of individual words
        words = self.text_to_words(text)

        # remove words not found in training data
        words = [word for word in words if word in self.word_list]

        # Create a list of probabilities for each word given the label
        pos_probs = [self.word_given_label[(word, 0)] for word in words]
        neg_probs = [self.word_given_label[(word, 1)] for word in words]

        # Convert to log probabilities
        log_pos_probs = [math.log(prob) for prob in pos_probs]
        log_neg_probs = [math.log(prob) for prob in neg_probs]

        # Calculate total probability for each label
        prob_dict = {}
        prob_dict[0] = math.log(self.label_prior[0]) + sum(log_pos_probs)
        prob_dict[1] = math.log(self.label_prior[1]) + sum(log_neg_probs)

        return prob_dict

    def evaluate(self, q_text, labels):
        # Initialize correct and incorrect counts
        num_correct = 0
        num_incorrect = 0

        for q, label in zip(q_text, labels):
            # Predict label for the text
            pred_dict = self.predict(q)
            pred = max(pred_dict, key=pred_dict.get)

            # Update counts
            if label == pred:
                num_correct += 1
            else:
                num_incorrect += 1

        return num_correct/(num_correct+num_incorrect)

    def generate_preds(self, q_text):

        num_pos = 0
        num_neg = 0
        preds = [0]*len(q_text)
        print('There are {} comments in the test file.'.format(len(preds)))
        for idx, q in enumerate(q_text):
            # Predict label for the text
            pred_dict = self.predict(q)
            pred = max(pred_dict, key=pred_dict.get)
            if pred == 0:
                num_pos += 1
            else:
                num_neg += 1
            preds[idx] = pred

        return preds



