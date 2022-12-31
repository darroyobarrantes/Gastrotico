#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis - Lexicon based approach with VADER
# Please create a folder called "algos" to be able to pickle the Machine Learning models

# # Step 1: Package Installation

# In[2]:


#Install Packages - only first time
get_ipython().system('pip install nltk')
get_ipython().system('pip install --upgrade nltk')
#Restart kernell


# In[3]:


get_ipython().system('pip install textblob')
#Restart kernell


# In[4]:


from textblob import TextBlob


# In[5]:


import nltk


# In[6]:


#Install Packages - only first time
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
import nltk
from nltk.sentiment import vader
import re
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# ## Step 2. Data Upload

# In[8]:


#Time experiment duration
start_time = time.time()


# In[9]:


df = pd.read_csv('test.csv', encoding = "UTF-8")


# In[10]:


df.head()


# In[11]:


#Splits positive and negative comments into two lists
positive = df.loc[df['HR Experience Rating'] == 'Positive']
positiveReviews = positive['Comments'].astype(str).tolist()

negative = df.loc[df['HR Experience Rating'] == 'Negative']
negativeReviews = negative['Comments'].astype(str).tolist()


# ## Step 3. Apply VADER and TextBlob
# ### VADER

# In[12]:


#Calls the Sentiment Analyzer
sia = vader.SentimentIntensityAnalyzer()


# In[13]:


#Let's try an example:
sia.polarity_scores("the agent was nice but the service was very bad")


# In[14]:


#This function evaluates the rating to all comments in our lists

def vaderSentiment(review):
  return sia.polarity_scores(review)['compound']

def getReviewSentiments(sentimentCalculator): 
  negReviewResult = [sentimentCalculator(oneNegativeReview) for oneNegativeReview in negativeReviews]
  posReviewResult = [sentimentCalculator(onePositiveReview) for onePositiveReview in positiveReviews]
  return {'results-on-positive':posReviewResult, 'results-on-negative':negReviewResult}

def runDiagnostics(reviewResult):
  positiveReviewsResult = reviewResult['results-on-positive']
  negativeReviewsResult = reviewResult['results-on-negative']
  pctTruePositive = float(sum(x > 0 for x in positiveReviewsResult))/len(positiveReviewsResult)
  pctTrueNegative = float(sum(x < 0 for x in negativeReviewsResult))/len(negativeReviewsResult)
  totalAccurate = float(sum(x > 0 for x in positiveReviewsResult)) + float(sum(x < 0 for x in negativeReviewsResult))
  total = len(positiveReviewsResult) + len(negativeReviewsResult)
  print ("Accuracy on positive reviews = " +"%.2f" % (pctTruePositive*100) + "%")
  print ("Accuracy on negative reviews = " +"%.2f" % (pctTrueNegative*100) + "%")
  print ("Overall accuracy = " + "%.2f" % (totalAccurate*100/total) + "%")

runDiagnostics(getReviewSentiments(vaderSentiment))


# In[15]:


#Now that we are happy with the results, let's apply it to all the data
#Add VADER metrics to dataframe
df['Comments'] = df['Comments'].astype(str)
df['compound'] = [sia.polarity_scores(v)['compound'] for v in df['Comments']]
df['compound'] = df['compound'].astype(float)
df['neg'] = [sia.polarity_scores(v)['neg'] for v in df['Comments']] 
df['neu'] = [sia.polarity_scores(v)['neu'] for v in df['Comments']]
df['pos'] = [sia.polarity_scores(v)['pos'] for v in df['Comments']]
df['SentimentRate'] = df['compound'].apply(lambda x: 'Positive' if x > 0 else "Neutral" if x == 0 else 'Negative')
df.head(3)


# ### Textblob

# In[16]:


#load the descriptions into textblob
desc_blob = [TextBlob(desc) for desc in df['Comments']]
#add the sentiment metrics to the dataframe
df['tb_Polarity'] = [b.sentiment.polarity for b in desc_blob]
df['tb_Subjectivity'] = [b.sentiment.subjectivity for b in desc_blob]
#show dataframe
df.head(15)


# ## Step 4. Visualize and Save final results

# In[17]:


#Visualize results
#Table
my_crosstab = pd.crosstab(index=df["SentimentRate"], 
                            columns="count")   # Include row and column totals
my_crosstab


# In[154]:


#Graph
import matplotlib.pyplot as plt
df.SentimentRate.value_counts().sort_values().plot(kind = 'barh')


# ## Step 5. Machine Learning

# ### Feature Prep and Engineering

# In[23]:


all_words = []
documents = []

stop_words = list(set(stopwords.words('english')))

#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J", "R"]

for p in  positiveReviews:
    
    # create a list of tuples where the first element of each tuple is a review
    # the second element is the label
    documents.append( (p, "pos") )
    
    # remove punctuations
    cleaned = re.sub(r'[^A-Za-z0-9 ]+','', p)
    
    # tokenize 
    tokenized = word_tokenize(cleaned)
    
    # remove stopwords 
    stopped = [w for w in tokenized if not w in stop_words]
    
    # parts of speech tagging for each word 
    pos = nltk.pos_tag(stopped)
    
    # make a list of  all adjectives identified by the allowed word types list above
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    
for p in negativeReviews:
    # create a list of tuples where the first element of each tuple is a review
    # the second element is the label
    documents.append( (p, "neg") )
    
    # remove punctuations
    cleaned = re.sub(r'[^A-Za-z0-9 ]+','', p)
    
    # tokenize 
    tokenized = word_tokenize(cleaned)
    
    # remove stopwords 
    stopped = [w for w in tokenized if not w in stop_words]
    
    # parts of speech tagging for each word 
    neg = nltk.pos_tag(stopped)
    
    # make a list of  all adjectives identified by the allowed word types list above
    for w in neg:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


# In[24]:


#Measure how many words are being analysed
len(all_words)


# In[28]:


# Creates a list of negative or positive words for analysis
pos_A = []
for w in pos:
    if w[1][0] in allowed_word_types:
        pos_A.append(w[0].lower())
pos_N = []
for w in neg:
    if w[1][0] in allowed_word_types:
        pos_N.append(w[0].lower())


# In[30]:


# pickling the list documents to save future recalculations 
save_documents = open("algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()


# In[31]:


# creating a frequency distribution of each adjectives. 
BOW = nltk.FreqDist(all_words)
BOW


# In[32]:


# listing the 5000 most frequent words (around half of the total words, that is 10110)
word_features = list(BOW.keys())[:5000]
word_features[0], word_features[-1]


# In[33]:


#Saves the features for analysis
save_word_features = open("algos/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


# In[34]:


# function to create a dictionary of features for each review in the list document.
# The keys are the words in word_features 
# The values of each key are either true or false for wether that feature appears in the review or not

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# Creating features for each review
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# Shuffling the documents 
random.shuffle(featuresets)
print(len(featuresets))


# ### Splitting training and testing sets
# (about 60% for training, as many words might not be represented in smaller testing sets)

# In[37]:


training_set = featuresets[:2000]
testing_set = featuresets[2000:]
print( 'training_set :', len(training_set), '\ntesting_set :', len(testing_set))


# ### Algorithm Modeling

# In[41]:


#We try the most used algorithm, that is Naive Bayes Classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)
# We measure accuracy and find out the most informative features
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)


# In[42]:


# Printing the most important features 
mif = classifier.most_informative_features()
mif = [a for a,b in mif]
print(mif)


# In[73]:


# Predictions
# getting predictions for the testing set by looping over each reviews featureset tuple
# The first element of the tuple is the feature set and the second element is the label 
ground_truth = [r[1] for r in testing_set]
preds = [classifier.classify(r[0]) for r in testing_set]


# In[46]:


#For example, print the prediction for the first element of the testing set:
print(preds[0])


# ### Model Evaluation

# In[47]:


#Evaluate the Naive Bayes algorithm using F1-score (more robust than accuracy)
from sklearn.metrics import f1_score
f1_score(ground_truth, preds, labels = ['neg', 'pos'], average = 'micro')


# In[48]:


#Now let's compare with other algorithms
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
#If you want to see the informative features of other algorithms, you can use
#classifier.show_most_informative_features(15)

MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_clf, testing_set))*100)

BNB_clf = SklearnClassifier(BernoulliNB())
BNB_clf.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BNB_clf, testing_set))*100)

LogReg_clf = SklearnClassifier(LogisticRegression())
LogReg_clf.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogReg_clf, testing_set))*100)

SGD_clf = SklearnClassifier(SGDClassifier())
SGD_clf.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGD_clf, testing_set))*100)

SVC_clf = SklearnClassifier(SVC())
SVC_clf.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_clf, testing_set))*100)


# In[50]:


#Let's pickle the algorithms for fast deployment
def create_pickle(c, file_name): 
    save_classifier = open(file_name, 'wb')
    pickle.dump(c, save_classifier)
    save_classifier.close()

classifiers_dict = {'ONB': [classifier, 'algos/ONB_clf.pickle'],
                    'MNB': [MNB_clf, 'algos/MNB_clf.pickle'],
                    'BNB': [BNB_clf, 'algos/BNB_clf.pickle'],
                    'LogReg': [LogReg_clf, 'algos/LogReg_clf.pickle'],
                    'SGD': [SGD_clf, 'algos/SGD_clf.pickle'], 
                    'SVC': [SVC_clf, 'algos/SVC_clf.pickle']}

for clf, listy in classifiers_dict.items(): 
    create_pickle(listy[0], listy[1])


# In[51]:


#Get Accuracy Results
from sklearn.metrics import f1_score, accuracy_score
ground_truth = [r[1] for r in testing_set]
predictions = {}
f1_scores = {}
acc_scores = {}
for clf, listy in classifiers_dict.items(): 
    # getting predictions for the testing set by looping over each reviews featureset tuple
    # The first elemnt of the tuple is the feature set and the second element is the label
    predictions[clf] = [listy[0].classify(r[0]) for r in testing_set]
    acc_scores[clf] = accuracy_score(ground_truth, predictions[clf])
    print(f'Accuracy_score {clf}: {acc_scores[clf]}')


# In[52]:


#Get F1 Scores
for clf, listy in classifiers_dict.items(): 
    # getting predictions for the testing set by looping over each reviews featureset tuple
    # The first elemnt of the tuple is the feature set and the second element is the label 
    predictions[clf] = [listy[0].classify(r[0]) for r in testing_set]
    f1_scores[clf] = f1_score(ground_truth, predictions[clf], labels = ['neg', 'pos'], average = 'micro')
    print(f'f1_score {clf}: {f1_scores[clf]}')


# ## Machine Learning Ensemble Model
# An ensemble of algorithms can provide better performing of more robust results.  Let's see how this works in our case

# In[53]:


from nltk.classify import ClassifierI

# Defininig the ensemble model class 

class EnsembleClassifier(ClassifierI):
    
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    # returns the classification based on majority of votes
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    # a simple measurement the degree of confidence in the classification 
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


# In[55]:


# Load all classifiers from the pickled files we created above

# function to load models given filepath
def load_model(file_path): 
    classifier_f = open(file_path, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier


# Original Naive Bayes Classifier
ONB_Clf = load_model('algos/ONB_clf.pickle')

# Multinomial Naive Bayes Classifier 
MNB_Clf = load_model('algos/MNB_clf.pickle')


# Bernoulli  Naive Bayes Classifier 
BNB_Clf = load_model('algos/BNB_clf.pickle')

# Logistic Regression Classifier 
LogReg_Clf = load_model('algos/LogReg_clf.pickle')

# Stochastic Gradient Descent Classifier
SGD_Clf = load_model('algos/SGD_clf.pickle')


# Initializing the ensemble classifier 
ensemble_clf = EnsembleClassifier(ONB_Clf, MNB_Clf, BNB_Clf, LogReg_Clf, SGD_Clf)

# List of only feature dictionary from the featureset list of tuples 
feature_list = [f[0] for f in testing_set]

# Looping over each to classify each review
ensemble_preds = [ensemble_clf.classify(features) for features in feature_list]


# ### Model Ensemble Evaluation

# In[56]:


#Now let's get the F1-score evaluation for the ensemble
f1_score(ground_truth, ensemble_preds, average = 'micro')


# In[58]:


#Let's print the most informative features
classifier.show_most_informative_features(25)


# In[61]:


#Let's visualize a Confusion Matrix to understand the distribution of False positives and False negatives
y_test = ground_truth
y_pred = ensemble_preds
class_names = ['neg', 'pos']



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# ### Making predictions

# In[83]:


# List of only feature dictionary from the featureset list of tuples 
feature_list = [f[0] for f in featuresets]

# Looping over each to classify each review
preds = [ensemble_clf.classify(features) for features in feature_list]

df['Ensemble Rating']=preds
df.head(20)


# ### Live Sentiment Analysis Demonstration

# ### Here you can test the performance of the ensemble model.  Create your own example phrases and check how well it performs

# In[62]:


# Function to classify a given review
# Returns positive or negative label and the confidence in the classification.

def sentiment(text):
    feats = find_features(text)
    return ensemble_clf.classify(feats), ensemble_clf.confidence(feats)


# In[63]:


# Examples
text_a = '''The experience was frustrating. the ticket was cloased and I had to connect again.
the response was vague and it took two weeks to answer the enquiry.'''
text_b = '''AskIvy did not provide the appropiate answer, it was frustating and had to connect with human agent'''
text_c = '''The agent was fantastic, quick reply, he deserves a recognition'''
text_d = '''"My experience was not frustrating at all. I did not feel it was a bad experience"'''


# In[64]:


print(text_a, sentiment(text_a))


# In[65]:


print(text_b, sentiment(text_b))


# In[66]:


print(text_c, sentiment(text_c))


# In[67]:


print(text_d, sentiment(text_d))


# ### Step 6 Print your predictions to Final File

# In[84]:


#Print Results to csv
df.to_csv('sentimentpredictions.csv', index = False)


# In[68]:


# Running Time
print("--- %s minutes ---" % ((time.time() - start_time)/60))

