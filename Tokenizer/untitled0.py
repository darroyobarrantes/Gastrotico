# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:52:41 2022

@author: fafec
"""

# tokenize 
tokenized = word_tokenize(cleaned)
# remove stopwords 
all_words = [w for w in tokenized if not w in stop_words]
#Count all words
len(all_words)


frequency = pd.Series(all_words).value_counts().sort_values(ascending = False)


#Frequency distribution
BOW = nltk.FreqDist(all_words)

# Listing the 12000 most frequent words (most of total words)
word_features = list(BOW.keys())[:12000]


#Creates features
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# Creating features for each comment
documents = df[column].tolist()
featuresets = [find_features(comment) for comment in documents]
# Shuffling the documents 
#import random
#random.shuffle(featuresets)
featuresets = pd.DataFrame.from_dict(featuresets)


df.reset_index(drop=True, inplace=True)
featuresets.reset_index(drop=True, inplace=True)
df = df.join(featuresets)

#Transform comma-separated Customer VoC categories for One Hot ending
df = pd.concat([df.drop('Customer VoC', 1), df['Customer VoC'].str.get_dummies(sep=".")], 1)


#Joins newly created columns and original 
df_join = df.join(df_2)
#Drops categorical columns no longer needed 
df_join = df_join.drop([
    'Incident Number', 'HR Experience Rating','HR Experience Answer',
    'HR Representative Rating', 'HR Representative Answer',
    'Assignee', 'Process Role','Region',
    'Country','Service Component','Support Skill','Product','Product Feature','Reported Source',
    'CSAT Customer Feedback', 'Description', 'Detailed Description', 'Comment_Description',
    'Submit Timestamp', 'Closed Timestamp', 'Survey Sent Date', 'Survey - Intel Quarter', 'Survey - Intel WW',
    #'Time',
    column,
    ],
    axis=1)


Despues de split:
# Detect class imbalance
"""Most of the contemporary works in class imbalance concentrate on imbalance ratios ranging from 1:4 up to 1:100. […] In real-life applications such as fraud detection or cheminformatics we may deal with problems with imbalance ratio ranging from 1:1000 up to 1:5000.
— Learning from imbalanced data – Open challenges and future directions, 2016."""

minorityclass = count.index[-1]
minoritycount = (df["Survey Feedback"] == minorityclass).sum() / df["Survey Feedback"].count() * 100
print(round(minoritycount, 2))

if minoritycount >25:
    print("Fairly Balanced data set")
elif minoritycount <=25 and minoritycount >1:
    print ("Imbalanced data set")
elif minoritycount <=1 and minoritycount >0.1:
    print ("Strongly imbalanced data set")
elif minoritycount <=0.1:
    print ("Severily imbalanced data set")


from imblearn.under_sampling import TomekLinks

sampling_strategy = "not majority"
ros = RandomOverSampler(sampling_strategy=sampling_strategy)

#X_train = df4.astype(float) #Reassign final f4 to X
#y_train = y_train

X, y = ros.fit_resample(X_train, y_train) #Upsampled set

#Visualize balanced classes
graph = sns.countplot(x=y).set_title("Survey Feedback")
