
# coding: utf-8

# In[194]:


import pandas as pd
import re


# In[195]:


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# In[196]:


def clean_tweet(tweet):
    tweet = re.sub("(http:\/\/)[^ ]*","",tweet)
    tweet = tweet.replace("&amp;","")
    tweet = re.sub("[^\w\s]","",tweet)
    tweet = re.sub("[\d]","",tweet)
    tweet = tweet.lower()
    return tweet
def remove_common_and_rare_words(tweet,com,rar):
    tweetWords = tweet.split(" ")
    newTweet = ""
    for word in tweetWords:
        if (word not in com) & (word not in rar):
            newTweet += word + " "
    return newTweet
def clean_dataset(dataset):
    dataset = dataset.apply(lambda x: clean_tweet(x))
    frequencyOfWords = pd.Series(' '.join(dataset).split()).value_counts()
    commonWords = frequencyOfWords[:10]
    rareWords = frequencyOfWords[-10:]
    dataset = dataset.apply(lambda x: remove_common_and_rare_words(x,commonWords,rareWords))
    return dataset


# In[197]:


train_data['TweetText'] = clean_dataset(train_data['TweetText'])
test_data['TweetText'] = clean_dataset(test_data['TweetText'])


# In[198]:


print(train_data['TweetText'])


# In[199]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
train_data_counts = count_vect.fit_transform(train_data['TweetText'])
train_data_counts.shape


# In[200]:


print(train_data_counts)


# In[201]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
train_data_tfidf = tfidf_transformer.fit_transform(train_data_counts)
train_data_tfidf.shape


# In[202]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(train_data_tfidf, train_data['Label'])


# In[203]:


test_data_counts = count_vect.transform(test_data['TweetText'])
test_data_tfidf = tfidf_transformer.transform(test_data_counts)
predicted = clf.predict(test_data_tfidf)


# In[204]:


for tweet, category in zip(test_data['TweetText'], predicted):
    print('%r => %s' % (tweet, category))


# In[205]:


import csv

header = ["TweetId","Label"]
rows = zip(test_data['TweetId'],predicted)
with open('sample_submission.csv', 'w') as submission:
    wr = csv.writer(submission, delimiter=',',lineterminator='\n', quoting=csv.QUOTE_ALL)
    wr.writerow(header)
    for row in rows:
        wr.writerow(row)

