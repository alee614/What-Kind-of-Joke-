from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
porter = PorterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score


label_dict = {'Animal': 0, 'At Work': 1, 'Bar': 2, 'Blond': 3, 'Blonde': 3, 'Children': 4, 'College': 5, 'Gross':6,
              'Insults': 7, 'Knock-Knock': 8, 'Lawyer': 9, 'Lightbulb': 10, 'Medical': 11, 'Men / Women': 12,
              'News / Politics': 13, 'One Liners': 14, 'Puns': 15, 'Redneck': 16, 'Religious': 17, 'Sports': 18,
              'Tech': 19, 'Yo Mama': 20, 'Yo Momma': 20}

stupid_dict = {'Animals': 0, 'Bar Jokes': 2, 'Blonde Jokes': 3, 'Children': 4,
              'Computers': 19, 'Insults': 7, 'Lawyers': 9, 'Light Bulbs': 10, 'Medical': 11, 'Men': 12,
              'One Liners': 14, 'Political': 13, 'Redneck': 16, 'Religious': 17,
              'Sports': 18, 'Women': 12, 'Yo Mama': 20, 'Office Jokes': 1}


df = pd.read_json('wocka.json')
stupid = pd.read_json('stupidstuff.json')

col = ['body', 'category']
df = df[col]
stupid = stupid[col]

df['category_code'] = df['category']
df = df[df.category != 'Other / Misc']

df = df.replace({'category_code': label_dict})

num = 0
for item in stupid.category:
    if item == 'Business' or item == 'Miscellaneous' or item == 'Aviation' or \
            item == 'Blind Jokes' or item == 'Crazy Jokes' or item == 'Deep Thoughts' or \
            item == 'English' or item == 'Ethnic Jokes' or item == 'Family, Parents' or \
            item == 'Farmers' or item == 'Food Jokes' or item == 'Heaven and Hell' or item == 'Holidays' \
            or item == 'Idiots' or item == 'Love & Romance' or item == 'Marriage' or item == 'Military' \
            or item == 'Money' or item == 'Music' or item == 'Old Age' or item == 'Police Jokes' or item == 'School' \
            or item == 'Sex' or item == 'State Jokes' or item == 'Science':
        #stupid = stupid.set_index("category")
        stupid.drop(num, inplace=True)
    num += 1

stupid['category_code'] = stupid['category']
stupid = stupid.replace({'category_code': stupid_dict})

df = pd.concat([df, stupid], ignore_index=True)

'''
fig, ax = plt.subplots(figsize=(30,10))
plt.xticks(fontsize=20) # X Ticks
plt.yticks(fontsize=20) # Y Ticks
ax.set_title('Number of jokes per Category', fontweight="bold", size=25) # Title
ax.set_ylabel('Number of Jokes', fontsize = 25) # Y label
ax.set_xlabel('Categories', fontsize = 25) # X label
df.groupby(['category_code']).count()['body'].plot(ax=ax, kind='bar')
plt.show()
'''

body_dict = {}

for body in df.body:
    body.replace("\n", " ")
    tokens = word_tokenize(body)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    stemmed = [porter.stem(w) for w in words]
    lemmas = [lemmatizer.lemmatize(w) for w in stemmed]
    words = " ".join(lemmas)
    body_dict[body] = words

df_2 = df
df = df.replace({'body': body_dict})

# Parameter election
ngram_range = (1,2)
min_df = 10
max_df = 1.

X_train, X_test, y_train, y_test = train_test_split(df['body'], df['category_code'], test_size=0.15)

vectorizer = TfidfVectorizer(encoding='utf-8',ngram_range=ngram_range, stop_words=None, lowercase=False,
                             max_df=max_df, min_df=min_df, norm='l2',sublinear_tf=True)

features_train = vectorizer.fit_transform(X_train).toarray()
labels_train = y_train
#print(features_train.shape)

features_test = vectorizer.transform(X_test).toarray()
labels_test = y_test
#print(features_test.shape)


input1 = word_tokenize(input("Your funny joke: "))
input1 = [w.lower() for w in input1]
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in input1]
input1 = [word for word in stripped if word.isalpha()]
stop_words = set(stopwords.words('english'))
eng_words = set(nltk.corpus.words.words())
input1 = [w for w in input1 if w in eng_words and w not in stop_words]
stemmed = [porter.stem(w) for w in input1]
lemmas = [lemmatizer.lemmatize(w) for w in stemmed]
input1 = " ".join(lemmas)

query_vec = vectorizer.transform([input1])
results = cosine_similarity(features_train, query_vec).reshape((-1,))

'''
rank = 1
related_categories = []
for i in results.argsort()[-20:][::-1]:
    if rank != 11:
        if df.iloc[i, 2] not in related_categories:
            print(rank, " ", df.iloc[i, 1])
            related_categories.append(df.iloc[i, 2])
            rank = rank + 1
'''

selected_index = results.argsort()[len(results) - 1]
selected_code = df.iloc[selected_index, 2]


text = df.loc[df['category_code'] == selected_code, 'body']
selected_features = vectorizer.fit_transform(text).toarray()
query_2 = vectorizer.transform([input1])
results = cosine_similarity(selected_features, query_2).reshape((-1,))
joke_index = results.argsort()[len(results) - 1]
joke = df_2.iloc[joke_index, 0]

print("Get ready to laugh:")
print(" ")
print(joke)




#take the body of the selected_category


'''
# check correlations with each category
# print(labels_train)
X = vectorizer.fit_transform(df['body']).toarray()

naive_bayes = MultinomialNB()
naive_bayes.fit(features_train, labels_train)
predictions = naive_bayes.predict(features_test)

scores = cross_val_score(naive_bayes, X, df['category_code'], cv=5)
print(scores)
print("Accuracy of Naive Bayes: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print("Naive Bayes Model: ")
print(classification_report(labels_test, predictions))

logreg = LogisticRegression(max_iter=10000)
logreg.fit(features_train, labels_train)
logreg_pred = logreg.predict(features_test)


scores = cross_val_score(logreg, X, df['category_code'], cv=5)
print(scores)
print("Accuracy of Logistic Regression: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print('Logistic Regression Model: ')
print(classification_report(labels_test, logreg_pred))


forest = RandomForestClassifier(max_depth=10000)
forest.fit(features_train, labels_train)
forest_pred = forest.predict(features_test)

scores = cross_val_score(forest, X, df['category_code'], cv=5)
print(scores)
print("Accuracy of Random Forest: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


print('Random Forest Model: ')
print(classification_report(labels_test, forest_pred))
'''

'''
from sklearn.feature_selection import chi2
import numpy as np

for Product, category_id in sorted(label_dict.items()):
    features_chi2 = chi2(features_train, labels_train == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(vectorizer.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}' category:".format(Product))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-2:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
    print("")
'''