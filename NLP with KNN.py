import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Restaurant_Reviews.tsv' , delimiter = '\t', quoting = 3 )
#print(df.iloc[0:100])
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords             #words which do not have any influence on learning of model(here, reviews)
from nltk.stem import WordNetLemmatizer    #to convert derived word into base form

corpus=[]                                     #to collect refined reviews
for i in range (0,1000):                      #1000 rows
    review = re.sub('[^a-zA-Z]',' ',df['Review'][i])    #what(all the characters which are not alphabets),with what/by what(space), from where(each row of Review column)
    review = review.lower()                   #converting all reviews in lower case
    review = review.split()                   #converting sentence into a list of words
    #to apply stemming we have to create object of the porterstemmer class
    lm = WordNetLemmatizer()
    #ps = PorterStemmer()
    all_stopwords = stopwords.words('english') #collecting all the english stopwords
    all_stopwords.remove('not')
    all_stopwords.remove('no')
    review = [lm.lemmatize(word) for word in review if not word in set (all_stopwords)]   #stemming applied on word in review if word not present in all_stopwords
    #review = [ps.stem(word) for word in review if not word in set (all_stopwords)]
    #to convert lost of words into sentence, use join function
    review = ' '.join(review)
    corpus.append(review)
print()
print(corpus[0:6])
#creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = df.iloc[:,-1].values
len(x[0])
print(len(x[0]))

#splitting x and y into training and test data
from sklearn.model_selection import train_test_split
accu = 0
rs = 0
for i in range (0,100):
    x_tr,x_te,y_tr,y_te = train_test_split(x,y,test_size = 0.045,random_state = i)
    #create and train KNN model
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors= 5, metric='minkowski',p=1)
    classifier.fit(x_tr,y_tr)
    y_pred = classifier.predict(x_te)
    from sklearn.metrics import confusion_matrix,plot_confusion_matrix,accuracy_score
    acc = accuracy_score(y_te,y_pred)
    if (accu<acc):
        accu=acc
        rs = i
print (rs)
print(accu)
print(confusion_matrix(y_te,y_pred))
plot_confusion_matrix(estimator = classifier, X=x_te, y_true = y_te,cmap = 'Blues')




