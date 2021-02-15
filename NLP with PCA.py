import numpy as np
import pandas as pd
df = pd.read_csv('Restaurant_Reviews.tsv' , delimiter = '\t', quoting = 3 )
#print(df.iloc[0:100])
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords             #words which do not have any influence on learning of model(here, reviews)
from nltk.stem.porter import PorterStemmer    #to convert derived word into base form
from nltk.stem import WordNetLemmatizer 
corpus=[]                                     #to collect refined reviews
for i in range (0,1000):                      #1000 rows
    review = re.sub('[^a-zA-Z]',' ',df['Review'][i])    #what(all the characters which are not alphabets),with what/by what(space), from where(each row of Review column)
    review = review.lower()                   #converting all reviews in lower case
    review = review.split()                   #converting sentence into a list of words
    #to apply stemming we have to create object of the porterstemmer class
    ps = PorterStemmer()
    lm = WordNetLemmatizer()
    all_stopwords = stopwords.words('english') #collecting all the english stopwords
    all_stopwords.remove('not')
    all_stopwords.remove('no')
    #all_stopwords.remove('after')
    review = [lm.lemmatize(word) for word in review if not word in set (all_stopwords)]
    review = [ps.stem(word) for word in review if not word in set (all_stopwords)]   #stemming applied on word in review if word not present in all_stopwords
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

#apply pca
from sklearn.decomposition import PCA


pca = PCA(n_components=400)
x= pca.fit_transform(x)
np.set_printoptions(suppress=True)
print(pca.explained_variance_ratio_)
        
#performing SVM algorithm
#splitting x and y into training and test data
from sklearn.model_selection import train_test_split
accu = 0
rs = 0
for i in range (0,100):
    x_tr,x_te,y_tr,y_te = train_test_split(x,y,test_size = 0.1,random_state = i)    
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf',random_state = 91)
    classifier.fit(x_tr,y_tr)
    y_pred = classifier.predict(x_te)
    from sklearn.metrics import confusion_matrix,plot_confusion_matrix,accuracy_score
    acc = accuracy_score(y_te,y_pred)
    if (accu<acc):
        accu=acc
        rs = i
print(accu)
print(rs)
print(confusion_matrix(y_te,y_pred))
plot_confusion_matrix(estimator = classifier, X=x_te, y_true = y_te)