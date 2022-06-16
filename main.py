import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

train_data=pd.read_csv("data1/train_4.csv",sep='\t',nrows=600)
test_data=pd.read_csv("data1/test_a2.csv",sep='\t',nrows=200)
# Tfidf
tf_idf=TfidfVectorizer(max_features=2000)
train_tfidf = tf_idf.fit_transform(train_data['text'].values)

test_tfidf=tf_idf.transform(test_data['text'].values)

# 岭回归
clf=RidgeClassifier()
clf.fit(train_tfidf, train_data['label'].values)
RidgeClassifier()

# # F1值
val_pred = clf.predict(train_tfidf[1:])
print(f1_score(train_data['label'].values[1:], val_pred, average='macro'))

# 测试集
df=pd.DataFrame()
df['label']=clf.predict(test_tfidf)
df.to_csv('submit.csv', index=False)

