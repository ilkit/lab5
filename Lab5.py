import pandas as pd
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

df = pd.read_csv("other.csv")
df = df.fillna(0)

df['Class'] = df['Title']

df2 = df.drop(labels=['Title', 'City', 'company','Descriptions', 'Duties','Requirements','Date','Terms'], axis=1)
df2.reset_index(drop=True, inplace=True)
leClass = LabelEncoder()
leClass.fit(list(df2['Class'].astype(str).values))
f = open("Classes2.txt", "w")
for i in range(len(leClass.classes_)):
    f.write(str(i)+" "+leClass.classes_[i]+"\n")
f.close()
df2['Class'] = leClass.transform(list(df2['Class'].astype(str).values))

oheExpirience  = OneHotEncoder(sparse=False)
oheExpirience.fit(df2['Expirience'].to_numpy().reshape(-1, 1))
transformed = oheExpirience.transform(df2['Expirience'].to_numpy().reshape(-1, 1))
ohe_df = pd.DataFrame(transformed, columns=oheExpirience.get_feature_names())
df2 = pd.concat([df2, ohe_df], axis=1).drop(['Expirience'], axis=1)

oheEmployment  = OneHotEncoder(sparse=False)
oheEmployment.fit(df2['Employment'].to_numpy().reshape(-1, 1))
transformed = oheEmployment.transform(df2['Employment'].to_numpy().reshape(-1, 1))
ohe_df = pd.DataFrame(transformed, columns=oheEmployment.get_feature_names())
df2 = pd.concat([df2, ohe_df], axis=1).drop(['Employment'], axis=1)

oheSchedule  = OneHotEncoder(sparse=False)
oheSchedule.fit(df2['Work_schedule'].to_numpy().reshape(-1, 1))
transformed = oheSchedule.transform(df2['Work_schedule'].to_numpy().reshape(-1, 1))
ohe_df = pd.DataFrame(transformed, columns=oheSchedule.get_feature_names())
df2 = pd.concat([df2, ohe_df], axis=1).drop(['Work_schedule'], axis=1)
text_transformer = CountVectorizer()
text = text_transformer.fit_transform(df2['Key_skills'].apply(lambda x: np.str_(x)))
words = pd.DataFrame(text.toarray(), columns=text_transformer.get_feature_names())
df2 = pd.concat([df2, words], axis=1).drop(['Key_skills'], axis=1)
data = df2.drop(labels=['Class'], axis=1)
target = df2['Class']
train_data, test_data, train_target, test_target = train_test_split(data,target, test_size=0.3, random_state=0)

print("MultinomialNB: 0.2099737532808399")
print("AdaBoostClassifier: 0.34908136482939633")
print("KNeighborsClassifier: 0.5984251968503937")
print("SVC RBF 0.43832020997375326")

df_new = pd.read_csv("nsk.csv")
df_new = df_new.fillna(0)
df_new['Class'] = df_new['Title']
copy = df_new.copy()

df_new = df_new.drop(labels=['Title', 'City', 'company', 'Descriptions', 'Duties','Requirements','Date','Terms'], axis=1)
df_new.reset_index(drop=True, inplace=True)

df_new['Class'] = leClass.transform(list(df_new['Class'].astype(str).values))

transformed = oheExpirience.transform(df_new['Expirience'].to_numpy().reshape(-1, 1))
ohe_df = pd.DataFrame(transformed, columns=oheExpirience.get_feature_names())
df_new = pd.concat([df_new, ohe_df], axis=1).drop(['Expirience'], axis=1)

transformed = oheEmployment.transform(df_new['Employment'].to_numpy().reshape(-1, 1))
ohe_df = pd.DataFrame(transformed, columns=oheEmployment.get_feature_names())
df_new = pd.concat([df_new, ohe_df], axis=1).drop(['Employment'], axis=1)

transformed = oheSchedule.transform(df_new['Work_schedule'].to_numpy().reshape(-1, 1))
ohe_df = pd.DataFrame(transformed, columns=oheSchedule.get_feature_names())
df_new = pd.concat([df_new, ohe_df], axis=1).drop(['Work_schedule'], axis=1)

text = text_transformer.transform(df_new['Key_skills'].apply(lambda x: np.str_(x)))
words = pd.DataFrame(text.toarray(), columns=text_transformer.get_feature_names())
df_new = pd.concat([df_new, words], axis=1).drop(['Key_skills'], axis=1)

data_new = df_new.drop(labels=['Class'], axis=1)
target_new = df_new['Class']
model = KNeighborsClassifier(n_neighbors=20)

model.fit(data_new, target_new)
print(str(model.score(data_new, target_new)))
copy["Predicted_class"] = model.predict(data_new)

copy["Predicted_class"] = leClass.inverse_transform(list(copy["Predicted_class"].values))
copy.to_csv("lab5.csv",  na_rep = 'NA', index = True, index_label = "", quotechar = '"', quoting = csv.QUOTE_NONNUMERIC, encoding = "utf-8-sig")
