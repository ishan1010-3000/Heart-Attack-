import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r'C:\Users\Ishan\Desktop\heart.csv')

pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6))
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()

corr = df.corr()
plt.figure(figsize=(18,10))
sns.heatmap(corr, annot=True)
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
X = df.drop(['target'], axis=1).values
y = df['target'].values


scale = StandardScaler()
X = scale.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = 0.3,random_state=42)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()


X = X_train
Y = Y_train

clf.fit(X,Y)

joblib.dump(clf,'Heart_model.pkl')