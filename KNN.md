

```python
import csv
import pandas as pd
from gensim.models import Word2Vec
import gensim
from sklearn.decomposition import PCA
import numpy as np
```


```python
df1=pd.read_csv("data.csv")
df1=df1.sample(frac=1)
tag=df1["Data"]
tag=np.array(tag)
tags=[]
for i in range(len(tag)):
    tags.append(tag[i])
Y=df1["Label"]
Y=np.array(Y)
Y=list(Y)
```


```python
d =[]
elim=['1','2','3','4','5','6','7','8','9','0',':',';','<','/',',','-','\\','#','"']
for i in tags:
    sent = " ".join("".join(["" if ch in elim else ch for ch in i]).split())
    d.append(sent)
final=[]
for i in d:
    k=i.split('>')
    final.append(k)
print((final[1]))
```

    ['', 'STYLE type=textcss', 'BODY{backgroundurl(javascriptdocument.cookie=true)}STYLE', '']



```python
model = gensim.models.Word2Vec(final,min_count =1,size=3)
model[model.wv.vocab]
length=[]
for i in final:
    length.append(len(i))
```

    /home/her/bhavya/lib/python3.6/site-packages/ipykernel_launcher.py:2: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      





    12




```python
zerovector=[]
for i in range(3):
  zerovector.append(0)
# print(label)
def sentencevector(sent,length):
  vector=[]
  for word in sent:
       vector.append(list(model[word]))
  if(len(sent)<max(length)):
    for i in range(len(sent),max(length)):
      vector.append(zerovector)
  return(vector)
Xtotal=[]
Ytotal=Y
for i in range(len(final)):
  a=(sentencevector(final[i],length))
  Xtotal.append(a)
```

    /home/her/bhavya/lib/python3.6/site-packages/ipykernel_launcher.py:8: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      



```python
len(Xtotal[3])
Xtotal[1]
```




    [[-0.11763515, 0.14821053, -0.027780948],
     [-0.0886701, 0.14920117, 0.11314271],
     [0.104578085, 0.074479245, 0.08459969],
     [-0.11763515, 0.14821053, -0.027780948],
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]]




```python
Xfinal=[]
for i in Xtotal:
    Xfinal.append(sum(list(i),[]))
len(Xfinal[0])
```




    36




```python
X_Train=Xfinal[:140]
Y_Train=Y[:140]
X_Test=Xfinal[140:]
Y_Test=Y[140:]
```


```python
type(X_Test)
```




    list




```python
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_Train,Y_Train)
Y_Predicted=neigh.predict(X_Test)
Y_Predicted=list(Y_Predicted)
Y_Test=list(Y_Test)
counter=0
for i in range (len(Y_Predicted)):
    if (Y_Predicted[i]==Y_Test[i]):
        counter=counter+1
accuracy=counter/len(Y_Predicted)
print(accuracy)
```

    0.8387096774193549

