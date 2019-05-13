# vulnerable-and-non-vulnerable-websites
## classification task using KNN classifier
#### The repo includes the following:

data.csv  -  It includes the hyperlinks of various trusted and non trusted websites
<br />
<br/>
knn.ipynb -  It is a python notebook the includes the  approach used
<br/>
<br/>
<br/>
The approach used here is to first convert the hyperlinks into vectors.For doing this we use gensim's Word2Vec library.
once converted and once the feature matrix is obtained which is padded with zero values as the length of the hyperlinks may not be the same.After this the feature matrix is fed into the KNN classifier, the one supplied with the sklearn package.
<br/>
<br/>
<br/>
#### ACCURACY
<br/>
Here  the accuracy obtained is calculated by taking the percentage of correctly classified instances with respect to the
total no. of instances.
<br/>
