## MIS IT 400 DESIGN PROJECT

**IMPROVEMENT OF COMPUTATIONAL PERFORMANCE OF PCA**

Submitted by:

_Buket Hancı (MIS)_

_Seval Uçar (MIS)_

_Umut Çiftci (MIS)_

Project Supervisor: Dr. Emrullah Fatih YETKİN

Faculty of Management

Kadir Has University

Spring 2020

**ABSTRACT**

The research questions of this project are &quot;How to improve computational performance in text data by using the PCA algorithm?&quot; and &quot;How to improve the computational performance of the PCA algorithm?&quot;. The biggest problem encountered in big data projects is that the computational power is too high, and the machine used is not sufficient. To overcome this, more powerful machines are purchased, and the cost of the project increases at the same rate. This project aims to improve the computational performance of the PCA technique. The algorithm uses the concepts of variance matrix, covariance matrix, eigenvector, and eigenvalues pairs to perform PCA, providing a set of eigenvectors and their respective eigenvalues as a result [1]. Calculating this matrix requires too much computational performance. Before developing a new approach, the performance of PCA has been shown by calculating the time of matrix and time of eigenvalues to prove the research question. The new approach has been applied in many datasets to compare results sensibly. During the project Python and its libraries being used.

**TABLE OF CONTENTS**

ABSTRACT………………………………………………………………………... 2

1. INTRODUCTION……………………………………………………………... 4
2. MATHEMATICS…………………………………………………………….... 5

  1. Standard Deviation………………………………………………………. 6
  2. Variance…………………………………………………………………. 6
  3. Covariance………………………………………………………………. 7
  4. Covariance Matrix………………………………………………………. 7
  5. Eigenvector and Eigenvalue ……………………………………………. 7

3 PRINCIPAL COMPONENT ANALYSIS (PCA) ……………………………... 8

4 APPLICATION OF PCA………………………………………………………. 10

4.1 Preparation of Dataset …………………………………………………… 10

  1. Subtracting from Average………………………………………………... 12
  2. Covariance Matrix………………………………………………………... 12

4.4 Eigenvector and Eigenvalue……………………………………………… 12

5 A NEW APPROACH TO IMPROVE PCA……………………………………. 14

5.1 A New Approach to Reduce Calculation Times– Python Code…………... 16

5.2 A New Approach to Reduce Calculation Times– Flowchart……………... 17

5.3 Proposed Method with Hand-Written Digits Data………………………... 18

5.4 Proposed Method with MNIST Data……………………………………... 22

1. CONCLUSION………………………………………………………………… 28

7 REFERENCES………………………………………………………………… 29

1. **INTRODUCTION**

Principal Component Analysis (PCA) is a mathematical technique to explain information in a multivariate dataset with fewer variables and minimal loss of information [1][3]. In other words, PCA is a transformation technique that enables the size of the dataset, which contains a large number of interrelated variables, to be reduced to a smaller size by preserving the data in the dataset [1]. PCA reduces dimensionality in large datasets [1][2].

In this study, the Principal Component Analysis is mentioned. This project aims to reduce the dimension of the data using the PCA (Principal Component Analysis) algorithm. Using the features of the PCA algorithm, it is applied to the text data and evaluated the results. Also, the second aim of this project is to improve the computational performance of the PCA technique. The algorithm uses the concepts of variance matrix, covariance matrix, eigenvector, and eigenvalues pairs to perform PCA, providing a set of eigenvectors and their respective eigenvalues as a result [5]. Calculating these matrices requires too much computational performance as you can see below in Figure 1.1 and Figure 1.2;

![](RackMultipart20201001-4-17socgu_html_98e6c372c86f3517.png)

Figure1.1 The calculation of eigenvalues is calculated as the sample size is fixed and the feature size is increased.

![](RackMultipart20201001-4-17socgu_html_834fc4a241387e57.png)

Figure 1.2 The calculation of eigenvalues is calculated as the feature size is fixed and the sample size is increased.

This project will show how the PCA algorithm will be improved. Simple examples are given to understand the subject and python is used to code the main application. Principal Component Analysis is expressed as PCA throughout the document. However, PCA is a statistical method widely used in areas such as face recognition, image compression, feature extraction, and pattern recognition. Before going into the details about PCA, the mathematical basis was mentioned to understand the issue more easily. Standard deviation, covariance, eigenvalue-eigenvector calculations are among the subjects that should be examined to understand PCA. After explaining the mathematical baseline, PCA will be explained in detail with examples and applied in text data.

In Section 2, the mathematical expression which is important for PCA is discussed, whereas, in Section 3, the PCA is described comprehensively. In Section 4, the application of PCA is described in detail, and Section 5, the new approach is proposed, and several examples are done in this way. Lastly, in Section 6, this article is concluded.

1. **MATHEMATICS**

In this section, some mathematical expressions required for understanding of PCA will be mentioned.

**2.1 Standard Deviation**

Standard deviation is that &quot;the average distance from the mean of the data set to a point&quot; [2]. The standard deviation is the square root of the variance. In a more mathematical expression, the standard deviation is the square root of the sum of the squares of the numbers in a series from the arithmetic mean of the series divided by the number of elements of the array.

(2.1.1)

As you can see from (2.1.1), standard deviation measures the distribution of data and it works for only one-dimension. However, in general, datasets have more than one dimension, and the main purpose of statistical analysis is to understand if there is a relationship between dimensions or not.

**2.2 Variance**

Variance is another concept that gives information about the distribution of data. It is often used to measure change.

(2.2.2)

Variance is the square of the standard deviation.

(2.2.3)

However, standard deviation and variance are used for one-dimensional data as you can see from (2.2.2) and (2.2.3).

**2.3 Covariance**

Covariance means changing together and used to analyze the direction of the relationship between variables. Standard deviation and variance are used for one-dimensional data. However, most of the time data sets have more than one size. Covariance is always used to measure between two dimensions.

(2.3.1)

The covariance determines the direction between the two variables as you can see from the (2.3.1). It can be +, - or 0. There is no upper or lower limit for the calculated covariance value, it completely depends on the values in the dataset.

**2.4**** Covariance Matrix**

PCA calculations need to measure how much the dimensions vary concerning each other. If there is a data set with more than 2 dimensions, this indicates that there will be more than one covariance calculation. When looking at more than two variables, the covariance matrix is used. The diagonal values in the covariance matrix are equal to the variance values of the variables.

(2.4.1)

The covariance matrix as (2.4.1) exhibits a symmetrical structure due to the feature.

**2.5 Eigenvector and Eigenvalue**

The matrix applied to a vector can change both the magnitude and direction of that vector. However, when a matrix applied to some vectors, it multiplies its size by a factor. So, it only changes its size, not its direction. These vectors, whose direction does not change, are expressed as eigenvectors of the matrix in question. Eigenvectors can only be obtained from square matrices. Therefore, covariance matrices are used to obtain an eigenvalue and eigenvector. However, not every square matrix has eigenvectors.

1. **PRINCIPAL COMPONENT ANALYSIS (PCA)**

The principal components approach is used to eliminate the dependency structure and reduce the size. It is a multivariate statistical method that enables recognition, classification, and size reduction. This approach tries to find the strongest pattern in the data. Therefore, it can be used as a pattern-finding technique. Often the variety of data can be captured with a small set selected from the entire dataset. The PCA Algorithm has the following features:

1. Decrease the dimension of dataset
2. Compress data by minimizing loss
3. Use of machine learning in supervised learning
4. It allows us to understand the structure of the data by reducing the multidimensional data to 2 or 3 dimensions and making visual meaning [4].

When PCA is applied, the true size of p-dimension is determined. This true dimension is called principal components. The principal components have three features:

1. The first principal component is the variable that most explains the total variability.
2. The next principal component is the variable that most closely describes the remaining variability.

Generally, the relationships in the data can be explained by looking at the multidimensional data from the right angle. The purpose of PCA finds this &quot;right angle&quot;. The appropriate coordinate system with PCA is sought as follows:

1. As the 1st axis, the direction that is the largest change of data is selected.
2. As the 2nd axis, the direction that is perpendicular to the 1st axis and the largest change in the data is selected.
3. As the 3rd axis, the direction that is perpendicular to the 1st and 2nd axis and the largest change of the remaining data is selected.
4. Always as the new axis, the direction that is the largest remaining change in its data is selected.

![](RackMultipart20201001-4-17socgu_html_252302a9065a4383.png)
 Figure 3.1 Principal component analysis explained simply by Linh Ngo(2018)[6].

As can be seen from Figure 3.1, the major change directions that are selected vertical are called principal components. PCA aspects first indicate the direction that contributes most to the exchange of data, and then explains the aspects that contribute less. The variance values retained are used to indicate the sufficient number of principal components. The total variance of the first principle components to be used should be 90% to 95% of the total variance of the original data. To represent the original data with 95% accuracy, a 10-20% PCA component may be sufficient. For example, instead of saving all 1000 features, 10-20% of the first principle component can be saved and stored on average (because they do not change more or less) for the values of other components to save 1000 data. Original data can be saved with 1-2% memory.

![](RackMultipart20201001-4-17socgu_html_3371589adc3d8cef.png)

Figure 3.2 After selecting the components with the principal component analysis, the object is described with the new variables [7].

As a result:

  1. PCA is a very useful method in size reduction.
  2. PCA represents multidimensional data approximately and with less dimensional data.
  3. PCA locates the largest variance directions perpendicular to the original data and displays the original data in this coordinate system.
  4. PCA can be used for visual display and analysis of multidimensional data.
  5. PCA can reduce the size of data.
  6. PCA can also be used for data compression.

1. **APPLICATION OF PCA**

In this part of the study, PCA will be applied to a text dataset and what will be done will be mentioned. So, text mining is a method for drawing interesting and meaningful patterns to explore actionable information from textual data sources [9]. The unstructured / semi-structured and noisy nature of text data makes it more difficult for machine learning methods to operate directly on raw data. Text mining techniques are used actively in business, academia, web applications, the internet, and other industries [10].

Transforming free-flowing text into some numerical representations which can then be understood by algorithms of machine learning is needed. To transform text data into a structured form (vectors) we need hundreds of thousands of features. Text is performed by a numeric vector acquired by checking the most significant lexical things. Then, the events of the most significant words are counted to represent each document by a vector regarding space. The first consideration is here using a minimum number of vectors without loss of information.

**4.1 Preparation of Dataset**

The dataset is taken from Kaggle and consists of the data of people on Wikipedia [8]. It has 3 columns which are URIs, names, and text. Columns have 42786 values. Firstly, the dataset has been cleared. This is mainly based on the identification of missing, incorrect, or irrelevant parts of the data, followed by the modification or deletion of these parts. This was done by changing special characters, white spaces, spaces, etc. You can see the code snippet below;

| def standardize\_text(dataframe, text):dataframe[text] = dataframe[text].str.replace(r&quot;http\S+&quot;, &quot;&quot;)dataframe[text] = dataframe[text].str.replace(r&quot;http&quot;, &quot;&quot;)dataframe[text] = dataframe[text].str.replace(r&quot;@\S+&quot;, &quot;&quot;)dataframe[text] = dataframe[text].str.replace(r&quot;[^A-Za-z0-9(),!?@\&#39;\`\&quot;\_\n]&quot;, &quot; &quot;)dataframe[text] = dataframe[text].str.replace(r&quot;@&quot;, &quot;at&quot;)dataframe[text] = dataframe[text].str.lower()return dataframedataframe= standardize\_text(dataframe, &quot;text&quot;) |
| --- |

However, a clean dataset will give you meaningful features and avoided irrelevant noise. After the pre-processing step, a technique that creates a vector representation of each document should be implemented. This move, therefore, reduces the difficulty of the documents and promotes their handling. Each word has a weight assigned based on the number of times it appears in the text. That process is known as a weighting of the word. There are many ways to measure weight, such as word frequency weighting, TFIDF, boolean weighting, or entropy. A widely used approach for term weighting is the TF-IDF (term frequency-inverse frequency of documents), which is an algorithm that has been used to figure out the similarity within the document. The value increases proportionally with the number of times that a word appears in the text, but is offset by the word frequency in the corpus [11]. For calculating the TF-IDF weight of a term: term frequency (TF(t,d)) which is the number that the word &quot;t&quot; occurred in document &quot;d&quot;, document frequency (DF(t)) is number of documents in which the term &quot;t&quot; occur at least once and inverse document frequency (IDF) that can be calculated from document frequency using the following formula in (4.1.1);

(4.1.1)

The measure of word important can be calculated by using the product of the term frequency and the inverse document frequency (TF IDF) [12].

The main idea of text document dimension reduction is presenting the minimum size of data without loss of information. PCA can remove the noisy information from the original document thanks to its properties while decreasing and preserving the relevant data. Representing the minimum size of the data without loss of information demonstrates that the documents using PCA helps to control time and receive results [13]. To reduce the dimension of a text document the data is represented by a matrix. With applying PCA, the text data is represented a matrix X (m\*n) containing the weights of terms in documents as mentioned before.

**4.2 Subtracting from Average**

To work with PCA, it is necessary to subtract each dimension of the data set from its average. You can see the code snippet below;

| df = np.transpose(df)data\_mean = np.mean(df)data\_center = df - data\_mean |
| --- |

  1. **Covariance Matrix**

The covariance matrix is used to obtain eigenvalues and eigenvectors. You can see the code snippet below;

| cov\_matrix = np.cov(data\_center) |
| --- |

After calculating the covariance matrix, eigenvectors and eigenvalues can be obtained from this matrix.

**4.4 Eigenvector and Eigenvalue**

The covariance matrix is a square matrix and eigenvectors and eigenvalues can be obtained from this matrix. The important point here is the information given by this matrix about the data set. You can see the code snippet below;

| eigenval, eigenvec = eigs(cov\_matrix, k=200) |
| --- |

However, in this part of the study, the duration of the eigenvector and covariance matrix calculations were measured.

![](RackMultipart20201001-4-17socgu_html_8f05e7a0048a89ef.png)

Figure 4.4.1 When the number of features increases, the calculation time increases significantly.

As can be seen from Figure 4.4.1, it can be observed that as the number of features increases, the calculation time increases. Here the blue curve shows the time it takes to calculate all eigenvalues. The green curve also belongs to the first 200 eigenvalues calculation time. According to this; it takes a lot of time to calculate all of them, it takes less time to calculate some, but we are trying to figure out how much of it is necessary for us. Even 200 eigenvector calculations are increasing dramatically as a serious computation time. The calculation of eigenvectors at a larger size was not even possible for a computer with normal features. The purpose of the project, as can be clearly understood from here, is to make these calculations efficiently on any computer. So, a method is being developed that can predict how many eigenvectors we need to calculate.

![](RackMultipart20201001-4-17socgu_html_542e6abed18aa99b.png)

Figure 4.4.2 Distribution of eigenvalues is shown.

Figure 4.4.2 shows the distribution of eigenvectors. By drawing a graph of the eigenvalue against the number of factors, the point of the elbow before the breakpoint where the slope changes a lot is determined as the number of factors.

**5 A NEW APPROACH TO IMPROVE PCA**

The main aim of this project is the section that tries to minimize the eigenvalue and covariance matrix calculation time. According to Hutchinson, instead of calculating all eigenvalues, calculating the diagonal matrix can achieve sufficient accuracy [14]. So, with this information, we have developed a new calculation method. Let&#39;s consider as the data matrix. The covariance matrix is C = . So; the result of the product of ( becomes . At this point, we create a random V vector with a size . Thus, according to Hutchinson, by using the total variance estimation method we calculate the formula with V vectors that are generated randomly k times. By expanding this formula, we can eliminate the long covariance matrix calculation as follows: . Besides, we know that . So, we can arrange the formula as follows: . We can give a value to the vector XV, so that y = XV. This allows us to transform our formula in this way; . You can see the mentioned formula transformed into a code snippet below;

| k=1for j in range(10):total = 0for i in range(k):v = np.random.randn(20000,1)y=np.dot(df\_k,v)y\_t = np.transpose(y)total = total + np.trace(np.dot(y,y\_t))k+=1
 |
| --- |

In the next part of the article, examples were made to better understand the results of this method.

**5.1 A New Approach to Reduce Calculation Times– Python Code**

|
**#Importing libraries** import pandas as pdimport timeimport numpy as npfrom sklearn.feature\_extraction.text import CountVectorizer,TfidfVectorizerimport timefrom scipy.sparse.linalg import eigsimport csvdataset = &#39;people\_wiki.csv&#39;people\_df = pd.read\_csv(dataset)

tfidf\_vec = TfidfVectorizer(max\_features=20000,min\_df=5, max\_df = 0.8, sublinear\_tf=True, use\_idf =True, stop\_words = &#39;english&#39;)df = tfidf\_vec.fit\_transform(people\_df.text.values).toarray()df\_k = df[:30000] #select first # rows
**#FIRST SOLUTION**
f\_code\_start = time.time()df\_t = np.transpose(df\_k)a = np.dot(df\_t,df\_k)first\_trace = np.trace(a)print(&quot;First code trace: &quot;,first\_trace)f\_code\_end = time.time()print(&quot;First code execution time: &quot;,f\_code\_end - f\_code\_start)
**#SECOND SOLUTION**
k=100total\_2 = 0f\_code\_start = time.time()for i in range(50):v = np.random.randn(20000,1)y=np.dot(df\_k,v)y\_t = np.transpose(y)total\_2 = total\_2 + np.dot(y\_t,y)total\_2 = total\_2/50f\_code\_end = time.time()print(&quot;Time : &quot;, f\_code\_end - f\_code\_start)while True:s\_code\_start = time.time()eigenval, eigenvec = eigs(a,k)sum\_eig= np.sum(eigenval)print(sum\_eig/(total\_2)\*100)s\_code\_end = time.time()print(&quot;Time : &quot;,s\_code\_end - s\_code\_start)print(&quot;K value is: &quot;, k)print(&quot;------------------------------&quot;)if sum\_eig/(total\_2)\*100 \&gt; 80:breakelse :k+=100 |
| --- |

**5.2 A New Approach to Reduce Calculation Times– Flowchart**

![](RackMultipart20201001-4-17socgu_html_da4661756cecffba.png)

Figure 5.2.1 It is a flowchart that shows the algorithm. The for loop is shown on the left and the while loop on the right.

  1. **Proposed Method with Hand-Written Digits Data**

Since this study is an experimental study, it has been observed that the developed method works much faster in image data. Because the eigenvalues are very close to each other in the text data and parsing it does not yield an efficient result.

![](RackMultipart20201001-4-17socgu_html_eb12923147c2e5f1.png)

Figure 5.1 The summation of the largest k eigenvalues ​​is calculated and shown how close it is to the estimated total variance.

In Figure 5.1, the blue line shows the values predicting the trace, while the green line shows the time elapsed when calculating them. The red line shows the real trace, which is a scaler value, while the yellow line shows the time it takes to calculate it. As you can see from this example, our method is not efficient in text data.

However, in the image data, it is enough to look at some of the pixels to process the image. For example, background pixels can be eliminated if appropriate. Firstly, this application will be performed in a hand-written digits dataset which is a small dataset [15,16]. Afterward, the larger dataset which is an MNIST dataset will be used [17]. Thus, it will be seen how this study works effectively.

Firstly, the necessary libraries and dataset are loaded. First, a visualization example is run to show how PCA works in noisy data. By taking only the largest principal components, nosy data is considerably reduced.

|
import pandas as pdimport timeimport numpy as npimport timefrom scipy.sparse.linalg import eigsimport matplotlib.pyplot as pltfrom sklearn.datasets import load\_digitsimport seaborn as snsfrom sklearn.decomposition import PCA
digits = load\_digits()df\_k = digits.data
def plot\_data(data):fig, axes = plt.subplots(4, 10, figsize=(10, 4),sub\_k={&#39;xticks&#39;:[], &#39;yticks&#39;:[]},grid\_k=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):ax.imshow(data[i].reshape(8, 8),cmap=&#39;binary&#39;, interpolation=&#39;nearest&#39;,clim=(0, 16))
plt.show(plot\_data(digits.data))
 |
| --- |

Its output is as follows;

![](RackMultipart20201001-4-17socgu_html_3a64c46da6316a5.png)

Figure 5.3.1 This is how to look hand-written digits data as noisy data [15,16].

After seeing how to actual data is in the Figure 5.3.1, our specially developed formula and algorithm are applied here. In this example, we found the components that work for us by randomly starting with 2 components(k value) and increasing them 2 times.

|
df\_t = np.transpose(df\_k)a = np.dot(df\_t,df\_k)first\_trace = np.trace(a)print(&quot;First code trace: &quot;,first\_trace)
k=2n=len(a)print(n)j=50target = 80total\_2 = 0for i in range(j):v = np.random.randn(n,1)y=np.dot(df\_k,v)y\_t = np.transpose(y)total\_2 = total\_2 + np.dot(y\_t,y)total\_2 = total\_2/jwhile True:eigenval, eigenvec = eigs(a,k)sum\_eig= np.sum(eigenval)print(sum\_eig/(total\_2)\*100) print(&quot;K value is: &quot;, k)print(&quot;--------&quot;)if sum\_eig/(total\_2)\*100 \&gt; target:breakelse :k+=2 |
| --- |

Thus, instead of the calculation of all eigenvalue which takes quite a long time, our new method which does this work in a very short time calculated the only required eigenvalue values and shortened the calculation time using only first a few principal components. As seen below, our program has seen that only the first 4 components are sufficient and have stopped. The level of satisfactoriness expressed as target value in the code we set is 80% of the trace value. However, in the original example, it is argued that 20 components will be sufficient automatically, but the method we developed has achieved a success rate of 80% and using only 4 components. You can see the output as below;

|
Estimated Trace: 64Execution Time for Trace Estimation:  0.00792694091796875------------------------------Summation of Largest k Eigenvalues: 51,03443347136Execution Time for Eigenvalue Calculation:  0.0037245750427246094Largest k Eigenvalues: 2------------------------------Summation of Largest k Eigenvalues: 56,4841148224Execution Time for Eigenvalue Calculation:  0.0021538734436035156Largest k Eigenvalues: 4------------------------------ |
| --- |

Afterward, the application sample taken as reference continued as follows;

|
def plot\_pca (x, coefficients=None, mean=0, components=None,imshape=(8, 8), n\_comp=k, fontsize=12,show\_mean=True):if coefficients is None:coefficients = xif components is None:components = np.eye(len(coefficients), len(x))mean = np.zeros\_like(x) + mean
fig = plt.figure(figsize=(1.2 \* (5 + n\_comp), 1.2 \* 2))g = plt.GridSpec(2, 4 + bool(show\_mean) + n\_comp, hspace=0.3)
def show(i, j, x, title=None):ax = fig.add\_subplot(g[i, j], xticks=[], yticks=[])ax.imshow(x.reshape(imshape), interpolation=&#39;nearest&#39;)if title:ax.set\_title(title, fontsize=fontsize)
show(slice(2), slice(2), x, &quot;True&quot;)approx = mean.copy()counter = 2if show\_mean:show(0, 2, np.zeros\_like(x) + mean, r&#39;$\mu$&#39;)show(1, 2, approx, r&#39;$1 \cdot \mu$&#39;)counter += 1
for i in range(n\_components):approx = approx + coefficients[i] \* components[i]show(0, i + counter, components[i], r&#39;$c\_{0}$&#39;.format(i + 1))show(1, i + counter, approx,r&quot;${0:.2f} \cdot c\_{1}$&quot;.format(coefficients[i], i + 1))if show\_mean or i \&gt; 0:plt.gca().text(0, 1.05, &#39;$+$&#39;, ha=&#39;right&#39;, va=&#39;bottom&#39;,transform=plt.gca().transAxes, fontsize=fontsize)
show(slice(2), slice(-2, None), approx, &quot;Approx&quot;)return fig
pca = PCA(n\_comp=k)X\_proj = pca.fit\_transform(digits.data)sns.set\_style(&#39;white&#39;)fig = plot\_pca(digits.data[10], X\_proj[10],pca.mean\_, pca.components\_)
fig.savefig(&#39;figure&#39;)plt.show(fig)
 |
| --- |

According to all these, the result is as follows;

![](RackMultipart20201001-4-17socgu_html_821227f7db486079.png)

Figure 5.3.2 The true value of 0(left) and what the program found(right) has shown approximately.

Figure 5.3.2 created by our program giving 4 components with 80% accuracy. Accordingly, the true value on the left and the image obtained with 4 components on the right are shown.

  1. **Proposed Method With MNIST Data**

After applying in the previous dataset, which is a hand-written digits dataset, the algorithm was applied in a larger dataset. MNIST dataset was used for this [17]. This dataset contains 784 features. The first column is the label column and the other columns belong to pixel values. Each pixel shows the lighting of the data, which means that while the pixels increasing, the lighting is getting darker.

However, firstly the necessary libraries and dataset are loaded.

|
import pandas as pd import csvimport matplotlib.image as mpimgimport timeimport numpy as npfrom scipy.sparse.linalg import eigsfrom PIL import Imageimport seaborn as snsfrom sklearn.decomposition import PCA import matplotlib.pyplot as plt


d0 = pd.read\_csv(&quot;mnist\_data.csv&quot;)l = d0[&#39;label&#39;]
d = d0.drop(&quot;label&quot;, axis=1)d.head()
 |
| --- |

The size of the dataset is (42000, 784). Instead of calculating 784 features, the program will be reduced the size of the feature by estimating the trace value. To show the MNIST data, an example suggests that a number can be displayed from the dataset [17].

|
plt.figure(figsize=(7,7))idx = 100
gridata = d.iloc[idx].as\_matrix().reshape(28,28) pixel arrayplt.imshow(gridata, interpolation = &quot;none&quot;, cmap = &quot;gray&quot;)plt.show()
print(l[idx])print(d.shape)print(l.shape)
from sklearn.preprocessing import StandardScalerstand\_data = StandardScaler().fit\_transform(d)
df\_k=stand\_data

 |
| --- |

The output of this code as follows in Figure 5.4.1;

![](RackMultipart20201001-4-17socgu_html_d4d97816a90a44fb.png)

Figure 5.4.1 Displaying a number from MNIST data.

After seeing how to actual data look like in the Figure 5.4.1, our specially developed formula and algorithm are applied here. In this example, we found the components that work for us by randomly starting with 10 components(k value) and increasing them 5 times.

|
f\_code\_start=time.time()df\_t = np.transpose(df\_k)a = np.dot(df\_t,df\_k)first\_trace = np.trace(a)print(&quot;First code trace: &quot;,first\_trace)f\_code\_end = time.time()print(&quot;First code execution time: &quot;,f\_code\_end - f\_code\_start)

k=10 **#component number** n=len(a)j=50target = 85total\_2 = 0for\_code\_start = time.time()for i in range(j):v = np.random.randn(n,1)y=np.dot(df\_k,v)y\_t = np.transpose(y)total\_2 = total\_2 + np.dot(y\_t,y)total\_2 = total\_2/jprint(&quot;Total 2 : &quot;,total\_2)for\_code\_end = time.time()print(&quot;Time : &quot;, for\_code\_end - for\_code\_start)while True:s\_code\_start = time.time()eigenval, eigenvec = eigs(a,k)sum\_eig= np.sum(eigenval)percet = sum\_eig/(total\_2)\*100print(percet)s\_code\_end = time.time()total\_time = s\_code\_end - s\_code\_startprint(&quot;Time : &quot;,total\_time)print(&quot;K value is: &quot;, k)print(&quot;------------------------------&quot;)output = [k,sum\_eig,total\_time]csv.writerow(output)if sum\_eig/(total\_2)\*100 \&gt; target:breakelse :k+=5 |
| --- |

The output can be seen from Figure 5.4.2 as follows;

![](RackMultipart20201001-4-17socgu_html_edff20087a2c0ee9.png)

Figure 5.4.2 Output of estimated trace calculation time.

In the Figure 5.4.2, trace number and trace calculation time are a scalar value. It shows that with the developed algorithm, 165 - k values gave 85% accuracy. The visualization of this output can be seen in Figure 5.4.3.

![](RackMultipart20201001-4-17socgu_html_e6b70c04b296b283.png)

Figure 5.4.3 Visualization of estimated trace calculation time.

In Figure 5.4.3, the trace can be seen on the red line and the time spent calculating it on the yellow line. The green line shows the estimate of the trace value and the blue line shows the time elapsed while calculating it. After this point, the example continues as suggested, you can see below [17];

|
from sklearn import decompositionpca = decomposition.PCA()
pca.n\_components = kpca\_data = pca.fit\_transform(df\_k)
print(&quot;shape of pca\_reduced.shape = &quot;, pca\_data.shape)

pca.n\_components = 784pca\_data = pca.fit\_transform(df\_k)
per\_variable = pca.explained\_variance\_ / np.sum(pca.explained\_variance\_)
variable\_explained = np.cumsum(per\_variable)
fig = plt.figure(1, figsize=(6, 4))
plt.title(&#39;Cumulative Variable Explained Variance&#39;)plt.plot(variable\_explained, linewidth=2)plt.axis(&#39;tight&#39;)plt.grid()plt.xlabel(&#39;k value&#39;)plt.ylabel(&#39;Cumulative Explained Variance&#39;)plt.show()
 |
| --- |

The output can be seen from Figure 5.4.3 as follows;

![](RackMultipart20201001-4-17socgu_html_36ed9bff5afaef84.png)

Figure 5.4.3 The distribution of explained variance for k component.

The Figure 5.4.3 show that how explained variance distributes k times. It shows that if we take 165 k components, approximately 85% variance is explained.

However, In the reference sample, 90% success was achieved with a value of 200 k. Thus, it can be observed that there is a nice improvement with the developed method.

1. **CONCLUSION**

In conclusion, this research study&#39;s main purpose is to improve the computational performance of the PCA algorithm by developing a new approach. First, it is important to know that PCA is a dimension reduction algorithm and uses the concepts of variance matrix, covariance matrix, eigenvector, and eigenvalues pairs to perform [1]. When working with high dimension datasets time and resources are taken as a consideration. Here, in this research main motivation is avoiding waste of time and resources. To demonstrate high computational performance, the calculation time of covariance matrix, and the calculation time of eigenvalues have been shown in graphs. It has been observed that, as features size increases calculation time increases simultaneously, and calculation of eigenvalues by trial-and-error-takes a lot of time. Also, there is no way to apply these calculations in standard computers. As known that principal components(eigenvalues) are the variable that most explains the total variability, thus, the calculation of all eigenvalues is not necessary. When developing a new approach, finding how much of eigenvalues is necessary will be taken to approach total variance was aimed. To achieve this, the total variance estimation method has been used by creating a random vector, V, to multiply a matrix, C (n x n), by the left and the right side instead of the C = XT X computation which takes a lot of time. After the new approach has been established, it was applied to the various datasets which are text data and image data with different sizes. First of all, when working in a text dataset it has been seen that in large datasets eigenvalues are very small and close to each other and it is considered a bottleneck in this research. Besides this, instead of the calculation of all eigenvalues, which takes considerable time, the new method works in a very short time calculates the necessary eigenvalues. As a result, to improve the computational performance of the PCA size of the reduction will be automatically found in effectively.

**REFERENCES**

1. Abdi, H., &amp; Williams, L. J. (2010). Principal component analysis. _Wiley interdisciplinary reviews: computational statistics_, _2_(4), 433-459.
2. Smith, L. I. (2002). _A tutorial on principal components analysis_.
3. Wold, S., Esbensen, K., &amp; Geladi, P. (1987). Principal component analysis. _Chemometrics and intelligent laboratory systems_, _2_(1-3), 37-52.
4. Ringnér, M. (2008). What is principal component analysis? _Nature biotechnology_, _26_(3), 303-304.
5. Tipping, M. E. (2001). Sparse kernel principal component analysis. In _Advances in neural information processing systems_(pp. 633-639).
6. Ngo, L., (2018), Principal component analysis explained simply. Retrieved from [https://blog.bioturing.com/2018/06/14/principal-component-analysis-explained-simply/](https://blog.bioturing.com/2018/06/14/principal-component-analysis-explained-simply/)
7. Meng., (2013), An intuitive explanation of PCA (Principal Component Analysis). Retrieved from [http://mengnote.blogspot.com/2013/05/an-intuitive-explanation-of-pca.html](http://mengnote.blogspot.com/2013/05/an-intuitive-explanation-of-pca.html)
8. Sameer Mahajan.(2018). People Wikipedia Data (Version 1) [Data file]. Retrieved from [https://www.kaggle.com/sameersmahajan/people-wikipedia-data](https://www.kaggle.com/sameersmahajan/people-wikipedia-data)
9. W. Fan, L. Wallace, S. Rich, and Z. Zhang, &quot;Tapping the power of text

mining,&quot; _Communications of the ACM_, vol. 49, no. 9, pp. 76–82, 2006.

1. S.-H. Liao, P.-H. Chu, and P.-Y. Hsiao, &quot;Data mining techniques and

applications–a decade review from 2000 to 2011,&quot; _Expert Systems with_

_Applications_, vol. 39, no. 12, pp. 11 303–11 311, 2012.

1. Rajaraman, A.; Ullman, J.D. (2011). &quot;Data Mining&quot; (PDF). _Mining of Massive Datasets_. pp. 1–17. doi:10.1017/CBO9781139058452.002. ISBN 978-1-139-05845-2.
2. Taloba, A. I., Eisa, D. A., &amp; Ismail, S. S. (2018). A Comparative Study on using Principal Component Analysis with Different Text Classifiers. _arXiv preprint arXiv:1807.03283_.
3. Jaffali, Soufiene &amp; Jamoussi, Salma. (2012). Text document dimension reduction using Principal Component Analysis.
4. Di Napoli, E., Polizzi, E., &amp; Saad, Y. (2016). Efficient estimation of eigenvalue counts in an interval. _Numerical Linear Algebra with Applications_, _23_(4), 674-692.
5. sklearn.datasets.load\_digits. (n.d.). Retrieved from [https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load\_digits.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
6. VanderPlas, J. (n.d.). In Depth: Principal Component Analysis. Retrieved from [https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html)
7. &quot;Digit Recognizer.&quot; _Kaggle_, www.kaggle.com/c/digit-recognizer/data.
