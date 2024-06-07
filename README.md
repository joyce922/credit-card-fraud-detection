# Credit card fraud detection

The dataset used is from kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud

<b>GOAL:</b> 
Detecting whether a transaction in a is a  normal payment or fraud.

<b>Introduction:</b>
<p> Fraud detection is a big challange for the credit card companies. This is because of the huge volume of transactions that are completed each day and many fraudulent transactions look a lot like normal transactions.
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers feel safe and are not charged for items that they did not purchase.</p>

<p>The dataset for this project is taken from Kaggle. This data will be used to train ML models. Further ahead the performance of these models will be evaluated.<p>

<b>DataSet:</b>  
    
<p> The dataset contans around 280,000 transactions carried out by Eurpoean cardholders in September 2013 spanning over two days. The dataset is highly unbalanced with fraud transaction constuitution barely 0.172% of all transactions. <p>
    
<p> Due to the confidentiality issues the multidimensional dataset is transformed in to its principal components using PCA. Attributes V1..V28 are the top 28 principal components which are provided in the dataset. In addition to that there are two other input columns - the transaction 'Amount' and 'Time' which contains the time elapsed since the first transaction in seconds. The outfut feature 'Class' is the last column which assumes binary value 1 or 0 depending on either the transaction if fradulant or genuine. <p>
    
<p> Dealing with the imbalanced datasets will give a high accuracy we will evaluate the models by using f1-score, precision/recall score and Roc_curv and confusion matrix<p>   
    
<b>PROBLEM STATEMENT:</b> 
The objective of this project is to design an ML binary classifier which can successfully identify the fraudulant vs non/fraudulant transactions. To build this classifier, dataset with over 280k transaction data is provided. 28 PCA transformed variables are provided which are the key features used for building the classifier. In addition to these, two additional input variables are the 'Time of transaction' and the 'Amount'. From the T-SNE multi-dimensional plot, it can seen that the classes are well separated. The dataset is heavily imbalanced with more than 98% transactions belonging to one class. Resampling is done to balance the classes and ML models are applied to this data. Performance of ML models is checked using the AUC\ROC curves and comparision between F1-score, Recall, Precision scores.


<b>ML ALGORITHMS:</b> 
    The algorithms used are
- Logistic Regression (LR)
- Random Forest (RF)
- Decision Tree (DT)
- Support Vector Machine
- K Nearest Classifier
- Gradient Bossting Technique
- Simple Neural Network

<b>STEPS:</b>
<ol>
<li>Loading and understanding the dataset</li>
<li>Preprocessing: Exploring the Data, Cleaning and Visualization</li>
<li>Baseline: Apply Machine learning models and evalulation through Cross validation</li>
<li>Splitting the dataset & scaling after splitting</li> 
<li>Build a Neural Network, fitting the data and evalution before sampling</li> 
<li>Handling imbalanced Dataset (undersamling: near miss, ... & oversamling: SMOTE, ...)</li>
<li>Apply machine learning models after sampling: Fitting and utilization of Machine learning algorithms</li>
<li>Apply a Neural Network, fitting the data and evalution After sampling</li> 
<li>Evaluating the Models</li>
<li>A comparison of models accuracy</li>

<b>REFERENCES:</b>
1. Account Number Verification Service - A quick method to verify accounts. VISA, 2016, https://usa.visa.com/dam/VCOM/global/support-legal/documents/acct-numb-verif-service-a-quick-method-to-verify-accounts-vbs-07-jun-16.pdf
2. pandas.DataFrame.corr. the pandas development team, 2021, https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html (accessed 22.11.2021)
3. sklearn.manifold.TSNE. scikit-learn developers, 2021, https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html (accessed 22.11.2021)
4. Géron, Aurélien. Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow : Concepts, Tools, and Techniques to Build Intelligent Systems. Second ed., O'Reilly, 2019.
5. Pedro Coelho, Luis, et al. Building Machine Learning Systems with Python. Packt Publishing, Limited, 2018.
6. Decision Trees. scikit-learn developers, 2021 https://scikit-learn.org/stable/modules/tree.html (accessed 23.11.2021)
7. Undersampling Algorithms for Imbalanced Classification. Jason Brownlee, 2020, https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/ (accessed 23.11.2021)
8. Validation curves: plotting scores to evaluate models. https://scikit-learn.org/stable/modules/learning_curve.html
9. Learning Curves Explained with Python Sklearn Example. https://vitalflux.com/learning-curves-explained-python-sklearn-example/
