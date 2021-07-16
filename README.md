# Credit_Risk_Analysis

## Purpose
The purpose of this repository is to analyze credit card risk data from Lending Club for Q1 2019 to optimize predictions for high versus low risk loans using different trainings for supervised machine learning linear regression models. In our analysis, we use six different ML models including: random oversampling, SMOTE (Synthetic Minority Oversampling Technique) oversampling, cluster centroid undersampling, SMOTEENN (SMOTE and Edited Nearest Neighbors) combination sampling, balanced random forest classifier, and easy ensemble AdaBoost classifier.

## Resources
Data: LoanStats_2019Q1.csv
Software: jupyter notebook, python 3.7, Pandas, Collections, Numpy, sklearn, imblearn


## Analysis
Using `imblearn` and `sklearn` module we built a series of supervised ML models to assess and predict loan risk. Then we evaluate them to find the best fitting model.

### Results
**Note**: In the confusion matrix tables: 
- "0" implies high-risk loan
- "1" implies low-risk loan 

1. **Naive Random Oversampling model**
Model Parameters:
- Model: imblearn RandomOverSampler
- Random State: 1

Model Results:
- Balanced Acurracy Score: 64%

- **Picture 1.1: Naive Random Oversampling Confusion Matrix**
** insert random oversampling confusion matrix **

- **Picture 1.2: Naive Random Oversampling Classification Report**
** insert random oversampling classification report **


2. **SMOTE (Synthetic Minority Oversampling Technique) Oversampling model**
Model Parameters:
- Model: imblearn SMOTE
- Random State: 1

Model Results:
- Balanced Acurracy Score: 66%

- **Picture 2.1: SMOTE Sampling Confusion Matrix**
** insert SMOTE confusion matrix **

- **Picture 2.2: SMOTE Sampling Classification Report**
** insert SMOTE classification report **


3. **Cluster Centroids Undersampling model**
Model Parameters:
- Model: imblearn ClusterCentroids
- Random State: 1

Model Results:
- Balanced Acurracy Score: 54%

- **Picture 3.1: Cluster Centroids Sampling Confusion Matrix**
** insert Cluster Centroids confusion matrix **

- **Picture 3.2: Cluster Centroids Sampling Classification Report**
** insert Cluster Centroids classification report **


4. **SMOTEENN (SMOTE and Edited Nearest Neighbors) Combination Sampling model**
Model Parameters:
- Model: imblearn SMOTEENN
- Random State: 1

Model Results:
- Balanced Acurracy Score: 67%

- **Picture 4.1: SMOTEENN Sampling Confusion Matrix**
** insert Cluster Centroids confusion matrix **

- **Picture 4.2: SMOTEENN Sampling Classification Report**
** insert Cluster Centroids classification report **


5. **Balanced Random Forest Classifier model**
Model Parameters:
- Model: imblearn BalanceRandomForestClassifier
- Random State: 1

Model Results:
- Balanced Acurracy Score: 79%

- **Picture 5.1: Balanced Random Forest Classifier Confusion Matrix**
** insert Balanced Random Forest confusion matrix **

- **Picture 5.2: Balanced Random Forest Classifier Report**
** insert Balanced Random Forest classification report **


6. **Easy Ensemble AdaBoost Classifier model**
Model Parameters:
- Model: imblearn EasyEnsembleClassifier
- Random State: 1

Model Results:
- Balanced Acurracy Score: 93%

- **Picture 6.1: Easy Ensemble AdaBoost Classifier Confusion Matrix**
** insert Easy Ensemble AdaBoost confusion matrix **

- **Picture 6.2: Easy Ensemble AdaBoost Classifier Report**
** insert Easy Ensemble AdaBoost classification report **


### Conclusions


## Limitations of current script/ future improvements
- To tune the models to produce more accurate predictions, we can add scaling to some of the data to eliminate more of the modeling variability using the `StandardScaler()` module.

- Using the Balanced Random Forest Classifier feature importances ranking list we can start to eliminate columns that may be less impactful on modeling predictions and therefore reduce the variability of the modeling.

- Additionally, can add more estimators or change the type of model fit to something other than linear regression to see if it improves the model's accuracy.

