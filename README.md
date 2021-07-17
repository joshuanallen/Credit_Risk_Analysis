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

![Naive Random Oversampling Confusion Matrix](https://github.com/joshuanallen/Credit_Risk_Analysis/blob/f1a2843f416ec869d229ba43e0e0f44c0ceef03e/images/1.1_naive_random_oversampling_confusion_matrix.png)

- **Picture 1.2: Naive Random Oversampling Classification Report**

![Naive Random Oversampling Classification Report](https://github.com/joshuanallen/Credit_Risk_Analysis/blob/f1a2843f416ec869d229ba43e0e0f44c0ceef03e/images/1.2_naive_random_oversampling_classification_report.png)


2. **SMOTE (Synthetic Minority Oversampling Technique) Oversampling model**

Model Parameters:
- Model: imblearn SMOTE
- Random State: 1

Model Results:
- Balanced Acurracy Score: 66%

- **Picture 2.1: SMOTE Sampling Confusion Matrix**

![SMOTE Sampling Confusion Matrix](https://github.com/joshuanallen/Credit_Risk_Analysis/blob/f1a2843f416ec869d229ba43e0e0f44c0ceef03e/images/2.1_SMOTE_confusion_matrix.png)

- **Picture 2.2: SMOTE Sampling Classification Report**

![SMOTE Sampling Classification Report](https://github.com/joshuanallen/Credit_Risk_Analysis/blob/f1a2843f416ec869d229ba43e0e0f44c0ceef03e/images/2.2_SMOTE_Classification_report.png)


3. **Cluster Centroids Undersampling model**

Model Parameters:
- Model: imblearn ClusterCentroids
- Random State: 1

Model Results:
- Balanced Acurracy Score: 54%

- **Picture 3.1: Cluster Centroids Sampling Confusion Matrix**

![Cluster Centroids Sampling Confusion Matrix](https://github.com/joshuanallen/Credit_Risk_Analysis/blob/f1a2843f416ec869d229ba43e0e0f44c0ceef03e/images/3.1_Cluster_Centroid_Undersampling_confusion_matrix.png)

- **Picture 3.2: Cluster Centroids Sampling Classification Report**

![Cluster Centroids Sampling Classification Report](https://github.com/joshuanallen/Credit_Risk_Analysis/blob/f1a2843f416ec869d229ba43e0e0f44c0ceef03e/images/3.2_Cluster_Centroid_Undersampling_Classification_report.png)


4. **SMOTEENN (SMOTE and Edited Nearest Neighbors) Combination Sampling model**

Model Parameters:
- Model: imblearn SMOTEENN
- Random State: 1

Model Results:
- Balanced Acurracy Score: 67%

- **Picture 4.1: SMOTEENN Sampling Confusion Matrix**

![SMOTEENN Sampling Confusion Matrix](https://github.com/joshuanallen/Credit_Risk_Analysis/blob/f1a2843f416ec869d229ba43e0e0f44c0ceef03e/images/4.1_SMOTEENN_combination_confusion_matrix.png)

- **Picture 4.2: SMOTEENN Sampling Classification Report**

![SMOTEENN Sampling Classification Report](https://github.com/joshuanallen/Credit_Risk_Analysis/blob/f1a2843f416ec869d229ba43e0e0f44c0ceef03e/images/4.2_SMOTEENN_combination_classfication_report.png)


5. **Balanced Random Forest Classifier model**

Model Parameters:
- Model: imblearn BalanceRandomForestClassifier
- Random State: 1

Model Results:
- Balanced Acurracy Score: 79%

- **Picture 5.1: Balanced Random Forest Classifier Confusion Matrix**

![Balanced Random Forest Classifier Confusion Matrix](https://github.com/joshuanallen/Credit_Risk_Analysis/blob/f1a2843f416ec869d229ba43e0e0f44c0ceef03e/images/5.1_Balanced_Random_Forest_Classifier_Confusion_Matrix.png)

- **Picture 5.2: Balanced Random Forest Classifier Classification Report**

![Balanced Random Forest Classifier Classification Report](https://github.com/joshuanallen/Credit_Risk_Analysis/blob/f1a2843f416ec869d229ba43e0e0f44c0ceef03e/images/5.2_Balanced_Random_Forest_Classifier_Classification_Report.png)

6. **Easy Ensemble AdaBoost Classifier model**

Model Parameters:
- Model: imblearn EasyEnsembleClassifier
- Random State: 1

Model Results:
- Balanced Acurracy Score: 93%

- **Picture 6.1: Easy Ensemble AdaBoost Classifier Confusion Matrix**

![Easy Ensemble AdaBoost Classifier Confusion Matrix](https://github.com/joshuanallen/Credit_Risk_Analysis/blob/f1a2843f416ec869d229ba43e0e0f44c0ceef03e/images/6.1_Easy_Ensemble_AdaBoost_Classifier_Confusion_Matrix.png)

- **Picture 6.2: Easy Ensemble AdaBoost Classifier Classification Report**

![Easy Ensemble AdaBoost Classifier Classification Report](https://github.com/joshuanallen/Credit_Risk_Analysis/blob/f1a2843f416ec869d229ba43e0e0f44c0ceef03e/images/6.2_Easy_Ensemble_AdaBoost_Classifier_Classification_Report.png)


### Conclusions
Evaluating the above supervised ML models to evaluate credit card risk data, we can conclude the **most accurate model was the Easy Ensemble AdaBoost Classifier model** using a linear regression fit with 100 estimators. The Easy Ensemble AdaBoost Classifier model had the highest balanced accuracy score with 93% compared to the next highest score for the Balanced Random Forest Classifier model at 79%. Additionally, the EEC model also out-performed well in prediction, recall, and f1 score for high-risk loans. Specifically for loan-risk evaluation, false positives, or wrongly predicting low risk ("1") when the actual loan is high-risk ("0"), is the prediction needed to be most accurate and the best performing model for this was also the EEC model. 

## Limitations of current script/ future improvements
- To tune the models to produce more accurate predictions, we can add scaling to some of the data to eliminate more of the modeling variability using the `StandardScaler()` module.

- Using the Balanced Random Forest Classifier feature importances ranking list we can start to eliminate columns that may be less impactful on modeling predictions and therefore reduce the variability of the modeling.

- Additionally, can add more estimators or change the type of model fit to something other than linear regression to see if it improves the model's accuracy.

