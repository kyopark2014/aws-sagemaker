# Direct Marketing with Amazon SageMaker XGBoost and Hyperparameter Tuning (SageMaker API)

## Dataset

[Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/bank+marketing)은 "Portuguese banking institution"의 direct marketing campaign 데이터셋입니다. 이것은 전화상담에 대한 campaign 자료이며, 같은 사용자에게 여러번 contact한 정보를 가지고 있을 수 있습니다. 여기서는 Sagemaker를 이용하여 XGBoost로 Direct Marketing 데이터를 학습할때 HPO 방법에 대해 설명합니다. 

[hpo_xgboost_direct_marketing_sagemaker_python_sdk.ipynb](https://github.com/kyopark2014/aws-sagemaker/blob/main/sagemaker-examples/direct-marketing-xgboost/hpo_xgboost_direct_marketing_sagemaker_python_sdk.ipynb)은 편의상 [원본 - hpo_xgboost_direct_marketing_sagemaker_APIs.ipynb](https://github.com/aws/amazon-sagemaker-examples/blob/main/hyperparameter_tuning/xgboost_direct_marketing/hpo_xgboost_direct_marketing_sagemaker_APIs.ipynb)을 가져와서 수정하였습니다. 


### Datasets

1) bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]

2) bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.

3) bank-full.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs).

4) bank.csv with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs).
The smallest datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM).

### Attribute Information:

#### Bank client data:
- age (numeric)
- job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
- marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
- education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
- default: has credit in default? (categorical: 'no','yes','unknown')
- housing: has housing loan? (categorical: 'no','yes','unknown')
- loan: has personal loan? (categorical: 'no','yes','unknown')

#### Related with the last contact of the current campaign:
- contact: contact communication type (categorical: 'cellular','telephone')
- month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
- day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
- duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

#### Other attributes:
- campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
- pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
- previous: number of contacts performed before this campaign and for this client (numeric)
- poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

#### Social and economic context attributes
- emp.var.rate: employment variation rate - quarterly indicator (numeric)
- cons.price.idx: consumer price index - monthly indicator (numeric)
- cons.conf.idx: consumer confidence index - monthly indicator (numeric)
- euribor3m: euribor 3 month rate - daily indicator (numeric)
- nr.employed: number of employees - quarterly indicator (numeric)

#### Output variable (desired target):
- y: has the client subscribed a term deposit? (binary: 'yes','no')


## Reference 

[Direct Marketing with Amazon SageMaker XGBoost and Hyperparameter Tuning (SageMaker API)](https://sagemaker-examples.readthedocs.io/en/latest/hyperparameter_tuning/xgboost_direct_marketing/hpo_xgboost_direct_marketing_sagemaker_APIs.html)
