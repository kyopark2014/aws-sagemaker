# Training

## Dataset

[보험청구사기 Dataset](https://github.com/kyopark2014/aws-sagemaker/blob/main/dataset.md)을 이용해 아래와 같이 ML 학습을 수행하고자 합니다.

## XGBoost를 이용한 자동차 보험 청구사기 탐지 방법

아래에서는 [SageMaker Training 예제](https://github.com/kyopark2014/aws-sagemaker/blob/main/training-basic/training.ipynb)를 기준으로 설명합니다.

1) 라이브러리를 선언합니다. 

```python
import boto3
import sagemaker
```

2) 세션과 Role을 정의 합니다.

```python
sagemaker_session = sagemaker.session.Session()
role = sagemaker.get_execution_role()
```

3) bucket과 폴더를 정의합니다.

```python
bucket = sagemaker_session.default_bucket()
code_location = f's3://{bucket}/xgboost/code'
output_path = f's3://{bucket}/xgboost/output'
```

4) Hyperparameter를 정의합니다. 여기서는 XGBoost 알고리즘에 대한 Hyperparameter를 아래처럼 정의하여 사용합니다.

```python
hyperparameters = {
       "scale_pos_weight" : "29",    
        "max_depth": "3",
        "eta": "0.2",
        "objective": "binary:logistic",
        "num_round": "100",
}
```

5) 학습에 사용할 Cluster의 사양을 정의합니다. instance_type을 "local"로 지정하면, 학습 Cluster에서 학습을 시작하기 전에 미리 시험(Local debugging)해볼 수 있습니다. 

```python
instance_count = 1
instance_type = "ml.m5.large"
# instance_type = "local"
max_run = 1*60*60

use_spot_instances = False
if use_spot_instances:
    max_wait = 1*60*60
else:
    max_wait = None

if instance_type in ['local', 'local_gpu']:
    from sagemaker.local import LocalSession
    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}
else:
    sagemaker_session = sagemaker.session.Session()
```    

6) Estimator를 정의합니다.

```python
from sagemaker.xgboost.estimator import XGBoost

estimator = XGBoost(
    entry_point="xgboost_starter_script.py",
    source_dir='src',
    output_path=output_path,
    code_location=code_location,
    hyperparameters=hyperparameters,
    role=role,
    sagemaker_session=sagemaker_session,
    instance_count=instance_count,
    instance_type=instance_type,
    framework_version="1.3-1",
    max_run=max_run,
    use_spot_instances=use_spot_instances,  # spot instance 활용
    max_wait=max_wait,
)
```

7) 데이터를 준비합니다. 

```python
data_path=f's3://{bucket}/xgboost/dataset'
!aws s3 sync ./dataset/ $data_path

if instance_type in ['local', 'local_gpu']:
    from pathlib import Path
    file_path = f'file://{Path.cwd()}'
    inputs = file_path.split('lab_1_training')[0] + '/data/dataset/'
    
else:
    inputs = data_path
inputs
```

8) 학습을 시작합니다. 

```python
estimator.fit(inputs = {'inputdata': inputs}, wait=False)
```

## 학습결과

학습결과는 Jupyter notebook에서 확인할 수 있지만, 아래와 같이 CloudWatch에서도 확인 가능합니다. 

![image](https://user-images.githubusercontent.com/52392004/190894809-c61fa9e8-8aee-4687-8eed-59a9f74437c8.png)



## Reference

[Amazon SageMaker 모델 학습 방법 소개 - AWS AIML 스페셜 웨비나](https://www.youtube.com/watch?v=oQ7glJfD-BQ&list=PLORxAVAC5fUULZBkbSE--PSY6bywP7gyr)

[SageMaker 스페셜 웨비나 - Github](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/sagemaker/sm-special-webinar)

[Dataset - Architect and build the full machine learning lifecycle with AWS: An end-to-end Amazon SageMaker demo](https://aws.amazon.com/ko/blogs/machine-learning/architect-and-build-the-full-machine-learning-lifecycle-with-amazon-sagemaker/)
