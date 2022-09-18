# SageMaker Experiment

실험의 여러 시도에 대해 사용자의 하이퍼파라미터, 평가 지표(metrics) 등을 기록 및 추적할 수 있습니다. 사용법은 아래와 같이 Experiment와 Trial을 생성하여 사용하면 됩니다.

## SageMaker Experiment 설치

```c
pip install sagemaker-experiments
```

## Experiments & Trials 

```python
from smexperiments.experiment import Experiment 
from smexperiments.trial import Trial

experiment = Experiment.create(
    experiment_name=”experiment_name", 
    description="Classification of mnist hand-written digits”
)

trial = Trial.create(
    trial_name=“trial_name”, 
    experiment_name=experiment.experiment_name,
    sagemaker_boto_client=sm,)
```

이후 실제 학습을 시도할때 아래와 같이 "experiment_config"을 넣어서 수행합니다. 

```python
channel_name = ”training”
estimator.fit(
	inputs={channel_name : data_path},
	job_name=job_name,
	experiment_config={
            "TrialName": trial.trial_name,
            "TrialComponentDisplayName": "Training"})
```

## 실제적인 Example

아래에서는 [training-experiment](https://github.com/kyopark2014/aws-sagemaker/blob/main/training-basic/training-experiment.ipynb) 코드에서 Experiment 관련된 부분을 설명합니다. 

1) Experiment를 선언합니다. 

```python
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from time import strftime

def create_experiment(experiment_name):
    try:
        sm_experiment = Experiment.load(experiment_name)
    except:
        sm_experiment = Experiment.create(experiment_name=experiment_name)
        
def create_trial(experiment_name):
    create_date = strftime("%m%d-%H%M%s")       
    sm_trial = Trial.create(trial_name=f'{experiment_name}-{create_date}',
                            experiment_name=experiment_name)

    job_name = f'{sm_trial.trial_name}'
    return job_name  

experiment_name='xgboost-poc-1'

create_experiment(experiment_name)
job_name = create_trial(experiment_name)
```

2) 학습할때 fit()에 아래처럼 넣어서 사용합니다. 

```python
estimator.fit(inputs = {'inputdata': inputs},
                  job_name = job_name,
                  experiment_config={
                      'TrialName': job_name,
                      'TrialComponentDisplayName': job_name,
                  },
                  wait=False)
```

## 결과 분석

```python
from sagemaker.analytics import ExperimentAnalytics
import pandas as pd
pd.options.display.max_columns = 50
pd.options.display.max_rows = 10
pd.options.display.max_colwidth = 100

trial_component_training_analytics = ExperimentAnalytics(
    sagemaker_session= sagemaker_session,
    experiment_name= experiment_name,
    sort_by="metrics.validation:auc.max",        
    sort_order="Descending",
    metric_names=["validation:auc"]
)

trial_component_training_analytics.dataframe()[['Experiments', 'Trials', 'validation:auc - Min', 'validation:auc - Max',
                                                'validation:auc - Avg', 'validation:auc - StdDev', 'validation:auc - Last', 
                                                'eta', 'max_depth', 'num_round', 'scale_pos_weight']]
						


## Training jobs

[Training jobs Console](https://ap-northeast-2.console.aws.amazon.com/sagemaker/home?region=ap-northeast-2#/jobs)에 가면 현재진행중 또는 완료된 학습에 대한 Job 결과를 확인할 수 있습니다.

![noname](https://user-images.githubusercontent.com/52392004/190890620-14d7950c-353b-499c-93d5-46f0f18f2ca4.png)


## Reference 

[Create an Amazon SageMaker Experiment](https://docs.amazonaws.cn/en_us/sagemaker/latest/dg/experiments-create.html)

[Amazon SageMaker Experiments Python SDK](https://sagemaker-experiments.readthedocs.io/en/latest/index.html)

[github-SageMaker Experiments Python SDK](https://github.com/aws/sagemaker-experiments)

[Amazon SageMaker 모델 학습 방법 소개 - AWS AIML 스페셜 웨비나](https://www.youtube.com/watch?v=oQ7glJfD-BQ&list=PLORxAVAC5fUULZBkbSE--PSY6bywP7gyr)

[SageMaker 스페셜 웨비나 - Github](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/sagemaker/sm-special-webinar)

