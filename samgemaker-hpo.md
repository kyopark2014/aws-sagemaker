# HPO

현재의 dataset을 이용하여 [HPO(Hyperparameter Optimization)](https://github.com/kyopark2014/ML-Algorithms/blob/main/hyperparameter-optimization.md)을 구하는 과정을 설명합니다. [HPO - training-experiment-HPO.ipynb](https://github.com/kyopark2014/aws-sagemaker/blob/main/training-basic/training-experiment-HPO.ipynb)의 내용을 아래와 같이 설명하고자 합니다.

아래와 같이 HPO를 수행합니다. max_depth와 eta에 대해 최적의 Hyperparameter를 찾도록 설정하였습니다. 

```python
max_jobs=4    
max_parallel_jobs=2  

create_experiment(experiment_name)
job_name = create_trial(experiment_name)

job_name =  job_name[15:] ## job_name must have length less than or equal to 32 for HPO

tuner = sagemaker.tuner.HyperparameterTuner(
    estimator,
    objective_metric_name="validation:auc",
    hyperparameter_ranges={
        "max_depth": sagemaker.tuner.IntegerParameter(2, 5),
        "eta": sagemaker.tuner.ContinuousParameter(0.1, 0.5)
    },
    objective_type="Maximize",
    max_jobs=max_jobs,   
    max_parallel_jobs=max_parallel_jobs,   
)

tuner.fit(
    job_name = job_name,
    inputs={'inputdata': inputs},
    experiment_config={
          'TrialName': job_name,
          'TrialComponentDisplayName': job_name,
    },
    wait=False
)

tuner.wait()
```

HPO 완료후 결과는 아래와 같이 확인할 수 있습니다.

```python
from sagemaker.analytics import ExperimentAnalytics, HyperparameterTuningJobAnalytics
import pandas as pd
pd.options.display.max_columns = 50
pd.options.display.max_rows = 10
pd.options.display.max_colwidth = 100

trial_component_training_analytics = HyperparameterTuningJobAnalytics(
    sagemaker_session= sagemaker_session,
    hyperparameter_tuning_job_name=job_name
)

trial_component_training_analytics.dataframe()[['TrainingJobName', 'TrainingJobStatus', 
                                                'eta', 'max_depth', 'FinalObjectiveValue']]
```

이때의 결과는 아래와 같습니다. eta가 0.3이고, max_depth가 2일때 최적의 결과를 가지고 있습니다. 

![image](https://user-images.githubusercontent.com/52392004/190898211-4c6c8801-1da4-43b4-8fc9-ce87cd33f9ad.png)



## Tuning jobs status

[Hyperparameter tuning jobs Console](https://ap-northeast-2.console.aws.amazon.com/sagemaker/home?region=ap-northeast-2#/hyper-tuning-jobs)에서 아래와 같이 현재의 tunning job의 상태 또는 완료된 상태를 확인 할 수 있습니다. 

![image](https://user-images.githubusercontent.com/52392004/190897794-bfae1e67-acae-4e81-b880-2d1fa85603cb.png)
