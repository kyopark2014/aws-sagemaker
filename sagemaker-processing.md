# SageMaker Processing

사전 처리, 후 처리 및 모델 평가를 실행할 수 있는 환경을 제공합니다. S3의 데이터를 입력으로 받아 로직 처리 후 S3에 출력으로 저장합니다.

아래에서는 processor를 정의한 후에 "preprocessing.py"을 이용해 데이터를 처리하고, 결과인 train/validation을 S3에 저장하는 코드를 보여주고 있습니다. 

```python
# Built-in Scikit Learn Container or FrameworkProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import Processor, ScriptProcessor, FrameworkProcessor

processor= FrameworkProcessor(PyTorch, framework_version="1.10", 
				   role=role, instance_type=‘ml.g5.xlarge’, 
				   instance_count=1)

from sagemaker.processing import ProcessingInput, ProcessingOutput

processor.run(
    code='preprocessing.py',
    inputs=[ProcessingInput(source=INPUT_S3_URI, destination='/opt/ml/processing/input')],
    outputs=[ProcessingOutput(source='/opt/ml/processing/output/train’, destination=OUTPUT_S3_URI_1),
             ProcessingOutput(source='/opt/ml/processing/output/validation’, destination=OUTPUT_S3_URI_2)]
)
```

## Evaluation

[Sagemaker Processing](https://github.com/kyopark2014/aws-sagemaker/blob/main/training-basic/training-processing.ipynb)에서는 processing을 설정하여 얻어진 결과를 Jupyter notebook으로 확인 할 수 있습니다. 

아래와 같이 processing을 정의한 후에 [evalutation.py](https://github.com/kyopark2014/aws-sagemaker/blob/main/training-basic/src/evaluation.py)을 실행합니다. 입력으로는 s3에서 test_data와 model_weight을 복사하여 "/opt/ml/processing/test"와 "/opt/ml/processing/model"에서 로드하여 사용하고, processing 결과는 "/opt/ml/processing/output"에 넣습니다. 
	    
```python
from sagemaker.processing import FrameworkProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

instance_count = 1
instance_type = "ml.m5.large"
# instance_type = 'local'

script_eval = FrameworkProcessor(
    XGBoost,
    framework_version="1.3-1",
    role=role,
    instance_type=instance_type,
    instance_count=instance_count
)

artifacts_dir = estimator.model_data

s3_test_path = data_path + '/test.csv'
detect_outputpath = f's3://{bucket}/xgboost/processing'

source_dir='src'

if instance_type == 'local':
    from sagemaker.local import LocalSession
    from pathlib import Path

    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}
    source_dir = f'{Path.cwd()}/src'
    s3_test_path=f'../data/dataset/test.csv'
    
create_experiment(experiment_name)
job_name = create_trial(experiment_name)

script_eval.run(
    code="evaluation.py",
    source_dir=source_dir,
    inputs=[ProcessingInput(source=s3_test_path, input_name="test_data", destination="/opt/ml/processing/test"),
            ProcessingInput(source=artifacts_dir, input_name="model_weight", destination="/opt/ml/processing/model")
    ],
    outputs=[
        ProcessingOutput(source="/opt/ml/processing/output", output_name='evaluation', destination=detect_outputpath + "/" + job_name),
    ],
    job_name=job_name,
    experiment_config={
        'TrialName': job_name,
        'TrialComponentDisplayName': job_name,
    },
    wait=False
)
```

이때의 결과는 아래와 같습니다. 

```python
artifacts_dir = estimator.model_data.replace('model.tar.gz', '')
print(artifacts_dir)
!aws s3 ls --human-readable {artifacts_dir}

model_dir = './model'

!rm -rf $model_dir

import json , os

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

!aws s3 cp {artifacts_dir}model.tar.gz {model_dir}/model.tar.gz
!tar -xvzf {model_dir}/model.tar.gz -C {model_dir}

!pip install xgboost graphviz

import xgboost as xgb
import matplotlib.pyplot as plt

model = xgb.XGBClassifier()
model.load_model("./model/xgboost-model")

test_prep_df = pd.read_csv('./data/dataset/test.csv')
x_test = test_prep_df.drop('fraud', axis=1)
feature_data = xgb.DMatrix(x_test)
model.get_booster().feature_names = feature_data.feature_names
model.get_booster().feature_types = feature_data.feature_types
fig, ax = plt.subplots(figsize=(15, 8))
xgb.plot_importance(model, ax=ax, importance_type='gain')
```

이때 얻어진 feature importance는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/190893110-1f3ce6f9-2f24-46b4-ae27-4b9a9d5dacd3.png)


```python
xgb.plot_tree(model, num_trees=0, rankdir='LR')

fig = plt.gcf()
fig.set_size_inches(50, 15)
plt.show()
````

이때의 결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/190893129-6ca3d28e-74cb-4fbf-9203-23bc4a544924.png)

## Evalution.py 

[evalutation.py](https://github.com/kyopark2014/aws-sagemaker/blob/main/training-basic/src/evaluation.py)의 내용을 보면 아래와 같습니다. 

아래와 같이 model과 test dataset, output에 대한 경로를 읽어옵니다.

```python
    parser.add_argument('--base_dir', type=str, default= "/opt/ml/processing")    
    parser.add_argument('--model_path', type=str, default= "/opt/ml/processing/model/model.tar.gz")
    parser.add_argument('--test_path', type=str, default= "/opt/ml/processing/test/test.csv")
    parser.add_argument('--output_evaluation_dir', type=str, default="/opt/ml/processing/output")
```    

model을 load 합니다. 

```python
    model = xgboost.XGBRegressor()
    model.load_model("xgboost-model")
````

Test dataset을 로드하고, predict를 수행합니다. 

```python
df = pd.read_csv(test_path)
df.drop(df.columns[0], axis=1, inplace=True)

predictions_prob = model.predict(X_test)
```


MSE를 계산합니다. 

```python
    mse = mean_squared_error(y_test, predictions)
    std = np.std(y_test - predictions)
    report_dict = {
        "regression_metrics": {
            "mse": {
                "value": mse,
                "standard_deviation": std
            },
        },
    }
```    

계산된 결과를 저장합니다. 

```python
    evaluation_path = f"{output_evaluation_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))	
```

결과적으로 아래와 같이 S3에 저장됩니다. 

![noname](https://user-images.githubusercontent.com/52392004/190893891-3be9aff3-e8cb-4461-9394-751c3e8b953c.png)

evaluation.json에는 아래와 같은 결과를 갖습니다. 

```json
{
   "regression_metrics":{
      "mse":{
         "value":0.277,
         "standard_deviation":0.4558848538830831
      }
   }
}
	
