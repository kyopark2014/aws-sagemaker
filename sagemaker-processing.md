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

아래와 같이 processing을 정의한 후에 [evalutation.py](https://github.com/kyopark2014/aws-sagemaker/blob/main/training-basic/src/evaluation.py)을 실행합니다. 

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

이때의 
