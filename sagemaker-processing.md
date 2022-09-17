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
