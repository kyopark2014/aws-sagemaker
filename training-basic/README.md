# Training

## 학습용 Container 

SageMaker에서 [학습용 Container 생성시 폴더의 경로 및 환경변수](https://github.com/kyopark2014/aws-sagemaker/blob/main/training-container.md)에 대해 설명합니다. 

## 학습용 Cluster 정의

학습 Cluster 사용할 IAM role과 Hyperparameter를 아래와 같이 정의합니다. 여기서, sagemaker.get_execution_role()을 하면 현재 노트북의 role을 가져옵니다. 별도의 role을 사용할 경우에 해당 role의 arn을 입력합니다. 

```python
import sagemaker 

sagemaker_session = sagemaker.Session()	 	# SageMaker 세션 정의
role = sagemaker.get_execution_role()		# SageMaker 노트북에서 사용하는 role 활용

hyperparameters = {“batch_size” : 32 ,
		   “lr” : 1e-4 , 
		   “image_size” : 128 }		# 학습 코드의 arguments 값
```


학습 클러스터의 인스턴스 종류/수, 실행할 학습 코드, 학습 환경 컨테이너 등을 Estimator로 정의합니다. 

```python
from sagemaker.pytorch import PyTorch 

estimator = PyTorch( 
	source_dir="code",                                   	# 학습 코드 폴더 지정
	entry_point="train_pytorch_smdataparallel_mnist.py",	# 실행 학습 스크립트 명
	role=role, 						# 학습 클러스터에서 사용할 Role
	framework_version="1.10",				# Pytorch 버전
	py_version="py38", 					# Python 버전
	instance_count=1,        				# 학습 인스턴스 수
	instance_type="ml.p4d.24xlarge",             		# 학습 인스턴스 명
	sagemaker_session=sagemaker_session,			# SageMaker 세션
	hyperparameters=hyperparameters,			# 하이퍼파라미터 설정
)
```

추가적으로 아래와 같은 파라메터를 estimater에서 추가하여 사용할수 있습니다. 

```python
estimator = PyTorch( 
	… ,
	max_run=5*24*60*60,			# 최대 학습 수행 시간 (초)
	use_spot_instances=True, 		# spot 인스턴스 사용 여부
	max_wait=3*60*60, 			# spot 사용 시 자원 재확보를 위한 대기 시간
	checkpoint_s3_uri= checkpoint_s3_uri,    # checkpoints 저장 S3 위치
	…		
)
```


## Data path

fit에서 추가되는 data_path는 S3, EFS, FSx for Lustre 등 3가지 타입이 가능합니다.

```python
# S3 
data_path = “s3://my_bucket/my_training_data/”

# EFS
data_path = FileSystemInput(file_system_id='fs-1’, file_system_type='EFS’,
			   directory_path=‘/dataset’, file_system_access_mode=‘ro’)

# FSx for Lustre
data_path = FileSystemInput(file_system_id='fs-2’, file_system_type='FSxLustre’, 
			   directory_path='/<mount-id>/dataset’, 
			   file_system_access_mode='ro’)
```


## 학습 시작

학습 클러스터에서 사용할 데이터 경로와 channel_name을 선언한 후 실행합니다.

```python
channel_name = ”training”

estimator.fit(
	inputs={channel_name : data_path},
	job_name=job_name
)
```

## Local Mode Debuging

생성한 SageMaker Notebook에서 학습 코드를 개발할 목적으로 Local Mode로 사용할 수 있습니다.

딥러닝 분산학습의 경우 노트북 인스턴스를 GPU 유형으로 생성합니다. 단, SageMaker의 Data parallel과 Model parallel Library는 ml.p3.16xlarge이상에서 테스트 가능합니다. 이것은 임시적 사용이며 비용을 위해 테스트 후 CPU 유형으로 변경하는것이 좋습니다.

- Local Mode 
<img width="658" alt="image" src="https://user-images.githubusercontent.com/52392004/190835603-a4ae3ab8-efeb-4d46-8312-4772ca49a675.png">

- 실제 학습 Mode
<img width="658" alt="image" src="https://user-images.githubusercontent.com/52392004/190835617-68cf5d32-cb0f-436f-b9f0-5684302521fc.png">

## Matric

### Matric definition

학습 코드에서 아래와 같은 로그를 찍는데, Train_Loss를 matric으로 만들어 사용하고 싶은 케이스가 있다고 가정합니다. 

```python
Epoch : [2][6/10] Train_Time = 0.355 : (3.134) , Train_Speed = 1803.360 (204.190), Train_Loss = 1.0813320875 : (1.3528) , Train_Prec@1 = 76.250 : (68.958)
```

이때, matric 정보를 hooking하여 아래처럼 사용할 수 있습니다. 

```python
metric_definitions = [ { ‘Name’ : ‘train:Loss’, ‘Regex’ : ‘Train_Loss = (.*?) :’}, ...]
```

이후, estimator 정의시 아래처럼 matric_definitions을 추가합니다. 

```python
estimator = PyTorch( 
	source_dir="code",                                   	# 학습 코드 폴더 지정
	entry_point="train_pytorch_smdataparallel_mnist.py",	# 실행 학습 스크립트 명
	role=role, 						# 학습 클러스터에서 사용할 Role
	framework_version="1.10",				# Pytorch 버전
	py_version="py38", 					# Python 버전
	instance_count=1,        				# 학습 인스턴스 수
	instance_type="ml.p4d.24xlarge",             		# 학습 인스턴스 명
	sagemaker_session=sagemaker_session,			# SageMaker 세션
	hyperparameters=hyperparameters,			# 하이퍼파라미터 설정
	**metric_definitions=metric_definitions,       		# Matric definitions
)
```

## SagmeMaker Experiment

[SageMaker Experiment와 Trial](https://github.com/kyopark2014/aws-sagemaker/blob/main/sagemaker-experiment.md)을 이용하여 여러 시도에 대해 사용자의 하이퍼파라미터, 평가 지표(metrics) 등을 기록 및 추적할 수 있습니다. 


## SageMaker Processing 

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

## Monitoring

GPU/CPU 리소스 사용량은 아래처럼 CloudWatch를 통해 확인할 수 있습니다.

<img width="725" alt="image" src="https://user-images.githubusercontent.com/52392004/190836077-464e9d89-8188-4814-8d8c-f8026ae55a5c.png">



## Reference

[Amazon SageMaker 모델 학습 방법 소개 - AWS AIML 스페셜 웨비나](https://www.youtube.com/watch?v=oQ7glJfD-BQ&list=PLORxAVAC5fUULZBkbSE--PSY6bywP7gyr)

[SageMaker 스페셜 웨비나 - Github](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/sagemaker/sm-special-webinar)

