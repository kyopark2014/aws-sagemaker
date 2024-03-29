# AWS SageMaker


SageMaker는 AWS의 완전 관리형 머신 러닝 학습 서비스로서, 데이터 과학자가 빠르고 쉽게 모델 개발 및 학습을 할 수 있도록 지원합니다. 


## SageMaker Training

SageMaker에서 제공되는 jupyter Notebook을 통해, 학습에 필요한 데이터를 전처리하거나, 모델을 개발할 수 있습니다. 하지만, 노트북 인스턴스에서 모델 학습을 수행할 수 있지만, 더 높은 성능의 CPU/GPU를 요구할때 노트북 인스턴스를 Scale-Up 하는것은 비용적으로 효율적이지 않습니다. 따라서, 별도 인스턴스를 띄워서 모델 학습을 진행하는데 이것을 SageMaker Training이라고 합니다.

학습을 위해서 S3에 학습에 필요한 데이터를 업로드합니다. 이후, SageMaker가 학습 클러스터로 S3의 학습데이터를 가져와서 학습을 수행하게 됩니다. 이때, 학습에 필요한 코드는 노트북에서 로드하여 학습클러스터에서 사용합니다. 

## 학습용 Container 

SageMaker에서 [학습용 Container 생성시 폴더의 경로 및 환경변수](https://github.com/kyopark2014/aws-sagemaker/blob/main/training-container.md)에 대해 설명합니다. 

## 학습용 Cluster 정의

학습 Cluster 사용할 IAM role과 Hyperparameter를 아래와 같이 정의합니다. 여기서, sagemaker.get_execution_role()을 하면 현재 노트북의 role을 가져옵니다. 별도의 role을 사용할 경우에 해당 role의 arn을 입력합니다. 

```python
import sagemaker 

sagemaker_session = sagemaker.Session()	 	# SageMaker 세션 정의
role = sagemaker.get_execution_role()		# SageMaker 노트북에서 사용하는 role 활용
```

Hyperparameter를 정의합니다. 

```python
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

학습할때 사용할 수 있는 data path에는 S3, EFS, FSx for Lustre 등 3가지 타입이 가능합니다.

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

S3에서 파일을 복사하는 시간이 오래 걸리면, GPU를 가진 프로세서가 기다려야하므로, Lustre를 검토할 수 있습니다. (수십 GB 이상) 

FileSystemInput으로 사용할때는 복사하지 않고 마운트하여 읽어오게 됩니다. 


## 학습 시작

학습 클러스터에서 사용할 데이터 경로와 channel_name을 선언한 후 실행합니다.

```python
channel_name = ”training”

estimator.fit(
	inputs={channel_name : data_path},
	job_name=job_name
)
```

## Local Mode Debugging

생성한 SageMaker Notebook에서 학습 코드를 개발할 목적으로 Local Mode로 사용할 수 있습니다.

딥러닝 분산학습의 경우 노트북 인스턴스를 GPU 유형으로 생성합니다. 단, SageMaker의 Data parallel과 Model parallel Library는 ml.p3.16xlarge이상에서 테스트 가능합니다. 이것은 임시적 사용이며 비용을 위해 테스트 후 CPU 유형으로 변경하는것이 좋습니다.

- Local Mode 
<img width="658" alt="image" src="https://user-images.githubusercontent.com/52392004/190835603-a4ae3ab8-efeb-4d46-8312-4772ca49a675.png">

- 실제 학습 Mode
<img width="658" alt="image" src="https://user-images.githubusercontent.com/52392004/190835617-68cf5d32-cb0f-436f-b9f0-5684302521fc.png">

## Matric

### Matric definition

학습 코드에서 아래와 같은 로그를 찍는다고 가정하면, Train_Loss를 matric으로 만들어 사용하고 싶을 수 있습니다.

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
	metric_definitions=metric_definitions,       		# Matric definitions
)
```

## SageMaker Basic

[SageMaker Training](https://github.com/kyopark2014/aws-sagemaker/tree/main/training-basic)에서는 xgboost를 이용한 보험사기를 검출하는 예제를 설명하고 있습니다. 


## SageMaker Experiment

[SageMaker Experiment와 Trial](https://github.com/kyopark2014/aws-sagemaker/blob/main/sagemaker-experiment.md)을 이용하여 여러 시도에 대해 사용자의 하이퍼파라미터, 평가 지표(metrics) 등을 기록 및 추적할 수 있습니다. 


## SageMaker Processing 

[SageMaker Processing](https://github.com/kyopark2014/aws-sagemaker/blob/main/sagemaker-processing.md)으로 사전 처리, 후 처리 및 모델 평가를 실행할 수 있는 환경을 제공합니다. S3의 데이터를 입력으로 받아 로직 처리 후 S3에 출력으로 저장후 SageMaker에서 dataset으로 사용할 수 있습니다. 

## Workshop

[SageMaker Immersion Day Workshop](https://github.com/kyopark2014/aws-sagemaker/tree/main/workshop)에 대해 설명합니다.

## Monitoring

GPU/CPU 리소스 사용량은 아래처럼 CloudWatch를 통해 확인할 수 있습니다.

<img width="725" alt="image" src="https://user-images.githubusercontent.com/52392004/190836077-464e9d89-8188-4814-8d8c-f8026ae55a5c.png">



## Reference

[Amazon SageMaker 모델 학습 방법 소개 - AWS AIML 스페셜 웨비나](https://www.youtube.com/watch?v=oQ7glJfD-BQ&list=PLORxAVAC5fUULZBkbSE--PSY6bywP7gyr)

[SageMaker 스페셜 웨비나 - Github](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/sagemaker/sm-special-webinar)

[Direct Marketing with Amazon SageMaker XGBoost and Hyperparameter Tuning (SageMaker SDK)](https://sagemaker-examples.readthedocs.io/en/latest/hyperparameter_tuning/xgboost_direct_marketing/hpo_xgboost_direct_marketing_sagemaker_APIs.html)
