# Training

## 학습용 Container의 구조 

Training시 생성되는 Container에 있는 폴더의 구조는 아래와 같습니다.

```c
/opt/ml 
├── input 
│ └── data 
│      └── <channel_name> 
│               └── <input data> 
├── model 
├── code 
├── output 
└── checkpoints
```

#### Dataset과 source 

ML을 Training하기 위해 필요한 데이터와 소스는 아래 경로에 위치하게 됩니다. 

- input/data/<channel_name>/<input data>: S3의 데이터가 복사되는 위치입니다.
- code: 노트북의 소스코드가 복사되어 저장됩니다.

따라서, 정의된 코드에서 데이터를 읽어오는 경로는 "/opt/ml/input/data/{channel_name}"이 되고, 이것은 아래처럼 환경변수로 부터 가져와서 사용할 수 있습니다. 

```python
os.environ.get("SM_CHANNEL_${channel_name}')
```

#### Outputs

모델 학습후 결과들을 저장할 경로는 아래와 같습니다. 학습이 종료가 되면, model의 파일들은 model.tar.gz으로, out의 파일들은 output.tar.gz으로 미리 지정한 S3 bucket에 저장됩니다. 

- model: 모델결과를 저장할 위치 예) torch.savme("/opt/ml/model/best.pt")
- output: log 파일등 저장할 위치 예) write("/opt/ml/output/\*\*\*.event")

#### Checkpoint

chk파일등은 "/opt/ml/checkpoints"에 저장되고, 거의 실시간으로 S3 bucket에 복사 됩니다. 


### Environment variables

[SageMaker environment variables](https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md)을 참조하여, 코드상에 주요 경로를 환경변수로부터 읽어와서 사용합니다. 

```python
# /opt/ml/model
parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR’)) 

# /opt/ml/input/data/training
parser.add_argument('--dataset_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING’)) 

# /opt/ml/output/data/algo-1
parser.add_argument('--output_data_dir', type=str,default=os.environ.get('SM_OUTPUT_DATA_DIR’))

# /opt/ml/output
parser.add_argument('--output-dir', type=str,default=os.environ.get('SM_OUTPUT_DIR’))
```

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


#### Data path

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

## Debuging

생성한 SageMaker Notebook에서 학습 코드를 개발할 목적으로 Local Mode로 사용할 수 있습니다.

딥러닝 분산학습의 경우 노트북 인스턴스를 GPU 유형으로 생성합니다. 단, SageMaker의 Data parallel과 Model parallel Library는 ml.p3.16xlarge이상에서 테스트 가능합니다. 이것은 임시적 사용이며 비용을 위해 테스트 후 CPU 유형으로 변경하는것이 좋습니다.

- Local Mode 
<img width="658" alt="image" src="https://user-images.githubusercontent.com/52392004/190835603-a4ae3ab8-efeb-4d46-8312-4772ca49a675.png">

- 실제 학습 Mode
<img width="658" alt="image" src="https://user-images.githubusercontent.com/52392004/190835617-68cf5d32-cb0f-436f-b9f0-5684302521fc.png">

## Reference

[Amazon SageMaker 모델 학습 방법 소개 - AWS AIML 스페셜 웨비나](https://www.youtube.com/watch?v=oQ7glJfD-BQ&list=PLORxAVAC5fUULZBkbSE--PSY6bywP7gyr)

