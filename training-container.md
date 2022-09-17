# 학습용 Container

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

## Reference 

[Amazon SageMaker 모델 학습 방법 소개 - AWS AIML 스페셜 웨비나](https://www.youtube.com/watch?v=oQ7glJfD-BQ&list=PLORxAVAC5fUULZBkbSE--PSY6bywP7gyr)

[Architect and build the full machine learning lifecycle with AWS: An end-to-end Amazon SageMaker demo](https://aws.amazon.com/ko/blogs/machine-learning/architect-and-build-the-full-machine-learning-lifecycle-with-amazon-sagemaker/)
