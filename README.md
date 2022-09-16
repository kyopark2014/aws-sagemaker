# AWS SageMaker


SageMaker는 완전 관리형 머신 러닝 학습 서비스로서, 데이터 과학자가 빠르고 쉽게 모델 개발 및 학습을 할 수 있도록 지원합니다. 


## SageMaker Training

SageMaker에서 제공되는 jupyter Notebook을 통해, 학습에 필요한 데이터를 전처리하거나, 모델을 개발할 수 있습니다. 하지만, 노트북 인스턴스에서 모델 학습을 수행할 수 있지만, 더 높은 성능의 CPU/GPU를 요구할때 노트북 인스턴스를 Scale-Up 하는것은 비용적으로 효율적이지 않습니다. 따라서, 별도 인스턴스를 띄워서 모델 학습을 진행하는데 이것을 SageMaker Training이라고 합니다.

## 학습방법

S3에 학습에 필요한 데이터를 업로드합니다. 이후, SageMaker가 학습 클러스터로 S3의 학습데이터를 가져와서 학습을 수행하게 됩니다. 

학습에 필요한 코드는 노트북에서 로드하여 학습클러스터에서 사용합니다. 




## Reference

[Amazon SageMaker 모델 학습 방법 소개 - AWS AIML 스페셜 웨비나](https://www.youtube.com/watch?v=oQ7glJfD-BQ&list=PLORxAVAC5fUULZBkbSE--PSY6bywP7gyr)

[SageMaker 스페셜 웨비나 - Github](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/sagemaker/sm-special-webinar)
