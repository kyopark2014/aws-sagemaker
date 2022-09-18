# Dataset: 보험청구사기

[Architect and build the full machine learning lifecycle with AWS: An end-to-end Amazon SageMaker demo](https://aws.amazon.com/blogs/machine-learning/architect-and-build-the-full-machine-learning-lifecycle-with-amazon-sagemaker/)에서는 자동차 보험 청구 사기를 탐지할 수 있는 데이터를 제공합니다.

## Dataset 구조 

아래처럼 제공된 dataset의 구조를 확인 할 수 있습니다.

```python
import pandas as pd

train_prep_df = pd.read_csv('data/dataset/train.csv')
train_prep_df.groupby('fraud').sample(n=5)
```

이때의 결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/190880293-1045d20b-9c5b-4d67-8036-55a502df54bb.png)

전체에서 froud의 숫자는 아래와 같습니다. 

```python
train_prep_df.groupby('fraud').size()
```

이때의 결과는 아래와 같습니다. 
```python
fraud
0    3869
1     131
dtype: int64
```

## Dataset Column

각 Cloumn에 대한 설명은 아래와 같습니다. 

- fraud: 보험 청구의 사기 여부 입니다. 1 이면 사기, 0 이면 정상 청구 입니다.
- vehicle_claim: 자동차에 대한 보험 청구액. 값으로서, $1000, $17,638 등이 있습니다.
- total_claim_amount: 전체 보험 청구액 입니다. $21,400, $10,000 등이 있습니다.    
- customer_age: 고객의 나이를 의미 합니다.
- months_as_customer: 고객으로서의 가입 기간을 의미합니다. 단위는 월로서 11, 30, 31 등의 값이 존재 합니다.
- num_claims_past_year: 작년의 보험 청구 수를 의미 합니다. 0, 1, 2, 3, 4, 5, 6 의 값이 존재 합니다.
- num_insurers_past_5_years: 과거 5년 동안의 보험 가입 회사 수를 의미 합니다. 1, 2, 3, 4, 5 의 값이 존재 합니다.
- policy_deductable: 보험의 최소 자기 부담금 입니다. $750, $800 등이 있습니다.    
- policy_annual_premium: 보험의 특약 가입에 대한 금액 입니다. $2000, $3000 등이 있습니다.
- customer_zip: 고객의 집 주소 우편 번호를 의미합니다.
- auto_year: 자동차의 년식을 의미 합니다. 2020, 2019 등이 있습니다.
- num_vehicles_involved: 몇 대의 자동차가 사고에 연관 되었는지 입니다. 1, 2, 3, 4, 5, 6 의 값이 있습니다.
- num_injuries: 몇 명이 상해를 입었는지를 기술합니다. 0, 1, 2, 3, 4, 의 값이 있습니다.
- num_witnesses: 몇 명의 목격자가 있었는지를 기술합니다. 0, 1, 2, 3, 4, 5 의 값이 있습니다.
- injury_claim: 상해에 대한 보험 청구액. \$5,500, \$70,700, \$100,700 등이 있습니다.    
- incident_month: 사고가 발생한 월을 의미합니다. 1~12 값이 존재 합니다.
- incident_day: 사고가 발생한 일자를 의미합니다. 1~31 값이 존재 합니다.
- incident_dow: 사고가 발생한 요일을 의미합니다. 0~6 값이 존재 합니다.
- incident_hour: 사고가 발생한 시간을 의미합니다. 0~23 값이 존재 합니다.
- policy_state: 보험 계약을 한 미국 주(State)를 의미 합니다. CA, WA, AZ, OR, NV, ID 가 존재 합니다.    
- policy_liability: 보험 청구의 한도를 의미 합니다. 예를 들어서 25/50 은  사람 당 상해 한도 $25,000, 사고 당 상해 한도가 $50,000 을  의미합니다. 25/50, 15/30, 30/60, 100/200 의 값이 존재 합니다. 
- customer_gender: 고객의 성별을 의미 합니다. Male, Female, Unkown, Other가 존재 합니다.
- customer_education: 고객의 최종 학력을 의미합니다. Bachelor, High School, Advanced Degree, Associate, Below High School 이 존재 합니다.
- driver_relationship: 보험 계약자와 운전자와의 관계 입니다. Self, Spouse, Child, Other 값이 존재 합니다.
- incident_type: 사고의 종류를 기술합니다. Collision, Break-in, Theft 값이 존재 합니다.
- collision_type: 충돌 타입을 기술합니다. Front, Rear, Side, missing 값이 존재 합니다.
- incident_severity: 사고의 손실 정도 입니다. Minor, Major, Totaled 값이 존재 합니다.
- authorities_contacted: 어떤 관련 기관에 연락을 했는지 입니다. Police, Ambuylance, Fire, None 값이 존재 합니다.
- police_report_available: 경찰 보고서가 존재하는지를 기술합니다. Yes, No 의 값이 있습니다.
