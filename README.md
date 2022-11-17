# Isolation Forest를 이용한 시계열 데이터에서의 이상치 탐지
이상치 탐지



## 1. Introduction

## 2. Isolation Forest
Isolation Forest는 

<p align="center"> <img src="https://github.com/cyp-ark/if/blob/main/figure/figure1.png?raw=true" width="40%" height="40%">
  
  위의 그림에서 정상값인 $x_{i}$를 고립시키는데 총 12번의 분기가 필요한 것을 확인할 수 있습니다. 반대로 이상치인 $x_{o}$의 경우 단 4번의 분기만으로 해당 값을 고립시킬 수 있습니다. Isolation Forest의 아이디어는 단순합니다. "이상치일수록 해당 데이터를 고립시키는데에 필요한 분기 수가 적고 정상값일 수록 고립에 필요한 분기 수가 많을 것이다"라고 축약하여 표현 할 수 있습니다. 이상치 점수는 다음의 식과 같이 표현 할 수 있습니다.
$$s(x,n) = 2^{-\frac{E(h(x))}{c(n)}}$$
여기서 $E(h(x))$는 해당 x값을 고립시키는데 필요한 분기 수의 평균입니다. 이상치일 수록 해당 값이 낮아지기 때문에 이상치 점수 $s(x,n)$은 1에 가까운 값을 가질 것이고, 정상 데이터의 경우 평균 분기 수가 높을 것이기 때문에 0에 가까운 값을 가질 것 입니다.
## 3. Robust Random Cut Forest
### 3.1.Shingling
Shingling은 최근 $k$개의 값을 열벡터로 결합하여 feature 벡터로 사용하는 방법입니다. 예를 들어 $k=4$일 경우 첫번째 데이터는 $(t_{1},t_{2},t_{3},t_{4})^{T}$, 두번째 데이터는 $(t_{2},t_{3},t_{4},t_{5})^{T}$의 모양으로 변환되게 됩니다. Sliding window 같이 한 시점 단위로 shifting 되면서 vector를 구성하게 되고, 이는 시계열 데이터를 캡슐화 하여 noise에 강건하게 대응할 수 있습니다. 또한 이 값이 정상 범주에서 벗어날 경우 이상치로 탐색 할 수 있습니다.
  
<p align="center"> <img src="https://github.com/cyp-ark/if/blob/main/figure/figure5.png?raw=true" width="60%" height="60%"> 

## 4. NYC taxi data
사용된 데이터셋은 unsupervised anomaly detection 분야에서 자주 사용되는 '뉴욕시 택시 탑승객 수'로 2014년 7월부터 2015년 1월까지 뉴욕시 택시 탑승객 수를 30분 단위로 측정한 데이터입니다. 원래는 unsupervised 데이터 셋이기 때문에 label이 없지만, rrcf 논문에서는 연휴나 기념일 등 8개의 이벤트를 anomaly로 간주하여 추가적인 비교를 할 수 있게 하였습니다. 먼저 기본적인 데이터 분석을 통해 해당 데이터셋의 형태에 대해 알아보겠습니다.

### 4.1. Data description


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("taxi_rides.csv",index_col=0)
df.index = pd.to_datetime(df.index)
data = df['value'].astype(float).values
```
```python
# Create events
events = {
'independence_day' : ('2014-07-04 00:00:00',
                      '2014-07-07 00:00:00'),
'labor_day'        : ('2014-09-01 00:00:00',
                      '2014-09-02 00:00:00'),
'labor_day_parade' : ('2014-09-06 00:00:00',
                      '2014-09-07 00:00:00'),
'nyc_marathon'     : ('2014-11-02 00:00:00',
                      '2014-11-03 00:00:00'),
'thanksgiving'     : ('2014-11-27 00:00:00',
                      '2014-11-28 00:00:00'),
'christmas'        : ('2014-12-25 00:00:00',
                      '2014-12-26 00:00:00'),
'new_year'         : ('2015-01-01 00:00:00',
                      '2015-01-02 00:00:00'),
'blizzard'         : ('2015-01-26 00:00:00',
                      '2015-01-28 00:00:00')
}
df['event'] = np.zeros(len(df))
for event, duration in events.items():
    start, end = duration
    df.loc[start:end, 'event'] = 1
```
```python
plt.figure(figsize=(60,16))
plt.plot(df['value'])
```
<p align="center"> <img src="https://github.com/cyp-ark/if/blob/main/figure/figure2.png?raw=true"> 

### 4.2. Anomaly detection using Isolation Forest
Scikit-learn의 IsolationForest 모듈을 이용해 이상치 탐색을 진행했습니다. Isolation Forest의 hyperparameter는 tree 개수 200개, 이상치 비율은 앞서 설정한 8개의 event의 비율만큼 설정해 모델을 실행합니다.

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(n_estimators=200,
                        contamination=df['event'].sum()/len(df),
                        random_state=0)

model.fit(df[['value']])
  
df['outliers']=model.predict(df[['value']])

scores = model.score_samples(df[['value']])
scores = pd.Series(-scores,index=(df.index))

```

```
fig,ax = plt.subplots(2,figsize=(70,16))

a = df.loc[df['outliers']==-1,['value']]
ax[0].plot(df.index,df['value'],color='black',label='normal')
ax[0].scatter(a.index,a['value'],color='red',label='abnormal',s=500)


for event, duration in events.items():
    start, end = duration
    ax[0].axvspan(start, end, alpha=0.3,color='springgreen')

ax[1].plot(scores.index,scores)

ax[0].legend()
plt.show()
```
<p align="center"> <img src="https://github.com/cyp-ark/if/blob/main/figure/figure3.png?raw=true"> 

   연한 초록색으로 되어있는 부분은 8개 event에 대한 기간을 나타낸 것입니다. 해당 기간의 값을 이상치로 탐지했는지, 그리고 다른 기간에서 어느 부분을 이상치로 판단했는지를 확인해보자면 우선 event가 있는 기간에 대한 판단을 맞춘 정도를 TP(True Positive) 비율로 계산하면 $38/536 = 0.071$정도로 매우 적은 비율로 이상치라고 탐지한 것을 확인할 수 있습니다. 해당 기간 이외에 다른 부분에서 어느 기간을 이상치로 탐지했는지 살펴보자면 주로 주변 기간대비 탑승자의 수가 가장 많은 피크시간대나 탑승자가 가장 적은 시간대를 이상치라고 탐지하는 것을 확인할 수 있습니다. 이러한 급격한 값의 움직임에 좀 더 강건하게 대응하기 위해 데이터에 Shingling을 적용해 다시 Isolation Forest 모델을 사용하여봅시다.

### 4.3. Anomaly detection using Isolation Forest with shingling (1)
하루동안 탑승자의 수가 가장 많은 시간대와 적은시간대를 이상치로 대부분 판단하였기 때문에 shingle의 $k=48$로 설정해 24시간동안의 탑승자의 수로 이상지 탐색을 진행하도록 하겠습니다. rrcf 패키지 안에 데이터를 shingle해주는 함수가 있지만 이번 튜토리얼에서는 for문을 통해 데이터를 shingle 해보도록 하겠습니다.
```python
#Shingle
n_shingling = 48

y = np.zeros(shape=(len(df)-n_shingling+1,n_shingling))
for i in range(len(df)-n_shingling+1):
    x = []
    for j in range(n_shingling):
        x.append(df.iloc[i+j].values[0])
    y[i] = x
```

```python
df_sh = pd.DataFrame(y,index=(df.iloc[(n_shingling - 1):].index))
#%%
model = IsolationForest(n_estimators=100,
                        contamination=df['event'].sum()/len(df),
                        random_state=0)
model.fit(df_sh)

scores = model.score_samples(df_sh)
scores = pd.Series(-scores,index=(df.iloc[(n_shingling - 1):].index))

df_sh['outliers'] = model.predict(df_sh)
```
```
fig,ax = plt.subplots(2,figsize=(70,16))

b = df_sh.loc[df_sh['outliers']==-1,[0]]
ax[0].plot(df.index,df['value'],color='black',label='normal')
ax[0].scatter(b.index,b[0],color='red',label='abnormal',s=500)

for event, duration in events.items():
    start, end = duration
    ax[0].axvspan(start, end, alpha=0.3,color='springgreen')

ax[1].plot(scores.index,scores)

ax[0].legend()
plt.show()
```
<p align="center"> <img src="https://github.com/cyp-ark/if/blob/main/figure/figure4.png?raw=true"> 

  모델의 결과를 확인해보면 shingling 이전의 모델에서 관측되었던 탑승자의 수가 가장 많은 시간대와 가장 적은 시간대를 이상치로 탐지하는 결과가 상당부분 해소된 것으로 확인됩니다. 8개의 event에 대한 TP 비율도 $129/536=0.241$정도로 이전에 비해 크게 향상된 것을 확인할 수 있습니다.

### 4.4. Anomaly detection using Isolation Forest with shingling (2)


## 5. Conclusion


## 6. Reference
1. Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation forest." 2008 eighth ieee international conference on data mining. IEEE, 2008. [[Link]](https://ieeexplore.ieee.org/abstract/document/4781136)
2. Guha, Sudipto, et al. "Robust random cut forest based anomaly detection on streams." International conference on machine learning. PMLR, 2016.[[Link]](https://proceedings.mlr.press/v48/guha16.html)
3. Implementation of the Robust Random Cut Forest Algorithm for anomaly detection on streams[[Link]](https://klabum.github.io/rrcf/)
4. Collins Kirui - Anomaly Detection Model on Time Series Data using Isolation Forest[[Link]](https://www.section.io/engineering-education/anomaly-detection-model-on-time-series-data-using-isolation-forest/)
5. Aayush Bajaj - Anomaly Detection in Time Series[[Link]](https://neptune.ai/blog/anomaly-detection-in-time-series)
6. HiddenBeginner (이동진) - [논문 리뷰] 실시간 이상 감지 모델 Robust Random Cut Forest (RRCF)[[Link]](https://hiddenbeginner.github.io/paperreview/2021/07/14/rrcf.html#ref4)
