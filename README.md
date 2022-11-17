# Isolation Forest를 이용한 time series data에서의 anomaly detection
이상치 탐지



## 1.Introduction

## 2.Isolation Forest

<p align="center"> <img src="https://github.com/cyp-ark/if/blob/main/figure/figure1.png?raw=true" width="40%" height="40%">


## 3. Robust Random Cut Forest
### 3.1.Shingling
Shingling은 최근 $k$개의 값을 열벡터로 결합하여 feature 벡터로 사용하는 방법입니다. 예를 들어 $k=4$일 경우 첫번째 데이터는 $(t_{1},t_{2},t_{3},t_{4})^{T}$, 두번째 데이터는 $(t_{2},t_{3},t_{4},t_{5})^{T}$의 모양으로 변환되게 됩니다. Sliding window 같이 한 시점 단위로 shifting 되면서 vector를 구성하게 되고, 이는 시계열 데이터를 캡슐화 하여 noise에 강건하게 대응할 수 있습니다. 또한 이 값이 정상 범주에서 벗어날 경우 이상치로 탐색 할 수 있습니다.
  
<p align="center"> <img src="https://github.com/cyp-ark/if/blob/main/figure/figure5.png?raw=true" width="60%" height="60%"> 

## 4.
사용된 데이터셋은 unsupervised anomaly detection 분야에서 자주 사용되는 '뉴욕시 택시 탑승객 수'로 2014년 7월부터 2015년 1월까지 뉴욕시 택시 탑승객 수를 30분 단위로 측정한 데이터입니다. 원래는 unsupervised 데이터 셋이기 때문에 label이 없지만, rrcf 논문에서는 연휴나 기념일 등 8개의 이벤트를 anomaly로 간주하여 추가적인 비교를 할 수 있게 하였습니다. 먼저 기본적인 Isolation Forest로 이상치 탐색을 수행하도록 하겠습니다.

### 4.1. Isolation Forest

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

   연한 초록색으로 되어있는 부분은 8개 event에 대한 기간을 나타낸 것입니다. 해당 기간의 값을 이상치로 탐지했는지, 그리고 다른 기간에서 어느 부분을 이상치로 판단했는지를 확인해보자면 우선 event가 있는 기간에 대한 판단을 맞춘 정도를 TP(True Positive) 비율로 계산하면 $38/536 = 0.071$정도로 매우 적은 비율로 이상치라고 탐지한 것을 확인할 수 있습니다. 해당 기간 이외에 다른 부분에서 어느 기간을 이상치로 탐지했는지 살펴보자면 주로 주변 기간대비 탑승자의 수가 가장 많은 피크시간대나 가장 적은 시간대를 이상치라고 탐지하는 것을 확인할 수 있습니다. 이러한 

### 4.2. Shingling
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
                        #contamination=0.004,
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


## 5.Conclusion


## 6. Reference
1. Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation forest." 2008 eighth ieee international conference on data mining. IEEE, 2008. [[Link]](https://ieeexplore.ieee.org/abstract/document/4781136)
2. Guha, Sudipto, et al. "Robust random cut forest based anomaly detection on streams." International conference on machine learning. PMLR, 2016.[[Link]](https://proceedings.mlr.press/v48/guha16.html)
3. Implementation of the Robust Random Cut Forest Algorithm for anomaly detection on streams[[Link]](https://klabum.github.io/rrcf/)
4. Collins Kirui - Anomaly Detection Model on Time Series Data using Isolation Forest[[Link]](https://www.section.io/engineering-education/anomaly-detection-model-on-time-series-data-using-isolation-forest/)
5. Aayush Bajaj - Anomaly Detection in Time Series[[Link]](https://neptune.ai/blog/anomaly-detection-in-time-series)
6. HiddenBeginner (이동진) - [논문 리뷰] 실시간 이상 감지 모델 Robust Random Cut Forest (RRCF)[[Link]](https://hiddenbeginner.github.io/paperreview/2021/07/14/rrcf.html#ref4)
