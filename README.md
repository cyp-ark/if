# Isolation Forest를 이용한 시계열 데이터에서의 이상치 탐지

## 1. Introduction
이상치 탐지는 정상 데이터(normal data)와 이상 데이터(abnormal data)를 분류하는 문제입니다. 이상치 탐지 문제는 일반적인 분류(classification) 문제로 풀 수 있겠지만, 첫번째로 이상 데이터는 정상 데이터에 비해 매우 값이 적기 때문에 학습하는데에 사용하기 어렵습니다. 두번째로 전체 데이터 중에 어떠한 값이 이상치 데이터인지 판단하기 어려운 경우도 있습니다. 마지막으로, 이상치에 대한 패턴은 무궁무진하기 때문에 특정 이상치에 대해서만 학습을 할 경우 다른 형태의 이상치에 대해서는 잘 탐지하지 못하는 문제가 있습니다.

   이러한 이상치 탐지 문제를 해결하기 위해 밀도기반, 거리기반 알고리즘들이 많이 제시되었지만 데이터의 크키나 차원이 증가할 수록 계산량이 기하급수적으로 올라간다는 단점이 있습니다. 이를 해결하기 위해 2008년 트리 기반의 Isolation Forest가 제시되었습니다. 우리는 이번 튜토리얼을 통해 Isolation Forest와 이를 기반으로 실시간 스트리밍 환경에 적용한 Robust Random Cut Forest에 대해 이해하고, 두 모델을 이용한 시계열 데이터에서의 이상치 탐지에 대해 알아보겠습니다.



## 2. Isolation Forest
Isolation Forest는 트리 기반의 이상치 탐지 모델입니다. 트리라 함은 의사결정나무와 같이 어떠한 변수에 대해 어떠한 기준점으로 양쪽으로 나누는 모델을 뜻합니다. Isolation Forest의 아이디어는 단순합니다. "이상치일수록 해당 데이터를 고립시키는데에 필요한 분기 수가 적고 정상값일 수록 고립에 필요한 분기 수가 많을 것이다"라고 표현 할 수 있습니다. 이 아이디어가 어떤 의미를 가지는지 다음 그림을 통해 확인해보도록 하겠습니다.
   
<p align="center"> <img src="https://github.com/cyp-ark/if/blob/main/figure/figure1.png?raw=true" width="40%" height="40%">
  
  위의 그림에서 정상값인 $x_{i}$를 고립시키는데 총 12번의 분기가 필요한 것을 확인할 수 있습니다. 반대로 이상치인 $x_{o}$의 경우 단 4번의 분기만으로 해당 값을 고립시킬 수 있습니다. 즉 앞선 설명처럼 이상치일수록 고립시키는데에 필요한 분기 수가 적은 것을 확인할 수 있습니다. 어떠한 데이터에 대해 이 값이 얼마나 이상치에 가까운지에 대한 이상치 점수는 다음의 식과 같이 표현 할 수 있습니다.
$$s(x,n) = 2^{-\frac{E(h(x))}{c(n)}}$$
여기서 $E(h(x))$는 해당 x값을 고립시키는데 필요한 분기 수의 평균입니다. 이상치일 수록 해당 값이 낮아지기 때문에 이상치 점수 $s(x,n)$은 1에 가까운 값을 가질 것이고, 정상 데이터의 경우 평균 분기 수가 높을 것이기 때문에 0에 가까운 값을 가질 것 입니다.
  
  데이터의 크기가 커질경우 정상 데이터와 이상 데이터를를 트리를 통해 분기하는데 연산이 오래걸리는 문제가 있는데, 저자들의 주장에 따르면 트리 하나에 256개 정도의 데이터를 샘플링 해 이용해도 충분하다고 합니다. 

<p align="center"> <img src="https://github.com/cyp-ark/if/blob/main/figure/figure6.png?raw=true" width="60%" height="60%"> 

이를 통해 기존의 알고리즘들이 가지고 있던 문제를 해결할 뿐만 아니라 앙상블 기법을 통한 모델의 강건성도 확보 할 수 있습니다.
## 3. Robust Random Cut Forest
Robust Random Cut Forest는 Isolation Forest를 실시간 스트리밍 환경에 맞춰 변형시킨 모델입니다. 기존 Isolation Forest와 다른점은 우선 분기를 위해 랜덤하게 feature를 선택하는 과정에서 Isolation Forest는 모든 feature가 모두 같은 확률을 가진다면, Robust Random Cut Forest에서 데이터셋 $S$에 대해 i번째 feature $p_{i}$가 선택될 확률은 다음과 같이 표현됩니다.

  $$P(p_{i})=\frac{l_{i}}{\sum_{j}l_{j}}, l_{i}=max_{x \in S}x_{i}-min_{x \in S}x_{i}$$
즉 feature $p$의 range가 클 수록 트리를 만드는데에 있어 해당 feature가 선택될 확률이 늘어난다는 것 입니다. 이로 인해 시간에 따라 변동성이 큰 데이터에 잘 대응하여 트리를 만들 수 있고, 따라서 RRCF가 실시간 스트리밍 데이터에 적합한 알고리즘이라고 해당 논문의 저자들은 주장합니다. 이 외에도 저자들은 Isolation Forest에서 세가지 정도를 추가적으로 제안합니다. 그 내용은 아래와 같습니다.
   
### 3.1. 실시간 스트리밍 데이터
현재 시점 $t$에서 tree를 만드는데 256개의 sample을 사용한다고 하면 데이터셋 $S_{t}=\lbrace \mathbf{x_{t-255}}, \mathbf{x_{t-254}},\ldots,\mathbf{x_{t}}\rbrace$을 사용하여 트리 $\mathcal{T(S_{t})}$를 만들수 있을 것 입니다. 이 때 $t+1$시점의 데이터가 들어오게 된다면 일반적인 Isolation Forest 모델은 데이터셋 $S_{t+1}=\lbrace \mathbf{x_{t-254}}, \mathbf{x_{t-253}},\ldots,\mathbf{x_{t+1}}\rbrace$을 이용해 새로운 트리 $\mathcal{T(S_{t+1})}$을 만들어야할 것입니다. 그러나 저자들은 기존의 트리 $\mathcal{T(S_{t})}$에서 $\mathbf{x_{t+1}}$을 추가 한 후 $\mathbf{x_{t-255}}$를 삭제해서 만든 새로운 트리 $\mathcal{T'(S_{t+1})}$를 사용하는 것을 제안합니다. 이를 통해 실시간 데이터가 들어와도 새롭게 트리를 만드는 것이 아닌 기존 트리를 활용할 수 있어 실시간 스트리밍 데이터에 적합한 알고리즘이라고 합니다.
### 3.2. CoDisp
Robust Random Cut Forest에서는 Isolation Forest의 이상치 점수를 Collusive Displacement(CoDisp)를 사용합니다. Codisp를 이해하기 위해서는 우선 Disp부터 알아야합니다.

<p align="center"> <img src="https://github.com/cyp-ark/if/blob/main/figure/figure7.png?raw=true" width="40%" height="40%">

위 그림의 트리에서 데이터 $x$가 만약 삭제가 된다면 어떻게 될까요? $x$의 자매노드인 $C$의 데이터들은 depth가 1씩 줄어들 것 입니다. 이렇게 데이터 $x$가 삭제될때 데이터셋 S로 만든 트리에서의 depth 변화를 $Disp(x,S)$라고 정의합니다. Isolation Forest에서도 이상치일 수록 해당 데이터를 고립시키는데에 적은 분기가 사용되기 때문에 depth가 작을 것이고 이상치 데이터일수록 그 데이터가 삭제가 된다면 $Disp(x,S)$값은 커지게 될것입니다.
   
   RRCF의 저자들은 Disp에서 한단계 더 나아가 데이터 $x$를 포함하는 모든 부분집합 $C$에 대해 Disp를 계산하고 이것을 그 집합의 크기인 $|C|$로 나눈 CoDisp를 제안합니다. 수식으로 나타내면 다음과 같습니다.
   $$CoDisp(C,S)=\mathbb{E_{T}}[max_{x \in C \subset S}\frac{Disp(C,S)}{|C|}]$$
Disp대신 CoDisp를 사용함으로써 $x$의 상위 노드까지 고려해 이상치 점수로 사용할 수 있습니다.
### 3.3. Shingling
마지막으로 제안한 방법은 Shingling으로 주로 자연어 처리나 시계열 데이터 분석에 많이 사용하는 방법입니다. Shingling은 최근 $k$개의 값을 열벡터로 결합하여 feature 벡터로 사용하는 방법입니다. 예를 들어 $k=4$일 경우 첫번째 데이터는 $(t_{1},t_{2},t_{3},t_{4})^{T}$, 두번째 데이터는 $(t_{2},t_{3},t_{4},t_{5})^{T}$의 모양으로 변환되게 됩니다. Sliding window 같이 한 시점 단위로 shifting 되면서 vector를 구성하게 되고, 이는 시계열 데이터를 캡슐화 하여 noise에 강건하게 대응할 수 있습니다. 또한 이 값이 정상 범주에서 벗어날 경우 이상치로 탐색 할 수 있습니다.
   

  
<p align="center"> <img src="https://github.com/cyp-ark/if/blob/main/figure/figure5.png?raw=true" width="60%" height="60%"> 



## 4. NYC taxi data
본 튜토리얼에서는 Isolation Forest와 Robust Random Cut Forest를 이용한 시계열 데이터 이상치 탐지에 대해 알아보겠습니다. Isolation Forest의 경우 본래 시계열 데이터 이상치 탐지를 위해 개발된 알고리즘이 아니므로 다양한 시도를 수행하고 이를 Robust Random Cut Forest의 결과와 비교하고자 합니다. 사용된 데이터셋은 unsupervised anomaly detection 분야에서 자주 사용되는 '뉴욕시 택시 탑승객 수'로 2014년 7월부터 2015년 1월까지 뉴욕시 택시 탑승객 수를 30분 단위로 측정한 데이터입니다. 먼저 기본적인 데이터 분석을 통해 해당 데이터셋의 형태에 대해 알아보겠습니다.

### 4.1. Data description


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("taxi_rides.csv",index_col=0)
df.index = pd.to_datetime(df.index)
data = df['value'].astype(float).values

plt.figure(figsize=(60,16))
plt.plot(df['value'])
```
<p align="center"> <img src="https://github.com/cyp-ark/if/blob/main/figure/figure2.png?raw=true"> 

모든 기간에 대해 택시 탑승자 수의 변화를 살펴보자면 요일별로, 시간대 별로 패턴이 존재해 보인다. 데이터를 좀 더 정리해 요일별, 시간대 별 평균 탑승객 수를 알아보자.


```python
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour']=df.timestamp.dt.hour
df['weekday']=pd.Categorical(df.timestamp.dt.strftime('%A'), 
   categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'], ordered=True)

plt.figure(figsize=(12,8))
plt.plot(df[['value','weekday']].groupby('weekday').mean())
```

<p align="center"> <img src="https://github.com/cyp-ark/if/blob/main/figure/figure8.png?raw=true" width="40%" height="40%">

요일별 평균 탑승객 수를 살펴보자면 월요일이 가장 탑승객 수가 적고, 점점 탑승객 수가 늘어나 토요일에 가장 많은 사람이 택시를 타고다니는 것을 알 수가 있다.

```python
plt.figure(figsize=(12,8))
plt.plot(df[['value','hour']].groupby('hour').mean())
plt.xticks(range(24))
plt.show()
```

<p align="center"> <img src="https://github.com/cyp-ark/if/blob/main/figure/figure9.png?raw=true" width="40%" height="40%">

시간대 별 평균 탑승객 수를 보자면 오전 5시에 가장 탑승객 수가 적고, 오후 6시에 가장 탑승객 수가 많은 것을 확인할 수 있다.
   
   본래 뉴욕시 택시 데이터는 unsupervised 데이터 셋이기 때문에 label이 없지만, Robust Random Cut Forest 논문에서는 연휴나 기념일 등 8개의 이벤트를 anomaly로 간주하여 추가적인 비교를 할 수 있게 하였습니다. 해당 기간을 다음과 같은 코드를 통해 labeling 할 수 있습니다.
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
이전에 비해 이상치 탐지 성능이 상승했지만, 좀 더 성능을 끌어올리기 위해 기존 30분 간격으로 측정된 데이터를 1시간간격으로 smoothing한 후 이상치 탐지를 진행해보겠습니다.
```python
#Shingle
n_shingling = 24

y = np.zeros(shape=(len(df2)-n_shingling+1,n_shingling))
for i in range(len(df2)-n_shingling+1):
    x = []
    for j in range(n_shingling):
        x.append(df2.iloc[i+j].values[0])
    y[i] = x

df_h = pd.DataFrame(y,index=(df2.iloc[(n_shingling - 1):].index))
#%%
model = IsolationForest(n_estimators=100,
                        contamination=df2['event'].sum()/len(df_h),
                        random_state=0)

model.fit(df_h)
```
```python
scores = model.score_samples(df_h)
scores = pd.Series(-scores, index=(df_h.index))
fig, ax = plt.subplots(2, figsize=(70, 16))

df_h['outliers'] = model.predict(df_h)

a = df_h.loc[df_h['outliers'] == -1]

ax[0].plot(df2.index, df2['value'], color='black', label='normal')
ax[0].scatter(a.index, a[0], color='red', label='abnormal', s=500)

for event, duration in events.items():
    start, end = duration
    ax[0].axvspan(start, end, alpha=0.3, color='springgreen')

ax[1].plot(scores.index, scores)

plt.legend()
plt.show
```
<p align="center"> <img src="https://github.com/cyp-ark/if/blob/main/figure/figure10.png?raw=true"> 

이전 30분 간격 데이터를 사용할때와 결과가 비슷한 것을 확인할 수 있다. 8개의 event에 대해 TP 비율을 확인하면 $62/272=0.228$로 소폭 감소한 것을 확인 할 수 있다. Event수에 대해서도 이전에는 6개의 event를 detect 했다면 이번 모델은 4개만 detect한 것을 알 수 있다. 

### 4.5. (Additional) Anomaly detection using Robust Random Cut Forest
마지막으로 Robust Random Cut Forest를 이용해 이상치 탐지를 진행하려고 한다. 논문에서 구현 된 코드를 토대로 진행하려고 했으나 API 안에 numpy 버전 충돌로 인해 직접 구현하지는 못하고, 논문 원문을 그대로 발췌해 소개하려고 합니다.
```python
# Set tree parameters
num_trees = 200
shingle_size = 48
tree_size = 1000

# Use the "shingle" generator to create rolling window
points = rrcf.shingle(data, size=shingle_size)
points = np.vstack([point for point in points])
n = points.shape[0]
sample_size_range = (n // tree_size, tree_size)

forest = []
while len(forest) < num_trees:
    ixs = np.random.choice(n, size=sample_size_range,
                           replace=False)
    trees = [rrcf.RCTree(points[ix], index_labels=ix)
             for ix in ixs]
    forest.extend(trees)
    
avg_codisp = pd.Series(0.0, index=np.arange(n))
index = np.zeros(n)

for tree in forest:
    codisp = pd.Series({leaf : tree.codisp(leaf)
                        for leaf in tree.leaves})
    avg_codisp[codisp.index] += codisp
    np.add.at(index, codisp.index.values, 1)
    
avg_codisp /= index
avg_codisp.index = taxi.iloc[(shingle_size - 1):].index
```
<p align="center"> <img src = "https://s3.us-east-2.amazonaws.com/mdbartos-img/rrcf/taxi.png">

Robust Random Cut Forest와 Isolation Forest의 결과를 비교하자면 평균적인 이상치 점수가 Isolation Forest가 높고, 대신 이상치에서의 이상치 점수는 RRCF가 높은 것을 확인할 수 있다. Isolation Forest의 경우 위의 예시들로부터도 확인 할 수 있지만 어떤 지점이 이상치인지 명확하게 확인 할 수 없다는 단점이 있다.

## 5. Conclusion
이번 튜토리얼을 통해 Isolation Forest를 이용한 시계열 데이터인 뉴욕시 택시 탑승객 데이터에 대한 이상치 탐지를 진행해보았다. 또한 이전의 밀도기반, 거리기반에서 벗어나 트리 기반의 Isolation Forest와 이를 시계열 데이터에 맞게 변형시킨 Robust Random Cut Forest 알고리즘에 대해 알아보는 시간을 가졌다. 이를 통해 기존의 다른 이상치 탐색 알고리즘을 적절히 변형한다면 시계열 데이터에 적합한 알고리즘을 만들어 낼 수 있다는 점을 확인했으며, RRCF를 이용해 다양한 분야에서 이상치 탐지를 적용 할 수 있을 것이다.

## 6. Reference
1. Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation forest." 2008 eighth ieee international conference on data mining. IEEE, 2008. [[Link]](https://ieeexplore.ieee.org/abstract/document/4781136)
2. Guha, Sudipto, et al. "Robust random cut forest based anomaly detection on streams." International conference on machine learning. PMLR, 2016.[[Link]](https://proceedings.mlr.press/v48/guha16.html)
3. Implementation of the Robust Random Cut Forest Algorithm for anomaly detection on streams[[Link]](https://klabum.github.io/rrcf/)
4. Collins Kirui - Anomaly Detection Model on Time Series Data using Isolation Forest[[Link]](https://www.section.io/engineering-education/anomaly-detection-model-on-time-series-data-using-isolation-forest/)
5. Aayush Bajaj - Anomaly Detection in Time Series[[Link]](https://neptune.ai/blog/anomaly-detection-in-time-series)
6. HiddenBeginner (이동진) - [논문 리뷰] 실시간 이상 감지 모델 Robust Random Cut Forest (RRCF)[[Link]](https://hiddenbeginner.github.io/paperreview/2021/07/14/rrcf.html#ref4)
