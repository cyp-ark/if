# Isolation Forest를 이용한 time series data에서의 anomaly detection

## 1.Introduction

## 2.Isolation Forest

## 3.

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
    
df
```
```python
plt.figure(figsize=(60,16))
plt.plot(df['value'])
```


## 4. Reference
1. Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation forest." 2008 eighth ieee international conference on data mining. IEEE, 2008. [[Link]](https://ieeexplore.ieee.org/abstract/document/4781136)
2. Guha, Sudipto, et al. "Robust random cut forest based anomaly detection on streams." International conference on machine learning. PMLR, 2016.[[Link]](https://proceedings.mlr.press/v48/guha16.html)
3. Implementation of the Robust Random Cut Forest Algorithm for anomaly detection on streams[[Link]](https://klabum.github.io/rrcf/)
4. Collins Kirui - Anomaly Detection Model on Time Series Data using Isolation Forest[[Link]](https://www.section.io/engineering-education/anomaly-detection-model-on-time-series-data-using-isolation-forest/)
5. Aayush Bajaj - Anomaly Detection in Time Series[[Link]](https://neptune.ai/blog/anomaly-detection-in-time-series)
