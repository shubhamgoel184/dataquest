import numpy as np
import pandas as pd

train_data=pd.read_csv('F:\\data_science\\dataquest\\train_DaEJRFg.csv')
test_data=pd.read_csv('F:\\data_science\\dataquest\\test_TQDFDgg.csv')

d=train_data['incident_date']+' '+train_data['incident_time']
d1=test_data['incident_date']+' '+test_data['incident_time']
d=pd.to_datetime(d)
d1=pd.to_datetime(d1)

del train_data['incident_date']
del train_data['incident_time']
del test_data['incident_date']
del test_data['incident_time']
train_data['datetime']=d
test_data['datetime']=d1

train_data['hour_of_accident']=train_data['datetime'].dt.hour
test_data['hour_of_accident']=test_data['datetime'].dt.hour

train_data['weekday']=train_data['datetime'].dt.dayofweek
test_data['weekday']=test_data['datetime'].dt.dayofweek


d=train_data['criticality'].groupby(train_data['incident_location']).count().sort_values()[-20:].index
lst=[]
for i in train_data['incident_location']:
    if i in d:
        lst.append(i)
    else:
        lst.append('other')
lst2=[]
for i in test_data['incident_location']:
    if i in d:
        lst2.append(i)
    else:
        lst2.append('other')

del train_data['incident_location']
del test_data['incident_location']
train_data['incident_location']=lst
test_data['incident_location']=lst2

del train_data['road_type']
del test_data['road_type']

categorical_variables=[x for x in train_data.columns if train_data[x].dtypes=='O' and x not in ['victim_id','datetime']]
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
d = defaultdict(LabelEncoder)
encoded_train_data=pd.DataFrame(data=train_data,copy=True)
encoded_test_data=pd.DataFrame(data=test_data,copy=True)
encoded_train_data[categorical_variables] =encoded_train_data[categorical_variables].apply(lambda x: d[x.name].fit_transform(x))
encoded_test_data[categorical_variables] =encoded_test_data[categorical_variables].apply(lambda x: d[x.name].transform(x))

independent_variables=[x for x in train_data.columns if x not in ['victim_id','datetime','criticality']]
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
def scorer(estimator,X,y):
    y1=np.array(estimator.predict(X))
    score=roc_auc_score(y,y1)
    return score

from sklearn.ensemble import AdaBoostClassifier

adam=AdaBoostClassifier(learning_rate=2,n_estimators=48,random_state=0)

adam.fit(encoded_train_data[independent_variables],encoded_train_data['criticality'])
test_predictions=adam.predict(encoded_test_data[independent_variables])
victim_id=test_data['victim_id']
submission=pd.DataFrame({
    'victim_id':victim_id,
    'criticality': test_predictions
})
submission.to_csv('dataquest_submission4.csv', index=False)
