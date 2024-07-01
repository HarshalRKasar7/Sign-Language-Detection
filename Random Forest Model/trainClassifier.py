import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle','rb'))
# print(data_dict['data'])
# print(data_dict['labels'])
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# print("X Train ", x_train)
# print("X Test", x_test)
# print("Y Train",y_train)
# print("Y test", y_test)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

print("\ny predict",y_predict)

accuracy = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(accuracy * 100))
# print(len(data_dict['labels']))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()