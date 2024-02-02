import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data_dict = pickle.load(open('./data.pickle', 'rb'))
X, y = np.asarray(data_dict['data']), np.asarray(data_dict['labels'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_test)
print(f'Accuracy on test data: {accuracy_score(y_test, y_pred) * 100}%')
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()