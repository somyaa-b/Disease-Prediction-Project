import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("disease_data.csv")  
print(df.columns) 


X = df.drop("Disease", axis=1)  
y = df["Disease"]  

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression()
dt_model = DecisionTreeClassifier()

lr_model.fit(X_train_scaled, y_train)
dt_model.fit(X_train_scaled, y_train)

with open("logistic_regression.pkl", "wb") as f:
    pickle.dump(lr_model, f)

with open("decision_tree.pkl", "wb") as f:
    pickle.dump(dt_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Models saved successfully!")


