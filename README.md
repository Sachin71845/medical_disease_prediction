# Medical Disease Prediction using Deep Learning

## 📌 Project Overview
This project is a **Medical Disease Prediction System** that uses a **Deep Neural Network (DNN)** to classify whether a patient has a disease (e.g., diabetes) based on medical parameters. The model is built using **TensorFlow/Keras** and trained on the **Pima Indians Diabetes Dataset**.

## 🚀 Features
✅ Deep Learning model using **DNN (Deep Neural Network)**  
✅ Data Preprocessing & **Feature Scaling** for better accuracy  
✅ Visualizations for **Exploratory Data Analysis (EDA)**  
✅ Model Training with **Adam optimizer & Dropout regularization**  
✅ **Save & Load** trained model for future predictions  
✅ **New Patient Prediction** using real-time input  
✅ **Deployable as a Flask API**  

---

## 🛠️ Tech Stack
- **Python** (NumPy, Pandas, Matplotlib, Seaborn)
- **TensorFlow & Keras** (Deep Learning Model)
- **Scikit-Learn** (Feature Scaling, Model Evaluation)
- **Flask** (For API Deployment)

---

## 📂 Dataset
The dataset used is the **Pima Indians Diabetes Dataset**, which contains medical records with the following features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin Level
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (1 = Diabetic, 0 = Not Diabetic)

You can download the dataset from [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv).

---

## 🔧 Installation
Follow these steps to set up the project on your local machine or **Google Colab**:

### 1️⃣ Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow keras flask
```

### 2️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/medical-disease-prediction.git
cd medical-disease-prediction
```

### 3️⃣ Run the Model in Colab
Upload the Jupyter Notebook (`Medical_Disease_Prediction.ipynb`) to Google Colab and run the cells step by step.

---

## 📊 Exploratory Data Analysis (EDA)
Before training the model, we analyze the dataset with **visualizations**:
```python
# Check missing values
df.isnull().sum()

# Class distribution
sns.countplot(x=df['Outcome'])
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
```

---

## 🎯 Model Architecture
The Deep Learning model consists of:
- **Input Layer:** 8 features
- **Hidden Layers:** 64, 32, 16 neurons (ReLU activation)
- **Dropout Layers:** To prevent overfitting
- **Output Layer:** Sigmoid activation for binary classification

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

Compile and train the model:
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=32)
```

---

## 📈 Model Evaluation
After training, evaluate model performance:
```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
```
Generate confusion matrix and classification report:
```python
from sklearn.metrics import confusion_matrix, classification_report
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))
```

---

## 🔮 Making Predictions
To predict whether a new patient has the disease:
```python
new_patient = np.array([[2, 130, 85, 23, 90, 28.1, 0.5, 45]])
new_patient_scaled = scaler.transform(new_patient)
prediction = model.predict(new_patient_scaled)
print("Prediction:", "Diabetic" if prediction[0] > 0.5 else "Not Diabetic")
```

---

## 🌍 Deploying as an API
Deploy the model using Flask:
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = keras.models.load_model("diabetes_dnn_model.h5")
scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array(data["features"]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    result = "Diabetic" if prediction[0] > 0.5 else "Not Diabetic"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
```
Run the API:
```bash
python app.py
```

---

## 🎯 Future Improvements
🔹 Improve accuracy by **hyperparameter tuning**  
🔹 Use a **CNN/LSTM model** for advanced feature learning  
🔹 Train on **larger datasets** (e.g., heart disease, kidney disease)  
🔹 Create a **Power BI dashboard** for real-time monitoring  

---

Feel free to contribute by **forking the repo**, creating a pull request, or raising issues! 🚀

---

## ⭐ Show Your Support!
If you liked this project, **give it a star ⭐ on GitHub** and share it with others!

