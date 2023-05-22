from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

app = Flask(__name__)

def predict_Parkinson(name, age, sex, test_time, motor_UPDRS, total_UPDRS, Jitter_percent, Jitter_Abs, Jitter_RAP, Jitter_PPQ5, Jitter_DDP, Shimmer, Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, Shimmer_APQ11, Shimmer_DDA, NHR, HNR, RPDE, DFA, PPE):
    """Data Collection & Analysis"""

    # loading the data from csv file to a Pandas DataFrame
    parkinsons_data = pd.read_csv('parkinsons.csv')

    # number of rows and columns in the dataframe
    num_rows, num_cols = parkinsons_data.shape

    """Data Pre-Processing

    Separating the features & Target
    """

    X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
    Y = parkinsons_data['status']

    """Splitting the data to training data & Test data"""

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    """Model Training

    Support Vector Machine Model
    """

    model = svm.SVC(kernel='linear')

    # training the SVM model with training data
    model.fit(X_train, Y_train)

    """Model Evaluation

    Accuracy Score
    """

    # accuracy score on training data
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

    # accuracy score on test data
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

    """Building a Predictive System"""

    input_data = [test_time, motor_UPDRS, total_UPDRS, Jitter_percent, Jitter_Abs, Jitter_RAP, Jitter_PPQ5, Jitter_DDP, Shimmer, Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, Shimmer_APQ11, Shimmer_DDA, NHR, HNR, RPDE, DFA, PPE]

    # Convert input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make prediction using the loaded model
    prediction = model.predict(input_data_reshaped)

    if prediction[0] == 0:
        result = "The person does not have Parkinson's disease."
    else:
        result = "The person has Parkinson's disease."

    """Result Analysis"""

    result_analysis = f"""*** Parkinson's Disease Prediction Analysis ***

    Data Analysis:
    - Number of rows in the dataset: {num_rows}
    - Number of columns in the dataset: {num_cols}
    - Training data accuracy: {training_data_accuracy}
    - Test data accuracy: {test_data_accuracy}

    Input Information:
    - Name: {name}
    - Age: {age}
    - Sex: {sex}
    - Test Time: {test_time}
    - Motor UPDRS: {motor_UPDRS}
    - Total UPDRS: {total_UPDRS}
    - Jitter(%): {Jitter_percent}
    - Jitter(Abs): {Jitter_Abs}
    - Jitter(RAP): {Jitter_RAP}
    - Jitter(PPQ5): {Jitter_PPQ5}
    - Jitter(DDP): {Jitter_DDP}
    - Shimmer: {Shimmer}
    - Shimmer(dB): {Shimmer_dB}
    - Shimmer(APQ3): {Shimmer_APQ3}
    - Shimmer(APQ5): {Shimmer_APQ5}
    - Shimmer(APQ11): {Shimmer_APQ11}
    - Shimmer(DDA): {Shimmer_DDA}
    - NHR: {NHR}
    - HNR: {HNR}
    - RPDE: {RPDE}
    - DFA: {DFA}
    - PPE: {PPE}

    Prediction Result:
    - {result}
    """

    """Saving the trained model"""

    filename = 'parkinsons_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    return result_analysis


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/blood_sugar')
def blood_sugar():
    return render_template('blood_sugar.html')

@app.route('/heart_disease')
def heart_disease():
    return render_template('heart_disease.html')

@app.route('/parkinsons_disease', methods=['GET', 'POST'])
def parkinsons_disease():
    if request.method == 'POST':
        input1 = request.form.get('input1')
        # Retrieve other form fields similarly
        
        # Print the form data in the console
        print(f'Input 1: {input1}')
        # Print other form fields similarly
        
        # Perform your prediction logic here
        
        return 'Prediction result'
    
    return render_template('parkinsons_disease.html')

if __name__ == '__main__':
    app.run()
