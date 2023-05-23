from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

app = Flask(__name__)


def train_parkinsons_model():
    # Loading the data from the CSV file
    parkinsons_data = pd.read_csv('parkinsons.csv')

    # Separating the features and target
    X = parkinsons_data.drop(columns=['name','status'], axis=1)
    Y = parkinsons_data['status']

    # Splitting the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # Training the SVM model
    model = svm.SVC(kernel='linear')
    model.fit(X_train, Y_train)

    # Saving the trained model
    filename = 'parkinsons_model.sav'
    pickle.dump(model, open(filename, 'wb'))

def predict_parkinsons(name, features):
    # Load the trained model
    loaded_model = pickle.load(open('parkinsons_model.sav', 'rb'))

    # Create a DataFrame with the input features
    input_data = pd.DataFrame([features], columns=[
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1',
        'spread2', 'D2', 'PPE'
    ])

    # Make the prediction
    prediction = loaded_model.predict(input_data)

    if prediction[0] == 0:
        return f"{name} does not have Parkinson's disease."
    else:
        return f"{name} has Parkinson's disease."

# Train the model (only needs to be done once)
train_parkinsons_model()

# Example usage:
name = "John Doe"
features = [197.07600, 206.89600, 192.05500, 0.00289, 0.00001, 0.00166, 0.00168, 0.00498,
            0.01098, 0.09700, 0.00563, 0.00680, 0.00802, 0.01689, 0.00339, 26.77500,
            0.422229, 0.741367, -7.348300, 0.177551, 1.743867, 0.085569]

result = predict_parkinsons(name, features)
print(result)



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
        name = request.form.get('name')
        age = request.form.get('age')
        sex = request.form.get('sex')
        
        input1 = float(request.form.get('MDVP_Fo_Hz'))
        input2 = float(request.form.get('MDVP_Fhi_Hz'))
        input3 = float(request.form.get('MDVP_Flo_Hz'))
        input4 = float(request.form.get('Jitter_percent'))
        input5 = float(request.form.get('Jitter_Abs'))
        input6 = float(request.form.get('Jitter_RAP'))
        input7 = float(request.form.get('ppq'))
        input8 = float(request.form.get('Jitter_DDP'))
        input9 = float(request.form.get('Shimmer'))
        input10 = float(request.form.get('Shimmer_dB'))
        input11 = float(request.form.get('Shimmer_APQ3'))
        input12 = float(request.form.get('Shimmer_APQ5'))
        input13 = float(request.form.get('MDVP_APQ'))
        input14 = float(request.form.get('Shimmer_DDA'))
        input15 = float(request.form.get('NHR'))
        input16 = float(request.form.get('HNR'))
        input17 = float(request.form.get('RPDE'))
        input18 = float(request.form.get('DFA'))
        input19 = float(request.form.get('Spread1'))
        input20 = float(request.form.get('Spread2'))
        input21 = float(request.form.get('D2'))
        input22 = float(request.form.get('PPE'))

        # Example usage:
        features = [
            input1, input2, input3, input4, input5, input6, input7, input8, input9, input10,
            input11, input12, input13, input14, input15, input16, input17, input18, input19,
            input20, input21, input22
        ]
        result = predict_parkinsons(name, features)
        

        print(result)
        
        return result
        
    return render_template('parkinsons_disease.html')

if __name__ == '__main__':
    app.run()
