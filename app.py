from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

model = pickle.load(open('stored_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')


@app.route('/week1')
def week1():
    return render_template('week1.html')

@app.route('/week2')
def week2():
    return render_template('week2.html')

@app.route('/week3')
def week3():
    return render_template('week3.html')

@app.route('/week4')
def week4():
    return render_template('week4.html')

@app.route('/week5')
def week5():
    return render_template('week5.html')

@app.route('/week6')
def week6():
    return render_template('week6.html')

@app.route('/model1')
def model1():
    return render_template('model1.html')

@app.route('/model2')
def model2():
    return render_template('model2.html')

@app.route('/model3')
def model3():
    return render_template('model3.html')

@app.route('/prediction', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction = model.predict(final)
    output = f"The predicted crop is {prediction[0]}"
    return render_template('model1.html', prediction_text=output)  # Adjust your template to display the prediction_text variable


# Load the fertilizer recommendation model
fertilizer_model = pickle.load(open('stored_model_fertilizer.pkl', 'rb'))  # Ensure the filename is correct

# Encoders for the categorical variables
soil_type_encoder = LabelEncoder()
crop_type_encoder = LabelEncoder()

# Fit the encoders with the categories present in the dataset
soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
crop_types = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']

soil_type_encoder.fit(soil_types)
crop_type_encoder.fit(crop_types)

@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer():
    # Extract form values
    form_values = request.form.to_dict()
    
    # Convert form values to appropriate data types
    int_features = [
        float(form_values['Temperature']),
        float(form_values['Humidity']),
        float(form_values['Moisture']),
        soil_type_encoder.transform([form_values['soil_type']])[0],
        crop_type_encoder.transform([form_values['crop_type']])[0],
        float(form_values['Nitrogen']),
        float(form_values['Potassium']),
        float(form_values['Phosphorous']),
    ]
    
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    myprediction = fertilizer_model.predict(final)
    output = f"The recommended fertilizer is {myprediction[0]}"
    return render_template('model2.html', prediction_text=output)

if __name__ == '__main__':
    app.run(debug=True)  # Consider switching off debug mode in production