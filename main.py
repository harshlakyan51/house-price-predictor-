from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load data and model
data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", "rb"))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = int(request.form.get('bhk'))
    bath = int(request.form.get('bath'))
    sqft = float(request.form.get('total_sqft'))

    input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                            columns=['location', 'total_sqft', 'bath', 'bhk'])

    prediction = pipe.predict(input_df)[0]*100000
    output = round(prediction, 2)

    return str(output)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

