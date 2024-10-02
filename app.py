from flask import Flask, render_template, request
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
app = Flask(__name__)


# Load the dataset
data = pd.read_excel("data/data.xlsx")

# Create predictions DataFrame


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    state = request.form['state']
    
    # Retrieve user-entered monthly production for 2023
    monthly_production_2023 = [
        float(request.form['jan']),
        float(request.form['feb']),
        float(request.form['mar']),
        float(request.form['apr']),
        float(request.form['may']),
        float(request.form['jun']),
        float(request.form['jul']),
        float(request.form['aug']),
        float(request.form['sep']),
        float(request.form['oct']),
        float(request.form['nov']),
        float(request.form['dec'])
    ]

    # Step 1: Predict the 2024 production for the state using the ARIMA model
    state_data = data[data['States/UTs'] == state].drop(['S. No.', 'States/UTs'], axis=1).T
    state_data.columns = ['Wool Production']
    state_data.index = pd.date_range(start='2016', periods=len(state_data), freq='Y')

    model = ARIMA(state_data, order=(1, 1, 1))
    model_fit = model.fit()
    a = model_fit.forecast(steps=1)[0]

    # Step 2: Retrieve the 2022-23 production value for the state from the dataset
    b = data.loc[data['States/UTs'] == state, '2022-23'].values[0]

    # Step 3: Predict the 2024 production for the user’s company for each month
    predicted_production_2024 = []
    for production in monthly_production_2023:
        predicted_value = (a / b) * production
        predicted_production_2024.append(predicted_value)

    # Month names
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]

    # Combine months with their predicted values
    month_wise_prediction = dict(zip(months, predicted_production_2024))

    # Display results or return them in the response
    return render_template('result.html', name=name, state=state, predictions=month_wise_prediction)

if __name__ == '__main__':
    app.run(debug=True)