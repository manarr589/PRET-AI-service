import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# 1. Load your Excel/CSV file
try:
    data = pd.read_csv('waste_data.csv')
    # Make sure your CSV columns are named: 'Date' and 'Amount_KG'
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date_Ordinal'] = data['Date'].map(datetime.toordinal)
    
    # 2. Training the Model
    X = data[['Date_Ordinal']] 
    y = data['Amount_KG']      
    model = LinearRegression()
    model.fit(X, y)

    # 3. Calculate the FIRST day of the NEXT month automatically
    today = datetime.now()
    next_month_first_day = (today + relativedelta(months=1)).replace(day=1)
    
    # 4. Predict for that date
    prediction_date_num = np.array([[next_month_first_day.toordinal()]])
    prediction = model.predict(prediction_date_num)

    # 5. Output in English
    print("\n" + "="*40)
    print("      WASTE PREDICTION      ")
    print("="*40)
    print(f"Current Date: {today.strftime('%Y-%m-%d')}")
    print(f"Target Prediction Date: {next_month_first_day.strftime('%Y-%m-%d')}")
    print(f"Predicted Waste Amount: {prediction[0]:.2f} KG")
    print("="*40)

except Exception as e:
    print(f"Error: {e}")