import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Title and description
st.title("Predictive Model for Term Deposit Subscription")
st.markdown(
    """
    This app predicts whether a customer will subscribe to a term deposit based on input features from a bank marketing campaign.
    """
)

# Sidebar for user input
st.sidebar.header("User Input Features")
def user_input_features():
    age = st.sidebar.slider('Age', 18, 95, 30)
    job = st.sidebar.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                                       'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
    marital = st.sidebar.selectbox('Marital Status', ['single', 'married', 'divorced', 'unknown'])
    education = st.sidebar.selectbox('Education', ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
                                                    'professional.course', 'university.degree', 'unknown'])
    default = st.sidebar.selectbox('Has credit in default?', ['no', 'yes', 'unknown'])
    housing = st.sidebar.selectbox('Has housing loan?', ['no', 'yes', 'unknown'])
    loan = st.sidebar.selectbox('Has personal loan?', ['no', 'yes', 'unknown'])
    contact = st.sidebar.selectbox('Contact Communication Type', ['cellular', 'telephone'])
    month = st.sidebar.selectbox('Last Contact Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    day_of_week = st.sidebar.selectbox('Day of Week', ['mon', 'tue', 'wed', 'thu', 'fri'])
    duration = st.sidebar.slider('Last Contact Duration (seconds)', 0, 5000, 180)
    campaign = st.sidebar.slider('Number of Contacts During Campaign', 1, 50, 2)
    pdays = st.sidebar.slider('Days Since Last Contact', 0, 999, 999)
    previous = st.sidebar.slider('Number of Contacts Before Campaign', 0, 50, 0)
    poutcome = st.sidebar.selectbox('Previous Outcome', ['failure', 'nonexistent', 'success'])
    emp_var_rate = st.sidebar.slider('Employment Variation Rate', -3.5, 1.5, 0.0)
    cons_price_idx = st.sidebar.slider('Consumer Price Index', 92.0, 95.0, 93.0)
    cons_conf_idx = st.sidebar.slider('Consumer Confidence Index', -50.0, -20.0, -30.0)
    euribor3m = st.sidebar.slider('Euribor 3-Month Rate', 0.0, 5.0, 1.0)
    nr_employed = st.sidebar.slider('Number of Employees', 4900, 5200, 5000)
    
    data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Load the pre-trained model
model = load('model.pkl')

# Encoding categorical features
le = LabelEncoder()
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
for col in cat_cols:
    input_df[col] = le.fit_transform(input_df[col])

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Displaying results
st.subheader("Input features:")
st.write(input_df)

st.subheader("Prediction:")
st.write("Subscribed" if prediction[0] == 1 else "Not Subscribed")

st.subheader("Prediction Probability:")
st.write(prediction_proba)