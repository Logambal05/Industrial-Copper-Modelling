import streamlit as st
import pandas as pd 
import numpy as np 
import pickle
import os
from streamlit_option_menu import option_menu

# [theme]
primaryColor="#6c5e71"
backgroundColor="#d9d3dc"
secondaryBackgroundColor="#9988a0"
textColor="#0a0a0a"
font="serif"


st.set_page_config(page_title = 'Industrial Copper Modeling')

with st.sidebar:
        SELECT = option_menu(None,
                options = ["🏡Home","🎢Data Prediction","🔚Exit"],
                default_index=0,
                orientation="vertical",
                styles={"container": {"width": "90%"},
                        "icon": {"color": "white", "font-size": "18px"},
                        "nav-link": {"font-size": "18px"}})

if SELECT == '🏡Home':
    st.header("**_Industrial Copper Modeling_**")
    st.subheader("**Introduction**")
    st.write("""Like many other industries, the copper sector struggles to deal with less complicated but distorted and noisy sales and pricing data. 
            Manual forecasting takes time and might not be precise. Making use of machine learning techniques can greatly enhance decision-making. 
            We will deal with problems like skewness and noisy data in this solution, and create a regression model to forecast selling
            rates and a classification algorithm to forecast the lead status (WON or LOST).""")
    st.subheader("**Key Objectives:**")
    st.markdown("""
        1. **Data Exploration:**
            - Identify and address skewness and outliers in the sales dataset.
        2. **Data Preprocessing:**
            - Transform data and implement strategies to handle missing values effectively.
        3. **Regression Model:**
            - Develop a robust regression model to predict '**Selling_Price.**'
            - Utilize advanced techniques such as data normalization and feature scaling.
        4. **Classification Model:**
            - Build a classification model to predict lead status (WON/LOST).
            - Leverage the '**STATUS**' variable, considering WON as Success and LOST as Failure.""")
    st.subheader("**Tools Used:**")
    st.markdown("""
        - **_Python:_** Facilitates versatile programming capabilities.
        - **_Pandas and NumPy:_** These libraries will be used for data manipulation and preprocessing.
        - **_Scikit-Learn:_** A powerful machine learning library that includes tools for regression and classification models.
        - **_Streamlit:_** A user-friendly library for creating web applications with minimal code, perfect for building an interactive interface for our models.""")
    

status_options = [None,'Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
status_values = {'Won': 0, 'Draft': 1, 'To be approved': 2, 'Lost': 3, 'Not lost for AM': 4,'Wonderful': 5, 'Revised': 6, 'Offered': 7, 'Offerable': 8}

item_type_options = [None,'W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
item_type_values =  {'W' : 0, 'WI' : 1, 'S' : 2, 'Others': 3, 'PL': 4, 'IPL': 5 , 'SLAWR': 6}

product_ref_options = [None, 611728, 611733, 611993, 628112, 628117, 628377, 640400, 640405, 640665, 164141591, 164336407,
                    164337175, 929423819, 1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 1665584642,
                    1665584662, 1668701376, 1668701698, 1668701718, 1668701725, 1670798778, 1671863738, 1671876026,
                    1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]

application_options = [None, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 27.0, 28.0, 29.0, 38.0, 
                    39.0, 40.0, 41.0, 42.0, 56.0, 58.0, 59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]

country_options = [None, 25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 77.0, 78.0, 79.0, 80.0, 84.0, 89.0, 107.0, 113.0]

if SELECT == "🎢Data Prediction":
    tab1, tab2 = st.tabs(['**_Prediction Of Selling Price_**', '**_Prediction Of Lead Status_**'])
    with tab1:
        st.subheader("**_Advanced Price Prediction Model_**")
        with st.form('Regression'):
            col1, col2 = st.columns(2)
            with col1:
                application = st.selectbox("**Select Application Number**", options = application_options)
                country = st.selectbox("**Select Country Code**", options = country_options)
                item_type = st.selectbox("**Select Item Type**", options = item_type_options)
                status = st.selectbox("**Select Status**", options = status_options)
                product_reference = st.selectbox("**Select Product Reference**", options = product_ref_options)

            with col2:
                thickness = st.number_input(label = "**Enter Thickness value**", help='Minimum_Value = 0.1 & Maximum_Value = 10.00')
                width = st.number_input("**Enter Width**", help= 'Minimum_Value= 700 & Maximum_Value= 2000')
                quantity_tons = st.number_input("**Enter Quantity Tons**", help='Minimum_value = 0.00001 & Maximum_value = 150.00')
                customer = st.number_input("**Enter Customer Number**", help= 'Minimum_Value= 12450 & Maximum_Value= 30400000')
            col3,col4 = st.columns([1,2])
            with col4:              
                SellingPrice_Result = st.form_submit_button("**_Click Me To Known Selling Price_**")
            if SellingPrice_Result:
                # Load the regression model
                with open("C:/Users/Logambal/Desktop/PROJECT-GUVI/RegressorModel_Selling_Price_Predt.pkl", 'rb') as file:
                    price = pickle.load(file)
                price_features = np.array([[float(quantity_tons), float(customer), float(country), status_values[status], item_type_values[item_type], float(application), float(thickness), float(width), float(product_reference)]])
                selling_price_value = price.predict(price_features)
                sp_value = np.exp(selling_price_value[0])
                st.markdown(f"<p style='font-size:22px; text-align: center;'><em>The Predicted Selling Price Is:</em> <strong>{sp_value}</strong></p>", unsafe_allow_html=True)
        
    with tab2:
        st.subheader("**_Advanced Stauts Prediction Model_**")
        with st.form('Classification'):
            col5,col6 = st.columns(2)
            with col5:
                application_cls = st.selectbox("**Select the Application Number**", options = application_options, key='option1')
                country_cls = st.selectbox("**Select Country Number**", options = country_options, key='option2')
                item_type_cls = st.selectbox("**Select the Item Type**", options = item_type_options, key='option3')
                product_reference_cls = st.selectbox("**Select Product Reference**", options = product_ref_options, key='option4')
            with col6:
                quantity_tons_cls = st.number_input("**Enter Quantity Tons**", help='Minimum_Value = 0.00001 & Maximum_Value = 150.00', key='option5')
                customer_cls = st.number_input("**Enter Customer Number**", help= 'Minimum_Value= 12450 & Maximum_Value= 30400000', key='option6')
                thickness_cls = st.number_input("**Enter Thickness value**", help='Minimum_Value = 0.1 & Maximum_Value = 10.00', key='option7') 
                width_cls = st.number_input("**Enter Width**", help= 'Minimum_Value= 700 & Maximum_Value= 2000',key ='option8')
                selling_price_cls = st.number_input("**Enter Selling Price Amount**",help = 'Price In $' , key='option9')
                log_selling_price = np.log(float(selling_price_cls))
            col7,col8 = st.columns([1,2])
            with col8:    
                staus_result = st.form_submit_button("**Click Me To Known the Status Lead**")

            if staus_result:
                with open("C:/Users/Logambal/Desktop/PROJECT-GUVI/ClassifierModel_Stauts_Predt.pkl", 'rb') as file:
                    reg_cls= pickle.load(file)

                status_features = np.array([[float(quantity_tons_cls), float(customer_cls), float(country_cls), item_type_values[item_type_cls], float(application_cls), float(thickness_cls), float(product_reference_cls), float(width_cls),float(selling_price_cls), log_selling_price]])
                status = reg_cls.predict(status_features)

                if status[0] == 0:
                    status_val = 'Won' 
                else: 
                    status_val = 'Lost'
                
                st.markdown(f"<p style='font-size:22px; text-align: center;'><em>The Predicted Status Leading Is:</em> <strong>{status_val}</strong></p>", unsafe_allow_html=True)

if SELECT == "🔚Exit":
    st.subheader("**_OverView_**")
    st.markdown("""By incorporating machine learning in data exploration, preprocessing, regression, and classification, this solution provides a comprehensive approach for the copper industry to improve pricing decisions and lead status assessments. 
                The Streamlit web application is a useful tool that guarantees decision-makers' accessibility and usability, with a focus on the special tasks of **_Selling Price_** and **_Stauts Lead_** prediction.""")
    button = st.button("EXIT!")
    if button:
        st.success("**Thank you for utilizing this platform. I hope you have received the predicted price and status for your copper industry!❤️**")