import streamlit as st
st.set_page_config(initial_sidebar_state ='expanded',layout = 'wide')
from eda import data_analyse
from ml_app import *
from ml import *
from anomaly_app import *
import datetime
from datetime import timedelta
#Main Fxn 
def main():
	col1, col2, col3 = st.beta_columns([1.35,1,1])

	with col1:
		st.write("")

	with col2:
		st.title("EnergyNET")

	with col3:
		st.write("")

	menu = ["Home","Exploratory Data Analysis", "Machine learning app","Anomaly Detection","About"]
	choice = st.sidebar.selectbox("Menu", menu)

	if  choice == "Home":
		st.markdown('<b><h1>About the App</b>', unsafe_allow_html=True)
		st.write("""
			
			EnergyNET is build to monitor and curb the problem of energy wastage at the consumer level, an oft-neglected segment of our lives. It might be an appliance in our homes switched on but kept unused, a dafty door leading to our A.C consuming twice as much energy, or Flimsy insulation. 
			These problems collectively amount to huge energy losses in our Households. An energy audit is thus required to assess where a consumer can improve to make the house more efficient. Enters EnergyNET. 

			EnergyNET uses the dedicated architecture of Long-Short Term Memory (LSTM) which is a special type of Recurrent Neural Network(RNN).It carefully analyzes the energy used by individual appliances contributing to the household as well as predicts the energy usage in the future so that the person can take appropriate steps to achieve his desired energy consumption goal.


			### Machine Learning Based Predictive Analysis  ###
 
			Machine learning based predictive analysis is an effective tool to predict future outcomes based on historical and current data.
			Identifying the various statistical trends and data modelling techniques helps to make informed decision.
			The use of machine learning will enable us to accurately predict the energy consumption 
			of a small household to a large building .
			Consumers will be aware of potential energy usage with the aid of ML and our project, 
			and the risk of future overloading can be reduced.
			The use of effective data analysis using Machine learning will this enable the consumer to 
			keep tab of his daily consumption and save energy.



			
			""")

	if choice == "Exploratory Data Analysis":
		st.subheader("Exploratory Data Analysis")
		data_analyse()
	if choice =="Machine learning app":
		st.subheader("Machine Learning App")
		st.write("""
			This is an introduction to machine learning app
			""")
		filtered_date = st.date_input("Enter Date", 
                         value = datetime.date(2010, 11, 2),
                         min_value = datetime.date(2010, 10, 28),
                         max_value = datetime.date(2010, 11, 26)
                        )
		col1 , col2  = st.beta_columns([1,1])
		with col1:
			st.subheader("Real Values")
			dataset1 = actual_data(filtered_date)
			st.dataframe(dataset1)
		with col2:
			st.subheader("Predicted Values")
			dataset2 = search_dates(filtered_date)
			st.dataframe(dataset2)
		data_predict(filtered_date)
		
	if choice == "Anomaly Detection":
		st.subheader("Anomaly Detection")
		anomaly_app()

	if choice == "About":
		st.subheader("About")











if __name__ == '__main__':
	main()	