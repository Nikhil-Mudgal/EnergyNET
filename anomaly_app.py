#Importing main Pkgs
import streamlit as st
from PIL import Image
#Importing EDA Pkgs
import pandas as pd
import plotly.express as px
from pickle import load

#Fxns
def anomaly_app():
	#-------------------Importing required datasets-----------------
	test_df = pd.read_csv("test_score_df.csv",index_col = 0)
	anomalies = pd.read_csv("anomaly_dataset.csv",index_col = 0)
	test = pd.read_csv("test_dataset_anomaly.csv",index_col = 0)
	scaler= load(open('scaler.pkl', 'rb'))
	time_steps = 24

	column1, column2,column3,column4 = st.beta_columns([1,1,1,1])
	choice = []
	with column1:
		choice_1 = st.checkbox('Units')     
	with column2:
		choice_2 = st.checkbox('App1')
	with column3:	
		choice_3 = st.checkbox('App2')
	with column4:
		choice_4 = st.checkbox('App3') 
	if choice_1:
		choice.append("Units")
	if choice_2:
		choice.append("App1")
	if choice_3:
		choice.append("App2")
	if choice_4:
		choice.append("App3")
	st.info("The use of a single variable at a time is encouraged to avoid cramming up of graphs")
	test_req = test[choice]
	anomalies_req = anomalies[choice]	
	test_1_inv = pd.DataFrame(scaler.inverse_transform(test_req[time_steps:]),columns=choice)	
	anomalies_1_inv = pd.DataFrame(scaler.inverse_transform(anomalies_req),columns=choice)
	
	#------------------Interactive graph----------------------------- 
	fig = px.line(x = test[time_steps:].index,y = test_1_inv[choice].squeeze(axis=1),width=1400, height=600);
	fig.add_scatter(x= anomalies.index, y = anomalies_1_inv.loc[:,choice].squeeze(axis=1),connectgaps=False, name="Anomalies")	
	st.write(fig)

	

	col1 ,col2 = st.beta_columns([1,1]) 
	with col1:
		with st.beta_expander("Anomaly_threshold_1"):
			img1 = Image.open("mae_test.png")
			st.image(img1)
		with st.beta_expander("Anomaly_list_1"):
			st.dataframe(test_df)
	with col2:
		with st.beta_expander("Anomaly_threshold_2"):
			img2 = Image.open("mae_train.png")
			st.image(img2)		
		with st.beta_expander("Anomaly_list_2"):
			st.dataframe(anomalies)
			
			

   