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
	
	choice = st.selectbox('Select the target variable',("Units","App1","App3","App4"))   
	if choice == "Units":
		choice = "Units"
	if choice == "App1":
		choice = "App1"
	if choice == "App2":
		choice = "App2"
	if choice == "App3":
		choice = "App3"
	st.info("The use of a single variable at a time is encouraged to avoid cramming up of graphs")
	test_req = test[choice]
	anomalies_req = anomalies[choice]	
	test_1_inv = pd.DataFrame(scaler.inverse_transform(test_req[time_steps:]))	
	anomalies_1_inv = pd.DataFrame(scaler.inverse_transform(anomalies_req))
	
	#------------------Interactive graph----------------------------- 
	fig = px.line(x = test[time_steps:].index,y = test_1_inv.squeeze(axis=1),width=1400, height=600);
	fig.add_scatter(x= anomalies.index, y = anomalies_1_inv.squeeze(axis=1),connectgaps=False, name="Anomalies")	
	st.write(fig)

	
	col1 ,col2 = st.beta_columns([1,1]) 
	if choice == "Units":
		with col1:
			with st.beta_expander("Anomaly on Test Data"):
				st.dataframe(test_df)
			with st.beta_expander("Test_error"):
				img1 = Image.open("mae_test.png")
				st.image(img1)
		with col2:
			with st.beta_expander("Anomalies"):
				st.dataframe(anomalies)
			with st.beta_expander("Train Loss"):
				img2 = Image.open("mae_train.png")
				st.image(img2)			
		with st.beta_expander("Anomaly Threshold"):
			img3 = Image.open("threshold_graph.png")
			st.image(img3)	
	if choice == "App1":
		# with col1:
		# 	with st.beta_expander("Anomaly on Test Data"):
		# 		st.dataframe(test_df)
		# 	with st.beta_expander("Test_error"):
		# 		img1 = Image.open("mae_test.png")
		# 		st.image(img1)
		# with col2:
		# 	with st.beta_expander("Anomalies"):
		# 		st.dataframe(anomalies)
		# 	with st.beta_expander("Train Loss"):
		# 		img2 = Image.open("mae_train.png")
		# 		st.image(img2)			
		# with st.beta_expander("Anomaly Threshold"):
		# 	img3 = Image.open("threshold_graph.png")
		# 	st.image(img3)
		pass

	if choice == "App2":
		# with col1:
		# 	with st.beta_expander("Anomaly on Test Data"):
		# 		st.dataframe(test_df)
		# 	with st.beta_expander("Test_error"):
		# 		img1 = Image.open("mae_test.png")
		# 		st.image(img1)
		# with col2:
		# 	with st.beta_expander("Anomalies"):
		# 		st.dataframe(anomalies)
		# 	with st.beta_expander("Train Loss"):
		# 		img2 = Image.open("mae_train.png")
		# 		st.image(img2)			
		# with st.beta_expander("Anomaly Threshold"):
		# 	img3 = Image.open("threshold_graph.png")
		# 	st.image(img3)		
		pass	
	if choice == "App3":
		# with col1:
		# 	with st.beta_expander("Anomaly on Test Data"):
		# 		st.dataframe(test_df)
		# 	with st.beta_expander("Test_error"):
		# 		img1 = Image.open("mae_test.png")
		# 		st.image(img1)
		# with col2:
		# 	with st.beta_expander("Anomalies"):
		# 		st.dataframe(anomalies)
		# 	with st.beta_expander("Train Loss"):
		# 		img2 = Image.open("mae_train.png")
		# 		st.image(img2)			
		# with st.beta_expander("Anomaly Threshold"):
		# 	img3 = Image.open("threshold_graph.png")
		# 	st.image(img3)	
   		pass