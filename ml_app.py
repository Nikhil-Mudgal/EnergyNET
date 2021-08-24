import matplotlib.pyplot as plt 
from ml import *
import streamlit as st
#import EDA Pkgs
import seaborn as sns
import pandas as pd 
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go
#Fxns 
def data_predict(filtered_date):
	col1 , col2 = st.beta_columns([1,1])

	with col1:
		with st.beta_expander("Accuracy Score of the Model"):
			score = evaluate_model(filtered_date)
			st.dataframe(score)
			pass
	with col2:
		with st.beta_expander("Loss vs Val loss and Hyperparameters"):
			pass
	st.write("Choose your Target entities")
	column1, column2,column3, column4 = st.beta_columns([1,1,1,1])
	choice = []
	with column1:
		choice_1 = st.checkbox('Units')     
	with column2:
		choice_2 = st.checkbox('App1')
	with column3:	
		choice_3 = st.checkbox('App2')
	with column4:
		choice_4 = st.checkbox('App3') 
	filtered_date = filtered_date
	
	predicted_values = search_dates(filtered_date)

	actual_values = actual_data(filtered_date)

	values = ["real","pred"]

	for i in range(len(values)):
		if choice_1:
			choice.append('Units_' + values[i])
		if choice_2:
			choice.append('App-1_' + values[i])
		if choice_3:		
			choice.append('App-2_' + values[i])
		if choice_4:
			choice.append('App-3_' + values[i])
	
	total_df = total_dataframe(filtered_date)
	graph1 = px.line(total_df[choice],width=1200, height=600)
	st.write(graph1)
	# col3,col4 = st.beta_columns([1,1])
	# with col3:
	# 	st.write("Actual Values")
	# 	graph1 = px.line(actual_values[choice])
	# 	st.write(graph1)
	# with col4:
	# 	st.write("Predicted values")	
	# 	graph2 = px.line(predicted_values[choice])
	# 	st.write(graph2)
	

	
	
	
	