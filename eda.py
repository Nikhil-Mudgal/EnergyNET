import streamlit as st
import matplotlib.pyplot as plt
from graph2 import *
#Importing EDA librabries
import seaborn as sns
import pandas as pd 
import numpy as np 
import plotly.express as px
#Fxn
def data_analyse():
	st.subheader("Description")
	st.write("""
		Exploratory Data Analysis refers to the shit
		""")
	df = pd.read_csv("paris_dataset_plotting.csv",index_col = 0)
	st.dataframe(df.head())
	#creating 2 columns to represent different variables
	
	
	
	
	st.write('Plot paramaters:')
	col1, col2, col3,col4 = st.beta_columns([1,1,1,1])
	with col1:
		option_1 = st.checkbox('Units')     
	with col2:
		option_2 = st.checkbox('App1')
	with col3:	
		option_3 = st.checkbox('App2')
	with col4:
		option_4 = st.checkbox('App3')     
	

	# plt_params = plt_params + option_2 + option_3 + option_4 
	
	variables =list(map(str,range(2007,2011)))
	year = st.multiselect("Select years to be plotted",variables,default = '2008')
	column1,column2 = st.beta_columns([1,1])
	with column1:
		sampling_type = st.radio("Select the sampling type for your data",("Monthly","Daily"))
	


	plt_params =[]
	#year = ['2007','2008']	
	for i in range(len(year)):
		if option_1:
			plt_params.append(year[i] + '_Units')
		if option_2:
			plt_params.append(year[i] + '_App1')
		if option_3:		
			plt_params.append(year[i] + '_App2')
		if option_4:
			plt_params.append(year[i] + '_App3')
	# st.write(str(plt_params))
	
	if sampling_type == "Monthly":
		sampling_type = '1M'
		sampled_df = required(plt_params,sampling_type)
		
	elif sampling_type =="Daily":
		sampling_type = '1D'			
		sampled_df = required(plt_params,sampling_type) 
	with column2:
		choice = st.radio("Select the chart type",('Line',"Area"))

	if choice == 'Line':
		line_graph = px.line(sampled_df,width=1400, height=600)
		st.write(line_graph)
	if choice == 'Area':
		area_chart = px.area(sampled_df,width=1400, height=600)
		st.write(area_chart)
	
	with st.beta_expander("Heatmap"):

		fig, ax = plt.subplots()
		sns.heatmap(df.corr(method = 'spearman'), ax=ax,cmap = 'YlGnBu',annot = True, fmt='.2f')
		st.write(fig)

	col_1,col_2 = st.beta_columns([1.5,1.5])
	with col_1:
	
		# with st.beta_expander("Daily Plot"):
		# 	variables =list(map(str,range(2007,2016)))
		# 	year = st.multiselect("Select years to be plotted",variables,default = '2015')
		# 	new_df = df2[lang_choices]
  #   		st.line_chart(new_df)
		with st.beta_expander("Dist Plot of Class"):
			pass

	with col_2:

		with st.beta_expander("Outlier Detection Plot"):
			pass

