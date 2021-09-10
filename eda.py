import streamlit as st
import matplotlib.pyplot as plt
from graph2 import *
#Importing EDA librabries
import seaborn as sns
import pandas as pd 
import numpy as np 
import plotly.express as px
from PIL import Image
#Fxn
def data_analyse():
	st.subheader("Description")
	st.write("""
		Exploratory Data Analysis refers to 
		""")
	df = pd.read_csv("data/paris_dataset_plotting.csv",index_col = 0)
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
		area_chart = px.area(sampled_df,facet_col_wrap=2,width=1400, height=600)
		st.write(area_chart)
	
	with st.beta_expander("Heatmap"):

		fig, ax = plt.subplots()
		sns.heatmap(df.corr(method = 'spearman'), ax=ax,cmap = 'YlGnBu',annot = True, fmt='.2f')
		st.write(fig)
	with st.beta_expander("Outlier Detection Plot"):
		st.write("Outlier Detection Plot for Units")
		fig = px.box(df ,x = "Year", y = "Units",width=1300, height=600,color_discrete_sequence=px.colors.qualitative.Safe)
		st.write(fig)
	
	key1,key2 = st.beta_columns([1,1])
	choice = st.selectbox('Select the Appliance for Exploratory Data Analysis',("Units","App1","App2","App3"))

	if choice == "Units":
		row1,row2,row3 = st.beta_columns([1.35,1,1])
		with row2:
			st.write(" ## Seasonal Plots for Units")
		st.write("""
				 
				 """)
		row1,row2= st.beta_columns([1.35,2])
		with row2:
			img = Image.open("Images/Units_seasonal.png")
			st.image(img)
		with row1:
			st.write("""

			Seasonal Plots exhibit the seasonality of our dataset i.e. inclusive of numerous factors how do our 
			target variables;Units, App1, App2, App3; experience predictable changes that would recur 
			every year. A quick look at the above plot confirms that our Global Units or the total power consumption 
			values would have hit high demand during January Season (Winter's) while the power consumption is at a 
			minimum during July(Spring) when the weather is pleasant. For an economically viable model, this is a salient 
			factor of consideration as it classifies the timeframe during which energy consumption of household spikes and dips.
			
			The residual plot represents a simple linear regression being run on our model where the darkened 
			line represents the model fitting for the linear regression. The plot gives us an insight into how 
			complicated our model should be to train our data on. As evident from the plot our model performs quite 
			poorly on a simple linear regression thus need for more complicated models arise.
			
			""")
		
		row1,row2,row3 = st.beta_columns([1.35,2,1])
		with row2:
			st.write("## Cumulative Hourly Energy Consumption")
			st.write(""" 

					""")

		row1,row2= st.beta_columns([2,1.35])
		with row1:
			img1 = Image.open("Images/Units_hourly.png")
			st.image(img1)
		with row2:
			st.write("""
			
			Cumulative Hourly Energy Consumption refers to the hourly energy consumption of the user for the whole year.
			 This parameter is practical in determining the time of the day where the household consumption is highest and 
			 at its subsequent lowest. The plot exhibits that the power consumption from night to early morning though 
			 significant is almost negligible compared to the rest of the day. This gives us valuable insight into the daily 
			 routine of the user and provides invaluable knowledge for the proper regulation of electrical energy in the household. 
			 Power consumption of the user increases drastically in the evening while showing a steady decline during the afternoons. 

			 """)
		
		row1,row2,row3 = st.beta_columns([1.35,2,1])
		with row2:
			st.write("## Cumulative Weekly Energy Consumption")
			st.write("""
				 
				 """)
		row1,row2= st.beta_columns([1.35,2])
		with row1:
			st.write("""
			Much like the Cumulative Hourly Energy Consumption represent the routine of the household
			power consumption daily, the weekly energy consumption constitutes the consumption of power weekly. 
			Weekly trends are an important metric in the evaluation as they represent the days where the power consumption was
			at its peak and when it was at its lowest.  This can help the user backtrack to the desired day to conclude why the
			consumption was high and try to avoid or at minimum limit the energy consumption if they ever face a similar scenario 
			in the future.
			""")
		with row2:
			img2 = Image.open("Images/Units_weekly.png")
			st.image(img2)
		
	if choice == "App1":
		img5 = Image.open("Images/App-1_seasonal.png")
		st.image(img5)
		st.write("This is a short Description of above image")
		img3 = Image.open("Images/App-1_hourly.png")
		st.image(img3)
		st.write("This is a short Description of above image")
		img4 = Image.open("Images/App_weekly.png")
		st.image(img4)
		st.write("This is a short Description of above image")			
	if choice == "App2":
		img8 = Image.open("Images/App-2_seasonal.png")
		st.image(img8)
		st.write("This is a short Description of above image")
		img6 = Image.open("Images/App-2_hourly.png")
		st.image(img6)
		st.write("This is a short Description of above image")
		img7 = Image.open("Images/App_weekly.png")
		st.image(img7)
		st.write("This is a short Description of above image")	
	if choice == "App3":
		img11 = Image.open("Images/App-3_seasonal.png")
		st.image(img11)
		st.write("This is a short Description of above image")
		img9 = Image.open("Images/App-3_hourly.png")
		st.image(img9)
		st.write("This is a short Description of above image") 
		img10 = Image.open("Images/App_weekly.png")
		st.image(img10)
		st.write("This is a short Description of above image")
		#start from image 12