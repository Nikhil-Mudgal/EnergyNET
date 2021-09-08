# Building Energy Management System (BeMS) using Deep learning
Welcome to the Building Energy Management repository here you will find the tools to create an app to predict the energy consumption of your household.

## Development Environment
The project was built on spyder v4.1.5. A detailed description of the environment setup while making the project has been saved in the requirements.txt file that can be accessed using the following command on the desired virtualenv:

```pip install -r requirements.txt```

Note:- Remember to set up and activate the virtual env before running the code.

## Data Acquistion
### ML Dataset
The data regarding energy consumption was acquired from the UCI machine learning repository which can be accessed through this [link](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption).

Note:- Fully engineered and raw datasets are already attached to the repository for convenience. 

### Weather Dataset
Weather dataset for Sceaux Paris for the desired years was obtained from the NSRDB data viewer although now the historical data is currently unavailable thus the data has been attached to the repository as well. The viewer is still a very useful asset to download historical weather data as well as solar irradiation data and thus the link has been shared:

[NSRDB Viewer](https://maps.nrel.gov/nsrdb-viewer/?aL=x8CI3i%255Bv%255D%3Dt%26Jea8x6%255Bv%255D%3Dt%26Jea8x6%255Bd%255D%3D1%26VRLt_G%255Bv%255D%3Dt%26VRLt_G%255Bd%255D%3D2%26mcQtmw%255Bv%255D%3Dt%26mcQtmw%255Bd%255D%3D3&bL=clight&cE=0&lR=0&mC=48.77743198758074%2C2.3000693321228027&zL=15)

The two datasets were merged according to their dates carefully to identify whether the weather data had an effect on power consumption and indeed it did. Thus solidifying our stance on the inclusion of the dataset. This furthers our methodology of including weather data for the evaluation of power consumption in a particular household.

# ML APP
## Data Preprocessing
- Handling Missing Data / NaN's
- Sorting the data
- Resampling the data Hourly
- Concatenating the weather and Energy dataset
- Feature Engineering
- Converting the scalers into pkl files

## Splitting the dataset
- Training dataset
- Test dataset

## Preparation for LSTM framework
- Creation of a look back function 
- Creating Tensor with desired layers

## LSTM Framework
Long Short Term Memory (LSTM) architecture using the sliding window algorithm was used to train the prepared dataset and the weights were exported on an h5 file to use with the test dataset. Since the Deep learning frameworks take a lot of time to run (the LSTM model ran for 1hr 46 min for 100 iterations) to evaluate on a test dataset it is best to import the datasets the weights and pkl from our pre-trained model.

## Streamlit App
Streamlit is an app framework designed for ML engineers and data scientists. It works on python and is compatible with many other data science libraries for an easier workflow.
Streamlit can be installed in the system using the following command:

```pip install streamlit```

Streamlit provides widgets to display dataframes, calendar, graphs thus eschewing the use python web designing frame such as flask to create backend for these elements and the ML engineer can thus focus his efforts in creating the linkage of the machine learning model with the interface. 

To run the app on your system,download the repository setup the environment and requirements and use the following code:

```streamlit run app.py``` in your terminal.

We are working day and night to deploy our app on the streamlit platform but it will take some time as we have to test for bugs and crashes in the app. However it will work fine in your system until the requirements and environment has been setup correctly.



![Sneek Peak of the EnergyNET app](https://github.com/Nikhil-Mudgal/EnergyNET/blob/main/Images/Home_page.jpg?raw=true)

![Sneek Peak of the EnergyNET app](https://github.com/Nikhil-Mudgal/EnergyNET/blob/main/Images/EDA.jpg?raw=true)

![Sneek Peak of the EnergyNET app](https://github.com/Nikhil-Mudgal/EnergyNET/blob/main/Images/ML.jpg?raw=true)












