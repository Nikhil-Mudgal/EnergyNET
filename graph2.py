import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("data/paris_dataset_plotting.csv",index_col= 0)
df.head

# year = ['2007_Units','2007_App1','2008_Units','2008_App1']
# #sampling_type = '1M'
# split_date ="2008"
daily_df = df[["DateTime","Units",'App-1','App-2','App-3']] 
daily_df['DateTime'] = pd.to_datetime(daily_df['DateTime'])
columns = []
#daily_df = daily_df.set_index("DateTime")

def sampling(sampling_type): 
    sampling_type = sampling_type
    if sampling_type == '1M':
        daily_resampled = daily_df.resample(sampling_type,on ="DateTime").sum()
        daily_resampled.reset_index(inplace = True)
    elif sampling_type == '1D':
        daily_resampled = daily_df.resample(sampling_type,on ="DateTime").sum()
        daily_resampled.reset_index(inplace = True)
        
    return daily_resampled


def required(year,sampling_type):
    if sampling_type == '1M':
        total_df = split_dataset_monthly()
        sampled = total_df[year]
    elif sampling_type == '1D':
        total_df = split_dataset_daily()
        sampled = total_df[year]
    return sampled
# sampling_type = '1D'
# a = required(year,sampling_type)

# def select_type(sampling_type):
#     daily_resampled = daily_df.resample(sampling_type,on ="DateTime").sum()
#     daily_resampled.reset_index(inplace = True)
#     return daily_resampled



def split_dataset_daily():
        
    total_df_daily = []
    total_df_daily = pd.DataFrame(total_df_daily)
    daily_resampled = sampling('1D')
    for split_date in range(2007,2011):
        if split_date == 2007:
            start_date = str(split_date)
            end_date = str(int(split_date) + 1)
            after_start_date = daily_resampled["DateTime"] >= start_date
            before_end_date  = daily_resampled["DateTime"] < end_date
            between_two_dates = after_start_date & before_end_date
            filtered_dates = daily_resampled.loc[between_two_dates]
            column = []
            nan_df = pd.DataFrame(np.nan, index=range(0,156), columns=["DateTime","Units",'App-1','App-2','App-3'], dtype='float')
            total_df_daily = pd.concat([nan_df,filtered_dates], axis = 0 )
            total_df_daily.reset_index(inplace=True, drop = True)
            total_df_daily.rename(columns={'DateTime': str('DateTime'+ "_"+ start_date),
                                'Units': str(start_date +'_Units'),
                                 'App-1': str(start_date +'_App1'),
                                 'App-2': str(start_date +'_App2'),
                                 'App-3': str(start_date +'_App3')},
                        inplace = True
                        )
                    
                
        elif split_date >= 2008:
            start_date = str(split_date)
            end_date = str(int(split_date) + 1)
            
            after_start_date = daily_resampled["DateTime"] >= start_date
            before_end_date  = daily_resampled["DateTime"] < end_date
            between_two_dates = after_start_date & before_end_date
            filtered_dates = daily_resampled.loc[between_two_dates]
            filtered_dates.reset_index(inplace=True, drop = True)
            total_df_daily = pd.concat([total_df_daily,filtered_dates], axis = 1)
            total_df_daily.rename(columns={'DateTime': str('DateTime_' + start_date),
                            'Units': str(start_date +'_Units'),
                                 'App-1': str(start_date +'_App1'),
                                 'App-2': str(start_date +'_App2'),
                                 'App-3': str(start_date +'_App3')},
                            inplace = True
                        )
    return total_df_daily

def split_dataset_monthly():
    total_df_monthly = []
    total_df_monthly = pd.DataFrame(total_df_monthly)
    daily_resampled = sampling('1M')
    for split_date in range(2007,2011):
        if split_date == 2007:
            start_date = str(split_date)
            end_date = str(int(split_date) + 1)
            after_start_date = daily_resampled["DateTime"] >= start_date
            before_end_date  = daily_resampled["DateTime"] < end_date
            between_two_dates = after_start_date & before_end_date
            filtered_dates = daily_resampled.loc[between_two_dates]
            column = []
            nan_df = pd.DataFrame(np.nan, index=range(0,5), columns=["DateTime","Units",'App-1','App-2','App-3'], dtype='float')
            total_df_monthly = pd.concat([nan_df,filtered_dates], axis = 0 )
            total_df_monthly.reset_index(inplace=True, drop = True)
            total_df_monthly.rename(columns={'DateTime': str('DateTime'+ "_"+ start_date),
                           'Units': str(start_date +'_Units'),
                                 'App-1': str(start_date +'_App1'),
                                 'App-2': str(start_date +'_App2'),
                                 'App-3': str(start_date +'_App3')},
                        inplace = True
                        )
                    
                
        elif split_date >= 2008:
            start_date = str(split_date)
            end_date = str(int(split_date) + 1)
            
            after_start_date = daily_resampled["DateTime"] >= start_date
            before_end_date  = daily_resampled["DateTime"] < end_date
            between_two_dates = after_start_date & before_end_date
            filtered_dates = daily_resampled.loc[between_two_dates]
            filtered_dates.reset_index(inplace=True, drop = True)
            total_df_monthly = pd.concat([total_df_monthly,filtered_dates], axis = 1)
            total_df_monthly.rename(columns={'DateTime': str('DateTime_' + start_date),
                                 'Units': str(start_date +'_Units'),
                                 'App-1': str(start_date +'_App1'),
                                 'App-2': str(start_date +'_App2'),
                                 'App-3': str(start_date +'_App3')},
                        inplace = True
                        )              
                    
    return total_df_monthly
    
   





# column = ["January","Feb","March","April","May","June","July","August","Sept","Oct","Nov","Dec"]
# x = pd.Series(column)
# req1 = pd.concat([x,a],axis =1)



# sampled_df = sampling(year) 
# st.line_chart(sampled_df)





# def make_graph():
#     sampled = sampling(year)
#     plt.plot(column,sampled)
#     plt.xlabel("Days")
#     plt.ylabel("Consumption")
#     plt.title(f"Power consumption of Household(in Paris,France) for the year {year}")
#     plt.legend(sampled,loc='upper right')




# columns = list(map(str,range(2008,2016)))
# list(columns)
# plt.plot(column,sampled)
# plt.xlabel("Days")
# plt.ylabel("Consumption")
# plt.title("Power consumption of Household(in Paris,France) for the year 2007-2016")
# plt.legend(sampled,loc='upper right')
# plt.show()
# a.info()



# start_date = str(split_date)
# end_date = str(int(split_date) + 1)
# after_start_date = daily_resampled["DateTime"] >= start_date
# before_end_date  = daily_resampled["DateTime"] < end_date
# between_two_dates = after_start_date & before_end_date
# filtered_dates = daily_resampled.loc[between_two_dates]
# #columns = daily_resampled.columns.values + "_" +str(split_date)

# nan_df = pd.DataFrame(np.nan, index=range(0,156), columns=['DateTime', 'Global_active_power'], dtype='float')
# total_df = pd.concat([nan_df,filtered_dates], axis = 0 )
# total_df.reset_index(inplace=True, drop = True)
# total_df.rename(columns={'DateTime': str('DateTime' + start_date),
#                          'Global_active_power': str(start_date)},
#                 inplace = True
                # )
# daily_resampled.columns.values
