import xarray as xr
import numpy as np
import pandas as pd
import datetime
from datetime import date, timedelta
import os
import time


stations = {
#     'BRA010': {'lat': 28.946262, 'lon': -95.255623},
#     'BRA011': {'lat': 28.956981, 'lon': -95.234259},
    # 'BRA012': {'lat': 28.956981, 'lon': -95.234259},
#     'CAM010': {'lat': 26.097997, 'lon': -97.157449},
#     'CAM011': {'lat': 26.097997, 'lon': -97.157449},
#     'CAM030': {'lat': 26.097997, 'lon': -97.157449},
#     'GAL036': {'lat': 29.271653, 'lon': -94.807565},
#     'GAL037': {'lat': 29.271653, 'lon': -94.807565},
#     'GAL038': {'lat': 29.271653, 'lon': -94.807565},
#     'JEF009': {'lat': 29.669334, 'lon': -94.043078},
#     'JEF012': {'lat': 29.669334, 'lon': -94.043078},
#     'JEF013': {'lat': 29.674012, 'lon': -94.019025},
    'NUE014': {'lat': 27.620335, 'lon': -97.188081},
    # 'NUE015': {'lat': 27.620335, 'lon': -97.188081},
    # 'NUE016': {'lat': 27.620335, 'lon': -97.188081},
}

def iterate_folders():
    current_dir = os.getcwd()+'/data_csv'

    for year in range(2009, datetime.datetime.now().year + 1):
        year_folder = os.path.join(current_dir, str(year))
        if not os.path.exists(year_folder):
            os.makedirs(year_folder)
        print(f"Year: {year}")
        for month in range(1, 13):
            month_folder = os.path.join(year_folder, f"{month:02d}")
            if not os.path.exists(month_folder):
                os.makedirs(month_folder)
            for station_code, station_data in stations.items():
                lat_target = station_data['lat']
                lon_target = station_data['lon']
                print(f"Station Code: {station_code}")
                print(f"Latitude: {lat_target}")
                print(f"Longitude: {lon_target}")
                print("--------------------")


                # lat_target = 28.955003
                # lon_target = -95.265708
                # url = 'https://gcoos5.geos.tamu.edu/thredds/dodsC/NcML/nowcast_agg.nc'
                # url = 'https://hafen.geos.tamu.edu/thredds/dodsC/NcML/forecast_his_archive_agg.nc'
                if year < datetime.datetime.now().year:
                    url = 'https://hafen.geos.tamu.edu/thredds/dodsC/NcML/txla_hindcast_agg'
                else:
                    url = 'https://hafen.geos.tamu.edu/thredds/dodsC/NcML/forecast_his_archive_agg.nc'
                
#                         url = 'https://hafen.geos.tamu.edu/thredds/dodsC/NcML/txla_nest_p_his_2021_v2_agg'
                start_date = datetime.date(year, month, 1)
                
                if month == 12:
                    end_date = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
                else:
                    end_date = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)

                end_date_str = end_date.strftime('%Y-%m-%d')
                
                chunks = {'ocean_time': 1}

                ds = xr.open_dataset(url, chunks=chunks)

                # Find closest latitude and longitude coordinates
                lat_distance = np.abs(ds.lat_rho.values - lat_target)
                lon_distance = np.abs(ds.lon_rho.values - lon_target)

                # Flatten the distances to a 1D array
                lat_distance = np.ravel(lat_distance)
                lon_distance = np.ravel(lon_distance)

                # Find the index of minimum distance
                index = np.argmin(np.sqrt(lat_distance**2 + lon_distance**2))

                # Get eta_rho and xi_rho values for closest coordinate
                eta_rho = np.unravel_index(index, ds.lat_rho.shape)
                eta = ds.lat_rho.values[eta_rho]
                xi = ds.lon_rho.values[eta_rho]

                print("Closest Coordinates:")
                print("eta_rho:", eta)
                print("xi_rho:", xi)

                # Convert the flattened index to the original shape indices
                eta_rho_indices = np.unravel_index(index, ds.lat_rho.shape)
                eta_index, xi_index = eta_rho_indices

                print("Closest Coordinate indexes:")
                print("eta_rho index:", eta_index)
                print("xi_rho index:", xi_index)
                
                print(end_date_str)
                # Adjust the end_date to the next day and set time to midnight
                end_date = pd.to_datetime(end_date_str) 
                end_date = end_date.replace(hour=0, minute=0, second=0)

                # Select subset of data within the desired time range
                subset_ds = ds.sel(ocean_time=slice(start_date, end_date))

                # Sort the subset dataset by 'ocean_time'
                sorted_ds = subset_ds.sortby('ocean_time')                
                

                
                ts2 = time.time()
                salt = sorted_ds.salt.sel(
                    ocean_time=slice(start_date.strftime('%Y-%m-%d'), end_date_str),
                    s_rho=sorted_ds.s_rho[0],
                    eta_rho=sorted_ds.eta_rho[eta_index],
                    xi_rho=sorted_ds.xi_rho[xi_index]
                )
                te2 = time.time()
                
                time2 = te2 - ts2
                print("time for sorting2:", time2)
                
                ts = time.time()


                # Convert x to a pandas DataFrame
                # Convert x to a pandas DataFrame with reduced memory usage
                df = salt.to_dataframe(name='Salinity').astype({'Salinity': 'float32'})
                df.reset_index(inplace=True)

                
                te = time.time()
                
                time1 = te - ts
                print("time for sorting:", time1)
                # Convert the column to string type
                df['ocean_time'] = df['ocean_time'].astype(str)

                # Split "ocean_time" column into "Date" and "Time"
                df[['Date', 'Time']] = df['ocean_time'].str.split(pat=' ', n=1, expand=True)

                df.drop('ocean_time', axis=1, inplace=True)
                df.drop('s_rho', axis=1, inplace=True)
                df.drop('lon_rho', axis=1, inplace=True)
                df.drop('lat_rho', axis=1, inplace=True)

                # Convert 'Date' column to datetime type
                df['Date'] = pd.to_datetime(df['Date'])

                # Extract the date from the datetime column
                df['Date'] = df['Date'].dt.date

                # Add 'T00:00:00Z' to the end of the 'Date' column values
                df['Date'] = df['Date'].astype(str) + 'T00:00:00Z'

                # Group by 'Date' and calculate the average 'Salinity' for each day
                daily_avg_df = df.groupby('Date')['Salinity'].mean().reset_index()

                #Save the DataFrame as a CSV file
                daily_avg_df.to_csv('data_csv/'+str(year)+'/'+str(month).zfill(2)+'/'+station_code+'_ocean_time_salinity_'+str(year)+'_'+str(month).zfill(2)+'.csv', columns=['Date', 'Salinity'], index=False)
            #     df.to_csv('/home/sjilla/ep_website/data/csv/'+station_code+'_ocean_time_salinity_30_day.csv', columns=['Date', 'Time', 'Salinity'], index=False)
                print(daily_avg_df)
                
                       

iterate_folders()