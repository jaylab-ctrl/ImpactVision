import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import boxcox
from scipy.interpolate import griddata
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from prettytable import PrettyTable
import warnings

pd.set_option('display.float_format', lambda x: '%.2f' % x)
sns.set_style('darkgrid')
warnings.filterwarnings("ignore")

# Read in the data in chunks in for loop and then convert it to a dataframe and see the time it takes to read the data
start_time = time.time()
accidents_vehicles_casualties = pd.DataFrame()

for chunk in pd.read_csv('UK_Accidents_Merger.csv', chunksize=200000, low_memory=False):
    print('Number of chunks read: ', chunk.shape)
    accidents_vehicles_casualties = pd.concat([accidents_vehicles_casualties, chunk])

print("Time taken to read the data: ", time.time() - start_time)

print("Shape of the dataframe: ", accidents_vehicles_casualties.shape)

#---------------------------------------------
# Data Wrangling
#---------------------------------------------

# Check for missing values
print("Missing Values:")
print(accidents_vehicles_casualties.isnull().sum()/accidents_vehicles_casualties.shape[0])

# Drop or impute missing values
accidents_vehicles_casualties.dropna(inplace=True)

# Save the dataframe to a csv file
# accidents_vehicles_casualties.to_csv('UK_Accidents_Merger_Cleaned.csv', index=False)

# Check for duplicates
print("Duplicates:")
print(accidents_vehicles_casualties.duplicated().sum())

# Check for data types
print("Data Types:")
print(accidents_vehicles_casualties.dtypes)

# Count number of numeric and non-numeric columns
print("Number of Numeric Columns:")
print(len(accidents_vehicles_casualties.select_dtypes(include=np.number).columns))

print("Number of Non-Numeric Columns:")
print(len(accidents_vehicles_casualties.select_dtypes(exclude=np.number).columns))

#------------------------------------------
# Viusalizations using Matplotlib, Seaborn
#------------------------------------------

# Convert date to year
accidents_vehicles_casualties['Date'] = pd.to_datetime(accidents_vehicles_casualties['Date'])
accidents_vehicles_casualties['Year'] = accidents_vehicles_casualties['Date'].dt.year

# Manipulate the data
accidents_vehicles_casualties['Junction_Detail'] = accidents_vehicles_casualties['Junction_Detail'].replace({
    'T or staggered junction': 'Staggered Junction',
    'Private drive or entrance': 'Entrance',
    'Other junction': 'Others',
    'More then 4 arms (not roundabout)': 'Fours',
    'Mini-roundabout': 'Miniroundabout',
    'Not at junction or within 20 metres': 'No Junction',
    'Data missing or out of range': 'Unknown',
})

accidents_vehicles_casualties['Junction_Control'] = accidents_vehicles_casualties['Junction_Control'].replace({
    'Authorised person': 'Authorised',
    'Data missing or out of range': 'Unknown',
    'Give way or uncontrolled': 'Uncontrolled',
    'Stop sign': 'Stop',
    'Auto traffic signal': 'Traffic Signal',
})


accidents_vehicles_casualties['Light_Conditions'] = accidents_vehicles_casualties['Light_Conditions'].replace({
    'Darkness - no lighting': 'Darkness',
    'Darkness - lighting unknown': 'Unknown',
    'Darkness - lights lit': 'Light',
    'Darkness - lights unlit': 'Unlit'
})

accidents_vehicles_casualties['Weather_Conditions'] = accidents_vehicles_casualties['Weather_Conditions'].replace({
    'Fine without high winds': 'Fine',
    'Raining without high winds': 'Rain',
    'Raining + high winds': 'Rainwinds',
    'Fine + high winds': 'Fine',
    'Fog or mist': 'Fog',
    'Snowing without high winds': 'Snow',
    'Snowing + high winds': 'Snowind',
    'Data missing or out of range': 'Unknown',
})

accidents_vehicles_casualties['Road_Surface_Conditions'] = accidents_vehicles_casualties['Road_Surface_Conditions'].replace({
    'Dry': 'Dry',
    'Wet or damp': 'Wet',
    'Frost or ice': 'Ice',
    'Snow': 'Snow',
    'Flood over 3cm. deep': 'Flood',
    'Data missing or out of range': 'Unknown',
})

accidents_vehicles_casualties['Did_Police_Officer_Attend_Scene_of_Accident'] = accidents_vehicles_casualties['Did_Police_Officer_Attend_Scene_of_Accident'].replace(
    'No - accident was reported using a self completion  form (self rep only)', 'Self')

accidents_vehicles_casualties['Vehicle_Type'] = accidents_vehicles_casualties['Vehicle_Type'].replace({
    'Bus or coach (17 or more pass seats)': 'Bus',
    'Van / Goods 3.5 tonnes mgw or under': 'Van',
    'Taxi/Private hire car': 'Taxi',
    'Motorcycle 125cc and under': 'Motorcycle',
    'Motorcycle over 500cc': 'Motorcycle',
    'Goods 7.5 tonnes mgw and over': 'Goods',
    'Motorcycle 50cc and under': 'Motorcycle',
    'Motorcycle over 125cc and up to 500cc': 'Motorcycle',
    'Goods over 3.5t. and under 7.5t': 'Goods',
    'Other vehicle': 'Others',
    'Minibus (8 - 16 passenger seats)': 'Minibus',
    'Agricultural vehicle (includes diggers etc.)': 'Agricultural',
    'Motorcycle - unknown cc': 'Motorcycle'
})

accidents_vehicles_casualties['Towing_and_Articulation'] = accidents_vehicles_casualties['Towing_and_Articulation'].replace({
    'No tow/articulation': 'No',
    'Articulated vehicle': 'Articulated',
    'Single trailer': 'Single',
    'Other tow': 'Others',
    'Double or multiple trailer': 'Multiple',
    'Data missing or out of range': 'Unknown',
})

accidents_vehicles_casualties['Vehicle_Manoeuvre'] = accidents_vehicles_casualties['Vehicle_Manoeuvre'].replace({
    'Going ahead other': 'Going ahead',
    'Turning right': 'Right',
    'Waiting to go - held up': 'Waiting',
    'Slowing or stopping': 'Slowing',
    'Turning left': 'Left',
    'Moving off': 'Moving',
    'Waiting to turn right': 'Waiting',
    'Going ahead right-hand bend': 'Going ahead',
    'Going ahead left-hand bend': 'Going ahead',
    'Overtaking moving vehicle - offside': 'Overtaking',
    'Waiting to turn left': 'Waiting',
    'Overtaking static vehicle - offside': 'Overtaking',
    'Changing lane to left': 'Changing lane',
    'Changing lane to right': 'Changing lane',
    'U-turn': 'Uturn',
    'Overtaking - nearside': 'Overtaking',
    'Data missing or out of range': 'Unknown',
})

accidents_vehicles_casualties['Vehicle_Location-Restricted_Lane'] = accidents_vehicles_casualties['Vehicle_Location-Restricted_Lane'].replace({
    'On main c\way - not in restricted lane': 'Mainlane',
    'Bus lane': 'Buslane',
    'Footway (pavement)': 'Footway',
    'Leaving lay-by or hard shoulder': 'Leavinglayby',
    'On lay-by or hard shoulder': 'Onlayby',
    'Busway (including guided busway)': 'Busway',
    'Cycle lane (on main carriageway)': 'Cyclelane',
    'Tram/Light rail track': 'Tramtrack',
    'Entering lay-by or hard shoulder': 'Enteringlayby',
    'Cycleway or shared use footway (not part of  main carriageway)': 'Cycleway',
    'Data missing or out of range': 'Unknown',
})

accidents_vehicles_casualties['Junction_Location'] = accidents_vehicles_casualties['Junction_Location'].replace({
    'Approaching junction or waiting/parked at junction approach': 'Approaching',
    'Mid Junction - on roundabout or on main road': 'Mid',
    'Cleared junction or waiting/parked at junction exit': 'Cleared',
    'Entering from slip road': 'Entering',
    'Leaving main road into minor road': 'Leaving',
    'Entering main road from minor road': 'Entering',
    'Leaving roundabout': 'Leaving',
    'Entering roundabout': 'Entering',
    'Data missing or out of range': 'Unknown',
    'Not at or within 20 metres of junction': 'No Junction',
})

accidents_vehicles_casualties['1st_Point_of_Impact'] = accidents_vehicles_casualties['1st_Point_of_Impact'].replace({
    'Did not impact': 'Nothing',
    'Data missing or out of range': 'Unknown'
})

accidents_vehicles_casualties['Was_Vehicle_Left_Hand_Drive?'] = accidents_vehicles_casualties['Was_Vehicle_Left_Hand_Drive?'].replace({
    'Data missing or out of range': 'Unknown'
})

accidents_vehicles_casualties['Propulsion_Code'] = accidents_vehicles_casualties['Propulsion_Code'].replace({
    'Gas/Bi-fuel': 'Gas',
    'Petrol/Gas (LPG)': 'LPG',
})

accidents_vehicles_casualties['Casualty_Class'] = accidents_vehicles_casualties['Casualty_Class'].replace({
    'Driver or rider': 'Driver'
})

accidents_vehicles_casualties['Sex_of_Casualty'] = accidents_vehicles_casualties['Sex_of_Casualty'].replace({
    'Data missing or out of range': 'Unknown'
})

accidents_vehicles_casualties['Car_Passenger'] = accidents_vehicles_casualties['Car_Passenger'].replace({
    'Data missing or out of range': 'Unknown'
})

accidents_vehicles_casualties['Urban_or_Rural_Area'] = accidents_vehicles_casualties['Urban_or_Rural_Area'].replace({
    1: 'Urban',
    2: 'Rural',
    3: 'Unallocated'
})

accidents_vehicles_casualties['Special_Conditions_at_Site'] = accidents_vehicles_casualties['Special_Conditions_at_Site'].replace({
    'Auto traffic signal - out': 'No signal',
    'Auto signal part defective': 'Signal defect',
    'Oil or diesel': 'Oil',
    'Road sign or marking defective or obscured': 'Sign defect',
    'Road surface defective': 'Road defect'
})


# Histogram plots with KDE and probability plot
cont_columns = list(accidents_vehicles_casualties.select_dtypes(include=[np.number]).columns.values)

cont_columns.remove('Location_Easting_OSGR')
cont_columns.remove('Location_Northing_OSGR')
cont_columns.remove('Longitude')
cont_columns.remove('Latitude')
cont_columns.remove('Year')
cont_columns.remove('1st_Road_Number')
cont_columns.remove('2nd_Road_Number')
cont_columns.remove('Pedestrian_Crossing-Human_Control')
cont_columns.remove('Pedestrian_Crossing-Physical_Facilities')
cont_columns.remove('Vehicle_Reference_x')
cont_columns.remove('Vehicle_Reference_y')
cont_columns.remove('Pedestrian_Location')
cont_columns.remove('Pedestrian_Movement')
cont_columns.remove('Pedestrian_Road_Maintenance_Worker')
cont_columns.remove('Casualty_Home_Area_Type')
cont_columns.remove('Age_Band_of_Driver')
cont_columns.remove('Age_Band_of_Casualty')
cont_columns.remove('Driver_Home_Area_Type')
cont_columns.remove('Bus_or_Coach_Passenger')

print("Numerical Columns: \n", cont_columns)

cat_cols = list(accidents_vehicles_casualties.select_dtypes(include=['object']).columns)

cat_cols.remove('Accident_Severity')

#-----------------------------------------
# Line plots
#-----------------------------------------

# Create a line graph of the number of accidents per year
accidents_by_year = accidents_vehicles_casualties.groupby('Year').count()['Accident_Index'].reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Accident_Index', data=accidents_by_year, palette='bright')
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents per Year')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Create a line graph of the number of casualties per year
casualties_by_year = accidents_vehicles_casualties.groupby('Year').sum()['Number_of_Casualties'].reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Number_of_Casualties', data=casualties_by_year, palette='bright')
plt.xlabel('Year')
plt.ylabel('Number of Casualties')
plt.title('Number of Casualties per Year')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

#-----------------------------------------
# Bar graphs
#-----------------------------------------

# Create a bar chart of the number of accidents per year with numbers on the top of plot as well as a line gping thorugh all the plots
plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Accident_Index', data=accidents_by_year, palette='bright')
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents per Year')
plt.xticks(rotation=45)

for k, v in accidents_by_year['Accident_Index'].items():
    if v > 1000:
        plt.text(k, v + 100, str(v), fontsize=12, color="black", ha="center",
                 rotation=90, va="bottom")

accidents_by_year['Accident_Index'].plot(label='Number of Accidents', color='black', linewidth=3,
                                         ax=plt.twinx(), marker='o', markersize=10)
plt.grid(True)
plt.show()

# Create a bar chart of the number of accidents per year per severity
accidents_by_year_severity = accidents_vehicles_casualties.groupby(['Year', 'Accident_Severity']).count()['Accident_Index'].reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Accident_Index', hue='Accident_Severity', data=accidents_by_year_severity, palette='bright')
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents per Year per Severity')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Create a bar chart of casualties per year
plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Number_of_Casualties', data=casualties_by_year, palette='bright')
plt.xlabel('Year')
plt.ylabel('Number of Casualties')
plt.title('Number of Casualties per Year')
plt.xticks(rotation=45)

for k, v in casualties_by_year['Number_of_Casualties'].items():
    if v > 1000:
        plt.text(k, v + 100, str(v), fontsize=12, color="black", ha="center",
                 rotation=90, va="bottom")

casualties_by_year['Number_of_Casualties'].plot(label='Number of Casualties', color='black', linewidth=3,
                                                ax=plt.twinx(), marker='o', markersize=10)
plt.grid(True)
plt.show()

# Create a bar chart of casualties per year per severity
casualties_by_year_severity = accidents_vehicles_casualties.groupby(['Year', 'Accident_Severity']).sum()['Number_of_Casualties'].reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Number_of_Casualties', hue='Accident_Severity', data=casualties_by_year_severity, palette='bright')
plt.xlabel('Year')
plt.ylabel('Number of Casualties')
plt.title('Number of Casualties per Year per Severity')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Create a bar graph that checks whether majority of the accidents happen during the urban or rural areas, with line graph of casulties based on the settlements
accidents_by_settlement = accidents_vehicles_casualties.groupby('Urban_or_Rural_Area').count()['Accident_Index'].reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Urban_or_Rural_Area', y='Accident_Index', data=accidents_by_settlement, palette='bright')
plt.xlabel('Settlement Type')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents per Settlement Type')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

time.sleep(10)
time.sleep(5)

# Create a graph of urban/rural accidents based on severity
accidents_by_settlement_severity = accidents_vehicles_casualties.groupby(['Urban_or_Rural_Area', 'Accident_Severity']).count()['Accident_Index'].reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Urban_or_Rural_Area', y='Accident_Index', hue='Accident_Severity', data=accidents_by_settlement_severity, palette='bright')
plt.xlabel('Settlement Type')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents per Settlement Type per Severity')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Create a graph explaining of many vehicles were involved in accidents based on the severity, and how many casulaties were involved in the accidents based on the severity
accidents_by_vehicles = accidents_vehicles_casualties.groupby('Number_of_Vehicles').count()['Accident_Index'].reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Number_of_Vehicles', y='Accident_Index', data=accidents_by_vehicles, palette='bright')
plt.xlabel('Number of Vehicles')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents per Number of Vehicles')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Create a graph to check whether many accidents happened in the weekdays or the weekends
accidents_by_day = accidents_vehicles_casualties.groupby('Day_of_Week').count()['Accident_Index'].reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Day_of_Week', y='Accident_Index', data=accidents_by_day, palette='bright')
plt.xlabel('Day of Week')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents per Day of Week')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Create a graph to check whether many accidents happened in the weekdays or the weekends based on severity
accidents_by_day_severity = accidents_vehicles_casualties.groupby(['Day_of_Week', 'Accident_Severity']).count()['Accident_Index'].reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Day_of_Week', y='Accident_Index', hue='Accident_Severity', data=accidents_by_day_severity, palette='bright')
plt.xlabel('Day of Week')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents per Day of Week per Severity')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Create a bar graph to check whether accident severity is based on the type of road
accidents_by_road = accidents_vehicles_casualties.groupby(['Road_Type', 'Accident_Severity']).count()['Accident_Index'].reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Road_Type', y='Accident_Index', data=accidents_by_road, palette='bright', hue='Accident_Severity')
plt.xlabel('Road Type')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents per Road Type')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

#Create a groupby function depicting the road surface conditions during the accidents
accidents_by_surface = accidents_vehicles_casualties.groupby('Road_Surface_Conditions').count()['Accident_Index'].reset_index()

# Create a bar graph to check whether accident severity is based on the surface condition of the road
accidents_by_surface_severity = accidents_vehicles_casualties.groupby(['Road_Surface_Conditions', 'Accident_Severity']).count()['Accident_Index'].reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Road_Surface_Conditions', y='Accident_Index', data=accidents_by_surface_severity, palette='bright', hue='Accident_Severity')
plt.xlabel('Road Surface Condition')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents per Road Surface Condition')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Create a graph depicting severity if accidents based on the speed limits
accidents_by_speed = accidents_vehicles_casualties.groupby(['Speed_limit', 'Accident_Severity']).count()['Accident_Index'].reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Speed_limit', y='Accident_Index', data=accidents_by_speed, palette='bright', hue='Accident_Severity')
plt.xlabel('Speed Limit')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents per Speed Limit')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Create a graph depicting which kind of vehicles were involved in accidents
accidents_by_vehicle = accidents_vehicles_casualties.groupby(['Vehicle_Type', 'Accident_Severity']).count()['Accident_Index'].reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Vehicle_Type', y='Accident_Index', data=accidents_by_vehicle, palette='bright', hue='Accident_Severity')
plt.xlabel('Vehicle Type')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents per Vehicle Type')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

#Create a groupby function depicting the weather conditions during the accidents
accidents_by_weather = accidents_vehicles_casualties.groupby('Weather_Conditions').count()['Accident_Index'].reset_index()

# Create a graph depicting the weather conditions during the accidents based on severity
accidents_by_weather_severity = accidents_vehicles_casualties.groupby(['Weather_Conditions', 'Accident_Severity']).count()['Accident_Index'].reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Weather_Conditions', y='Accident_Index', data=accidents_by_weather_severity, palette='bright', hue='Accident_Severity')
plt.xlabel('Weather Condition')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents per Weather Condition')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

#Create a groupby function depicting the light conditions during the accidents
accidents_by_light = accidents_vehicles_casualties.groupby('Light_Conditions').count()['Accident_Index'].reset_index()

# Create a graph depicting the light conditions during the accidents based on severity
accidents_by_light_severity = accidents_vehicles_casualties.groupby(['Light_Conditions', 'Accident_Severity']).count()['Accident_Index'].reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Light_Conditions', y='Accident_Index', data=accidents_by_light_severity, palette='bright', hue='Accident_Severity')
plt.xlabel('Light Condition')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents per Light Condition')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

#Create a groupby function depicting the special conditions during the accidents
accidents_by_special = accidents_vehicles_casualties.groupby('Special_Conditions_at_Site').count()['Accident_Index'].reset_index()

# Create a graph depicting the special conditions during the accidents based on severity
accidents_by_special_severity = accidents_vehicles_casualties.groupby(['Special_Conditions_at_Site', 'Accident_Severity']).count()['Accident_Index'].reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Special_Conditions_at_Site', y='Accident_Index', data=accidents_by_special_severity, palette='bright', hue='Accident_Severity')
plt.xlabel('Special Condition')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents per Special Condition')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# Create a graph of how the accidents happened
accidents_happen_how = accidents_vehicles_casualties.groupby(['Skidding_and_Overturning', 'Accident_Severity']).count()['Accident_Index'].reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Skidding_and_Overturning', y='Accident_Index', data=accidents_happen_how, palette='bright', hue='Accident_Severity')
plt.xlabel('Accident Incidents')
plt.ylabel('Number of Accidents')
plt.title('How the Accidents Happened')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Create a graph of what is the 1st point of impact
accidents_by_impact = accidents_vehicles_casualties.groupby(['1st_Point_of_Impact', 'Accident_Severity']).count()['Accident_Index'].reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='1st_Point_of_Impact', y='Accident_Index', data=accidents_by_impact, palette='bright', hue='Accident_Severity')
plt.xlabel('1st Point of Impact')
plt.ylabel('Number of Accidents')
plt.title('1st Point of Impact')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Create a graph based on the how many accidents were reported to the police
accidents_by_police = accidents_vehicles_casualties.groupby('Did_Police_Officer_Attend_Scene_of_Accident').count()['Accident_Index'].reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Did_Police_Officer_Attend_Scene_of_Accident', y='Accident_Index', data=accidents_by_police, palette='bright')
plt.xlabel('Police Attended')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents reported to Police')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

time.sleep(10)
time.sleep(10)
time.sleep(10)

# Create a graph depicting no of accidents to the sex of the casualty
accidents_by_sex = accidents_vehicles_casualties.groupby('Sex_of_Casualty').count()['Accident_Index'].reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Sex_of_Casualty', y='Accident_Index', data=accidents_by_sex, palette='bright')
plt.xlabel('Sex of Casualty')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents per Sex of Casualty')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Create a graph depicting no of accidents to the sex of the driver
accidents_by_sex_driver = accidents_vehicles_casualties.groupby('Sex_of_Driver').count()['Accident_Index'].reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Sex_of_Driver', y='Accident_Index', data=accidents_by_sex_driver, palette='bright')
plt.xlabel('Sex of the Driver')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents per Sex of the Driver')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

#-----------------------------------------
# Pie charts
#-----------------------------------------

def pie_plts(df, column):
    plt.figure(figsize=(10, 6))
    df[column].value_counts().plot(kind='pie', autopct='%1.1f%%', shadow=True)
    plt.title(column)
    plt.show()

# Create a pie chart of accident severity
pie_plts(accidents_vehicles_casualties, 'Accident_Severity')

# Create a pie chart of settlement type
pie_plts(accidents_vehicles_casualties, 'Urban_or_Rural_Area')

# Create a pie chart of road type accidents
pie_plts(accidents_vehicles_casualties, 'Road_Type')

# Create a pie chart of road surface conditions
pie_plts(accidents_vehicles_casualties, 'Road_Surface_Conditions')

# Create a pie chart of weather conditions
pie_plts(accidents_vehicles_casualties, 'Weather_Conditions')

# Create a pie chart of light conditions
pie_plts(accidents_vehicles_casualties, 'Light_Conditions')

# Create a pie chart of whether police attended the scene of accident
pie_plts(accidents_vehicles_casualties, 'Did_Police_Officer_Attend_Scene_of_Accident')

time.sleep(10)

# Create a pie chart of first point of impact
pie_plts(accidents_vehicles_casualties, '1st_Point_of_Impact')

#-----------------------------------------
# Count plots
#-----------------------------------------

def count_plts(df, column):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df)
    plt.xlabel(column)
    plt.ylabel('Number of Accidents')
    plt.title('Number of Accidents per ' + column)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

# Count plot of number of vechicles affected
count_plts(accidents_vehicles_casualties, 'Number_of_Vehicles')

# Count plot of number of casualties affected
count_plts(accidents_vehicles_casualties, 'Number_of_Casualties')

# Count plot of age of the vehicles that were involved in accidents
count_plts(accidents_vehicles_casualties, 'Age_of_Vehicle')

time.sleep(10)

#-----------------------------------------
# Pairplot
#-----------------------------------------

# Create a pairplot of selected features
pairplot_vars = pd.DataFrame(accidents_vehicles_casualties, columns= cont_columns)
pairplot_vars['Accident_Severity'] = accidents_vehicles_casualties['Accident_Severity']

sns.pairplot(data=pairplot_vars, hue="Accident_Severity" , palette='bright')
plt.title('Pairplot of Selected Features')
plt.show()

#-----------------------------------------
# Heatmap
#-----------------------------------------

# Create a heatmap of selected features
heatmap_vars = pd.DataFrame(accidents_vehicles_casualties, columns= cont_columns)

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_vars.corr(method='spearman'), annot=True, cmap='icefire', cbar=False)
plt.title('Correlation between the Features')
plt.show()

time.sleep(10)

#-----------------------------------------
# Subplots
#-----------------------------------------

# Create a subplot of number of accidents per year and number of casualties per year
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
plt.suptitle('Number of Accidents and Casualties per Year', fontsize=20)

# Number of accidents per year line plot
sns.lineplot(x='Year', y='Accident_Index', data=accidents_by_year, palette='bright', ax=axes[0, 0])
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Number of Accidents')
axes[0, 0].set_title('Number of Accidents per Year')
axes[0, 0].grid(True)
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)

# Number of accidents per year bar plot per severity
sns.barplot(x='Year', y='Accident_Index', data=accidents_by_year, palette='bright', ax=axes[0, 1])
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Number of Accidents')
axes[0, 1].set_title('Number of Accidents per Year')
axes[0, 1].grid(True)
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)

# Number of accidents per year per severity
sns.barplot(x='Year', y='Accident_Index', hue='Accident_Severity', data=accidents_by_year_severity, palette='bright', ax=axes[0, 2])
axes[0, 2].set_xlabel('Year')
axes[0, 2].set_ylabel('Number of Accidents')
axes[0, 2].set_title('Number of Accidents per Year per Severity')
axes[0, 2].grid(True)
axes[0, 2].set_xticklabels(axes[0, 2].get_xticklabels(), rotation=45)

# Number of casualties per year line plot
sns.lineplot(x='Year', y='Number_of_Casualties', data=casualties_by_year, palette='bright', ax=axes[1, 0])
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Number of Casualties')
axes[1, 0].set_title('Number of Casualties per Year')
axes[1, 0].grid(True)
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)

# Number of casualties per year bar plot
sns.barplot(x='Year', y='Number_of_Casualties', data=casualties_by_year, palette='bright', ax=axes[1, 1])
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Number of Casualties')
axes[1, 1].set_title('Number of Casualties per Year')
axes[1, 1].grid(True)
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)

# Number of casualties per year per severity
sns.barplot(x='Year', y='Number_of_Casualties', hue='Accident_Severity', data=casualties_by_year_severity, palette='bright', ax=axes[1, 2])
axes[1, 2].set_xlabel('Year')
axes[1, 2].set_ylabel('Number of Casualties')
axes[1, 2].set_title('Number of Casualties per Year per Severity')
axes[1, 2].grid(True)
axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.legend()
plt.show()

time.sleep(15)

# Create a subplot for the different conditions that lead to an accident
fig, axes = plt.subplots(3, 3, figsize=(25, 15))
plt.suptitle('Conditions that lead to an Accident', fontsize=20)

# Road surface conditions
sns.barplot(x='Road_Surface_Conditions', y='Accident_Index', data=accidents_by_surface, palette='bright', ax=axes[0, 0])
axes[0, 0].set_xlabel('Road Surface Condition')
axes[0, 0].set_ylabel('Number of Accidents')
axes[0, 0].set_title('Total Accidents based on Road Surface Condition')
axes[0, 0].grid(True)
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)

# Road surface conditions per severity
sns.barplot(x='Road_Surface_Conditions', y='Accident_Index', hue='Accident_Severity', data=accidents_by_surface_severity, palette='bright', ax=axes[0, 1])
axes[0, 1].set_xlabel('Road Surface Condition')
axes[0, 1].set_ylabel('Number of Accidents')
axes[0, 1].set_title('Total Accidents Severity based on Road Surface Condition')
axes[0, 1].grid(True)
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)

# Raod surface pie chart
accidents_vehicles_casualties['Road_Surface_Conditions'].value_counts().plot(kind='pie', autopct='%1.1f%%', shadow=True, ax=axes[0, 2])
axes[0, 2].set_title('Road Surface Condition')
axes[0, 2].set_ylabel('')
axes[0, 2].set_xlabel('')
axes[0, 2].set_xticklabels(axes[0, 2].get_xticklabels(), rotation=45)

# Weather conditions
sns.barplot(x='Weather_Conditions', y='Accident_Index', data=accidents_by_weather, palette='bright', ax=axes[1, 0])
axes[1, 0].set_xlabel('Weather Condition')
axes[1, 0].set_ylabel('Number of Accidents')
axes[1, 0].set_title('Total Accidents based on Weather Condition')
axes[1, 0].grid(True)
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)

# Weather conditions per severity
sns.barplot(x='Weather_Conditions', y='Accident_Index', hue='Accident_Severity', data=accidents_by_weather_severity, palette='bright', ax=axes[1, 1])
axes[1, 1].set_xlabel('Weather Condition')
axes[1, 1].set_ylabel('Number of Accidents')
axes[1, 1].set_title('Total Accidents Severity based on Weather Condition')
axes[1, 1].grid(True)
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)

# Weather conditions pie chart
accidents_vehicles_casualties['Weather_Conditions'].value_counts().plot(kind='pie', autopct='%1.1f%%', shadow=True, ax=axes[1, 2])
axes[1, 2].set_title('Weather Condition')
axes[1, 2].set_ylabel('')
axes[1, 2].set_xlabel('')
axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=45)

# Light conditions
sns.barplot(x='Light_Conditions', y='Accident_Index', data=accidents_by_light, palette='bright', ax=axes[2, 0])
axes[2, 0].set_xlabel('Light Condition')
axes[2, 0].set_ylabel('Number of Accidents')
axes[2, 0].set_title('Total Accidents based on Light Condition')
axes[2, 0].grid(True)
axes[2, 0].set_xticklabels(axes[2, 0].get_xticklabels(), rotation=45)

# Light conditions per severity
sns.barplot(x='Light_Conditions', y='Accident_Index', hue='Accident_Severity', data=accidents_by_light_severity, palette='bright', ax=axes[2, 1])
axes[2, 1].set_xlabel('Light Condition')
axes[2, 1].set_ylabel('Number of Accidents')
axes[2, 1].set_title('Total Accidents Severity based on Light Condition')
axes[2, 1].grid(True)
axes[2, 1].set_xticklabels(axes[2, 1].get_xticklabels(), rotation=45)

# Light conditions pie chart
accidents_vehicles_casualties['Light_Conditions'].value_counts().plot(kind='pie', autopct='%1.1f%%', shadow=True, ax=axes[2, 2])
axes[2, 2].set_title('Light Condition')
axes[2, 2].set_ylabel('')
axes[2, 2].set_xlabel('')
axes[2, 2].set_xticklabels(axes[2, 2].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()

time.sleep(15)

# Create a subplot depicting the damage to the vehicles

# Age of vehicles
accidents_by_age = accidents_vehicles_casualties.groupby('Age_of_Vehicle').count()['Accident_Index'].reset_index()

# Type of vehicles
accidents_by_vehicle_affected = accidents_vehicles_casualties.groupby('Vehicle_Type').count()['Accident_Index'].reset_index()

# 1st point of impact
accidents_by_impact = accidents_vehicles_casualties.groupby('1st_Point_of_Impact').count()['Accident_Index'].reset_index()

fig, axes = plt.subplots(2, 2, figsize=(20, 10))
plt.suptitle('Vehicles affected in Accidents', fontsize=20)

# Number of vehicles affected
sns.barplot(x='Number_of_Vehicles', y='Accident_Index', data=accidents_by_vehicles, palette='bright', ax=axes[0, 0])
axes[0, 0].set_xlabel('Number of Vehicles')
axes[0, 0].set_ylabel('Number of Accidents')
axes[0, 0].set_title('Number of Accidents based on Number of Vehicles')
axes[0, 0].grid(True)
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)

# Age of vehicles affected
sns.barplot(x='Age_of_Vehicle', y='Accident_Index', data=accidents_by_age, palette='bright', ax=axes[0, 1])
axes[0, 1].set_xlabel('Age of Vehicles')
axes[0, 1].set_ylabel('Number of Accidents')
axes[0, 1].set_title('Number of Accidents based on Age of Vehicles')
axes[0, 1].grid(True)

# Type of vehicles affected
sns.barplot(x='Vehicle_Type', y='Accident_Index', data=accidents_by_vehicle_affected, palette='bright', ax=axes[1, 0])
axes[1, 0].set_xlabel('Type of Vehicles')
axes[1, 0].set_ylabel('Number of Accidents')
axes[1, 0].set_title('Number of Accidents based on Type of Vehicles')
axes[1, 0].grid(True)
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)

# 1st point of impact on the vehicles affected
sns.barplot(x='1st_Point_of_Impact', y='Accident_Index', data=accidents_by_impact, palette='bright', ax=axes[1, 1])
axes[1, 1].set_xlabel('1st Point of Impact')
axes[1, 1].set_ylabel('Number of Accidents')
axes[1, 1].set_title('Number of Accidents based on 1st Point of Impact')
axes[1, 1].grid(True)
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()

time.sleep(15)

#-----------------------------------------
# Create a subplot for dist plot, histogram with KDE, KDE plot, Q-Q plot
#-----------------------------------------

def plot_continuous_columns(df, columns):
    for col in columns:
        if col in df.columns:

            fig, axes = plt.subplots(4, 1, figsize=(25, 36))

            # Displot
            sns.histplot(df[col], kde=False, ax=axes[0])
            axes[0].set_title(f'Distribution Plot for {col}')
            axes[0].set_xlabel(col)
            axes[0].set_ylabel('Density')

            # Histogram with KDE
            sns.histplot(df[col], kde=True, ax=axes[1], stat="density", linewidth=3, alpha=0.6)
            axes[1].set_title(f'Histogram with KDE for {col}')
            axes[1].set_xlabel(col)
            axes[1].set_ylabel('Density')

            # KDE plot
            sns.kdeplot(df[col], ax=axes[2], fill=True)
            axes[2].set_title(f'KDE Plot for {col}')
            axes[2].set_xlabel(col)
            axes[2].set_ylabel('Density')

            # Q-Q plot
            stats.probplot(df[col], dist="norm", plot=axes[3])
            axes[3].set_title(f'Q-Q plot of {col}')
            axes[3].set_xlabel('Theoretical Quantiles')
            axes[3].set_ylabel('Ordered Values')

            time.sleep(5)

            plt.tight_layout()
            plt.show()
        else:
            print(f'Column {col} not found in DataFrame')

plot_continuous_columns(accidents_vehicles_casualties, cont_columns)

time.sleep(10)


#-----------------------------------------
# Area plots
#-----------------------------------------

# Area plots of casualties per year
plt.figure(figsize=(10, 6))
casualties_by_year.plot.area(x='Year', y='Number_of_Casualties', color='black', linewidth=3, alpha=0.6)
plt.xlabel('Year')
plt.ylabel('Number of Casualties')
plt.title('Number of Casualties per Year')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

time.sleep(10)

# Area plot of number of vehicles affected per year
plt.figure(figsize=(10, 6))
accidents_by_year.plot.area(x='Year', y='Accident_Index', color='black', linewidth=3, alpha=0.6)
plt.xlabel('Year')
plt.ylabel('Number of Vehicles affected')
plt.title('Number of Vehicles affected per Year')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

time.sleep(10)

#-----------------------------------------------
# Joint plot with KDE and scatter representation
#-----------------------------------------------

# Create a joint plot of speed limit and engine capacity
plt.figure(figsize=(10, 6))
sns.jointplot(x='Speed_limit', y='Engine_Capacity_(CC)', data=accidents_vehicles_casualties, kind='kde', color='black')
plt.title('Joint Plot of Speed Limit and Engine Capacity')
plt.tight_layout()
plt.show()

time.sleep(5)

# Create a joint plot of age of driver and number of casualties
plt.figure(figsize=(10, 6))
sns.jointplot(x='Age_of_Driver', y='Number_of_Casualties', data=accidents_vehicles_casualties, kind='kde', color='black')
plt.title('Joint Plot of Age of Driver and Number of Casualties')
plt.tight_layout()
plt.show()

time.sleep(5)

# Create a joint plot of age of vehicle and number of casualties
plt.figure(figsize=(10, 6))
sns.jointplot(x='Age_of_Vehicle', y='Number_of_Casualties', data=accidents_vehicles_casualties, kind='kde', color='black')
plt.title('Joint Plot of Age of Vehicle and Number of Casualties')
plt.tight_layout()
plt.show()

time.sleep(5)

#-----------------------------------------
# Rug plots
#-----------------------------------------

# Create a rug plot of age of driver
plt.figure(figsize=(10, 6))
sns.rugplot(x='Age_of_Driver', data=accidents_vehicles_casualties)
plt.title('Rug Plot of Age of Driver')
plt.show()

time.sleep(5)

# Create a rug plot of age of vehicle
plt.figure(figsize=(10, 6))
sns.rugplot(x='Age_of_Vehicle', data=accidents_vehicles_casualties)
plt.title('Rug Plot of Age of Vehicle')
plt.show()

time.sleep(5)

# Create a reg plot of engine capacity
plt.figure(figsize=(10, 6))
sns.rugplot(x='Engine_Capacity_(CC)', data=accidents_vehicles_casualties)
plt.title('Rug Plot of Engine Capacity')
plt.show()

time.sleep(5)

#-----------------------------------------
# Cluster map
#-----------------------------------------

# Create a cluster map
plt.figure(figsize=(10, 6))
sns.clustermap(accidents_by_vehicles, cmap='coolwarm')
plt.title('Cluster Map')
plt.show()

time.sleep(5)

#-----------------------------------------
# Hexbin plot
#-----------------------------------------

# Create a hexbin plot of age of driver against number of casualties
plt.figure(figsize=(10, 6))
plt.hexbin(x='Age_of_Driver', y='Number_of_Casualties', data=accidents_vehicles_casualties, gridsize=25, cmap='coolwarm')
plt.title('Hexbin Plot of Age of Driver against Number of Casualties')
plt.show()

time.sleep(5)

# Create a hexbin plot of speed limit against number of vehicles
plt.figure(figsize=(10, 6))
plt.hexbin(x='Speed_limit', y='Number_of_Vehicles', data=accidents_vehicles_casualties, gridsize=25, cmap='coolwarm')
plt.title('Hexbin Plot of Speed Limit against Number of Vehicles')
plt.show()

time.sleep(5)

#-----------------------------------------
# Strip plot
#-----------------------------------------

# Create a strip plot of engine capacity
plt.figure(figsize=(10, 6))
sns.stripplot(x='Engine_Capacity_(CC)', data=accidents_vehicles_casualties)
plt.title('Strip Plot of Engine Capacity')
plt.show()

time.sleep(5)

# Create a strip plot of speed limit
plt.figure(figsize=(10, 6))
sns.stripplot(x='Speed_limit', data=accidents_vehicles_casualties)
plt.tight_layout()
plt.show()

time.sleep(5)

# Create a strip plot of age of driver
plt.figure(figsize=(10, 6))
sns.stripplot(x='Age_of_Driver', data=accidents_vehicles_casualties)
plt.title('Strip Plot of Age of Driver')
plt.show()

time.sleep(5)

#-----------------------------------------
# Violin plots and swarm plot
#-----------------------------------------

# Create a subplot of violin plots and swarm plot for speed limit, engine capacity

fig, axes = plt.subplots(2, 2, figsize=(20, 10))
plt.suptitle('Subplots of Violin Plots and Swarm Plot', fontsize=20)

# Speed limit
sns.violinplot(x='Speed_limit', y='Accident_Index', data=accidents_by_speed, palette='bright', ax=axes[0, 0])
axes[0, 0].set_xlabel('Speed Limit')
axes[0, 0].set_ylabel('Number of Accidents')
axes[0, 0].set_title('Number of Accidents due to Speed Limit')
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)

sns.swarmplot(x='Speed_limit', y='Accident_Index', data=accidents_by_speed, color='black', ax=axes[0, 1])
axes[0, 1].set_xlabel('Speed Limit')
axes[0, 1].set_ylabel('Number of Accidents')
axes[0, 1].set_title('Number of Accidents due to Speed Limit')
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)

# Engine capacity
sns.violinplot(x='Engine_Capacity_(CC)', y='Accident_Index', data=accidents_vehicles_casualties, palette='bright', ax=axes[1, 0])
axes[1, 0].set_xlabel('Engine Capacity')
axes[1, 0].set_ylabel('Number of Accidents')
axes[1, 0].set_title('Engine capacity of vehicles involved in Accidents')
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)

sns.swarmplot(x='Engine_Capacity_(CC)', y='Accident_Index', data=accidents_vehicles_casualties, color='black', ax=axes[1, 1])
axes[1, 1].set_xlabel('Engine Capacity')
axes[1, 1].set_ylabel('Number of Accidents')
axes[1, 1].set_title('Engine capacity of vehicles involved in Accidents')
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.grid(True)
plt.show()

time.sleep(10)

# Violin plot of speed limit
plt.figure(figsize=(10, 6))
sns.violinplot(x='Speed_limit', y='Accident_Index', data=accidents_by_speed, palette='bright')
plt.xlabel('Speed Limit')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents due to Speed Limit')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

time.sleep(5)


#-----------------------------------------
# 3d plot and contour plot
#-----------------------------------------

# Create a 3d plot of number of vehicles, number of casualties, and speed limit
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=accidents_vehicles_casualties['Number_of_Vehicles'], ys=accidents_vehicles_casualties['Number_of_Casualties'], zs=accidents_vehicles_casualties['Speed_limit'], c='black', marker='o')
ax.set_xlabel('Number of Vehicles')
ax.set_ylabel('Number of Casualties')
ax.set_zlabel('Speed Limit')
plt.title('3D Plot of Number of Vehicles, Number of Casualties, and Speed Limit')
plt.show()

time.sleep(5)

# Create a contour plot of number of vehicles, number of casualties, and speed limit
accidents_axis = accidents_vehicles_casualties[['Number_of_Vehicles', 'Number_of_Casualties', 'Speed_limit']]

X, Y = np.meshgrid(np.linspace(accidents_axis['Number_of_Vehicles'].min(), accidents_axis['Number_of_Vehicles'].max(), 100),
                   np.linspace(accidents_axis['Number_of_Casualties'].min(), accidents_axis['Number_of_Casualties'].max(), 100))

Z = griddata((accidents_axis['Number_of_Vehicles'], accidents_axis['Number_of_Casualties']),
             accidents_axis['Speed_limit'],
             (X, Y), method='cubic')

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.contour(X, Y, Z)
ax.set_xlabel('Number of Vehicles')
ax.set_ylabel('Number of Casualties')
ax.set_zlabel('Speed Limit')
plt.title('Contour Plot of Number of Vehicles, Number of Casualties, and Speed Limit')
plt.show()

time.sleep(10)

#-----------------------------------------
# Scatter matrix plot
#-----------------------------------------
plt.figure(figsize=(35, 35))

axs = pd.plotting.scatter_matrix(accidents_vehicles_casualties[cont_columns], alpha=0.1)

for ax in axs[:,0]:
    ax.set_ylabel(ax.get_ylabel(), rotation=45, horizontalalignment='right')

for i in range(len(cont_columns)):
    for j in range(len(cont_columns)):
        axs[i, j].xaxis.label.set_rotation(45)
        axs[i, j].yaxis.label.set_rotation(45)
        axs[i, j].yaxis.label.set_ha('right')

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)

for i in range(len(cont_columns)):
    f = plt.gcf()
    f.axes[i * len(cont_columns) + i].clear()
    pd.Series(accidents_vehicles_casualties[cont_columns[i]]).plot(kind='kde', ax=f.axes[i * len(cont_columns) + i])

plt.suptitle('Scatter Matrix Plot', y=0.95)

plt.show()

time.sleep(10)

#-----------------------------------------
# Tables
#-----------------------------------------

# Create a table of number of accidents per year with prettytable
table = PrettyTable()
table.field_names = ['Year', 'Number of Accidents']
for i in range(len(accidents_by_year)):
    table.add_row([accidents_by_year['Year'][i], accidents_by_year['Accident_Index'][i]])

print(table)

time.sleep(10)

# Create a table of number of casualties per year with prettytable
table = PrettyTable()
table.field_names = ['Year', 'Number of Casualties']
for i in range(len(casualties_by_year)):
    table.add_row([casualties_by_year['Year'][i], casualties_by_year['Number_of_Casualties'][i]])

print(table)

time.sleep(10)


accidents_by_hazards = accidents_vehicles_casualties.groupby('Carriageway_Hazards').count()['Accident_Index'].reset_index()
# Create a table of carriage way hazards with prettytable
table = PrettyTable()
table.field_names = ['Carriage Way Hazards', 'Number of Accidents']
for i in range(len(accidents_by_hazards)):
    table.add_row([accidents_by_hazards['Carriageway_Hazards'][i], accidents_by_hazards['Accident_Index'][i]])

print(table)

time.sleep(10)

# Create a table of raod surface conditions with prettytable
table = PrettyTable()
table.field_names = ['Road Surface Condition', 'Number of Accidents']
for i in range(len(accidents_by_surface)):
    table.add_row([accidents_by_surface['Road_Surface_Conditions'][i], accidents_by_surface['Accident_Index'][i]])

print(table)

time.sleep(10)

# Create a table of weather conditions with prettytable
table = PrettyTable()
table.field_names = ['Weather Condition', 'Number of Accidents']
for i in range(len(accidents_by_weather)):
    table.add_row([accidents_by_weather['Weather_Conditions'][i], accidents_by_weather['Accident_Index'][i]])

print(table)


accidents_vehicles_casualties.drop(['Location_Easting_OSGR', 'Location_Northing_OSGR', 'Longitude', 'Latitude',
                                    'Police_Force', 'Local_Authority_(District)', 'Local_Authority_(Highway)',
                                    '1st_Road_Number', '2nd_Road_Number', 'Accident_Index', 'Date', 'Time',
                                    'LSOA_of_Accident_Location', 'Year', 'Journey_Purpose_of_Driver', 'Sex_of_Driver',
                                    'Skidding_and_Overturning', 'Special_Conditions_at_Site', 'Carriageway_Hazards',
                                    'Hit_Object_in_Carriageway', 'Hit_Object_off_Carriageway', 'Vehicle_Leaving_Carriageway',
                                    'Hit_Object_off_Carriageway', 'Driver_IMD_Decile', 'Age_of_Casualty', 'Casualty_Type',
                                    'Vehicle_Reference_x', 'Vehicle_Reference_y', 'Pedestrian_Crossing-Human_Control',
                                    'Pedestrian_Crossing-Physical_Facilities', 'Pedestrian_Road_Maintenance_Worker', 'Casualty_Home_Area_Type',
                                    'Age_Band_of_Driver', 'Age_Band_of_Casualty', 'Pedestrian_Location', 'Pedestrian_Movement', 'Bus_or_Coach_Passenger',
                                    'Driver_Home_Area_Type'], axis=1, inplace=True)

#-----------------------------------------
# Box plot
#-----------------------------------------

plt.figure(figsize=(20, 10))
sns.boxplot(data=accidents_vehicles_casualties, orient='h')
plt.show()

#-----------------------------------------
# IQR
#-----------------------------------------

# Create a function to calculate the IQR
def iqr(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper

lower, upper = iqr(accidents_vehicles_casualties['Engine_Capacity_(CC)'])
accidents_vehicles_casualties = accidents_vehicles_casualties[(accidents_vehicles_casualties['Engine_Capacity_(CC)'] > lower) & (accidents_vehicles_casualties['Engine_Capacity_(CC)'] < upper)]

# Create a box plot of the data after removing the outliers
plt.figure(figsize=(20, 10))
sns.boxplot(data=accidents_vehicles_casualties, orient='h')
plt.show()

# -----------------------------------------
# Normality test
# -----------------------------------------

# Create a function to test the normality of the data to check if numerical data is normally distributed
def shapiro_test(x, title):
    stats, p = shapiro(x.dropna())
    print('=' * 50)
    print(f'Shapiro test : {title} dataset : statistics = {stats:.2f} p-value of ={p:.2f}' )
    if p > 0.05:
        print(f'{title} dataset is normal')
    else:
        print(f'{title} dataset is not normal')


shapiro_test(accidents_vehicles_casualties['Number_of_Vehicles'], 'Number_of_Vehicles')
shapiro_test(accidents_vehicles_casualties['Number_of_Casualties'], 'Number_of_Casualties')
shapiro_test(accidents_vehicles_casualties['Speed_limit'], 'Speed_limit')
shapiro_test(accidents_vehicles_casualties['Age_of_Driver'], 'Age_of_Driver')
shapiro_test(accidents_vehicles_casualties['Engine_Capacity_(CC)'], 'Engine_Capacity_(CC)')
shapiro_test(accidents_vehicles_casualties['Age_of_Vehicle'], 'Age_of_Vehicle')
shapiro_test(accidents_vehicles_casualties['Age_of_Driver'], 'Age_of_Driver')

# Remove all the negative values from the numerical columns
accidents_vehicles_casualties = accidents_vehicles_casualties[(accidents_vehicles_casualties['Age_of_Driver'] > 0) & (accidents_vehicles_casualties['Age_of_Vehicle'] > 0) & (accidents_vehicles_casualties['Engine_Capacity_(CC)'] > 0)]


# If the data is not normally distributed, we can make it gaussian by applying a boxcox transformation
def boxcox_transform(x):
    transformed_data, lamda = boxcox(x)
    print(f'Best lambda value: {lamda:.2f}')

    # After transformation
    sns.distplot(transformed_data, kde=True)
    plt.title(f'After Transformation of {x.name}')
    plt.show()

    time.sleep(2)

boxcox_transform(accidents_vehicles_casualties['Number_of_Vehicles'])
boxcox_transform(accidents_vehicles_casualties['Number_of_Casualties'])
boxcox_transform(accidents_vehicles_casualties['Speed_limit'])
boxcox_transform(accidents_vehicles_casualties['Age_of_Driver'])
boxcox_transform(accidents_vehicles_casualties['Engine_Capacity_(CC)'])
boxcox_transform(accidents_vehicles_casualties['Age_of_Vehicle'])
boxcox_transform(accidents_vehicles_casualties['Age_of_Driver'])


cat_cols = list(accidents_vehicles_casualties.select_dtypes(include=['object']).columns)
cat_cols.remove('Accident_Severity')


# One-hot encode the categorical columns
accidents_vehicles_casualties = pd.get_dummies(accidents_vehicles_casualties, columns=cat_cols, drop_first=True)

# Scale the continuous columns
def scale(x):
    return (x - x.min()) / (x.max() - x.min())

accidents_vehicles_casualties[cont_columns] = accidents_vehicles_casualties[cont_columns].apply(scale, axis=0)


# Divide the data into features and target
X, y = accidents_vehicles_casualties.drop('Accident_Severity', axis=1), accidents_vehicles_casualties['Accident_Severity']

#Oversample the minority class
oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X, y)

# Check the balance of the classes
print('Processisng SMOTE...')
fig, ax = plt.subplots(1, 2, figsize = (10, 5))
y.value_counts().plot.pie(explode = [0, 0.1, 0.2], autopct = "%1.1f%%", ax = ax[0], colors = ['#ff9999', '#66b3ff', '#964B00'],
                          shadow = True)
ax[0].set_title("Before SMOTE")
ax[0].set_ylabel('')
y_smote.value_counts().plot.pie(explode = [0, 0.1, 0.2], autopct = "%1.1f%%", ax = ax[1],  colors = ['#66b3ff', '#ff9999', '#964B00'],
                                shadow = True)
ax[1].set_title("After SMOTE")
ax[1].set_ylabel('')

print("Overall dataset values have been increased.")
print("Original size:\n", y.value_counts())
print("New size:\n", y_smote.value_counts())
plt.show()

X_smote = sm.add_constant(X_smote)

print('New size of dataset:', X_smote.shape, y_smote.shape)

#-----------------------------------------
# PCA
#-----------------------------------------

# Start PCA
pca = PCA(svd_solver="full", n_components=0.9, random_state=5764)
X_pca = pca.fit_transform(X_smote)
print("Original Shape: ", X_smote.shape)
print("Reduced Shape: ", X_pca.shape)

print("Number of features needed to explain more than 90% of the dependent variance:",
    np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9)[0][0] + 1, )

plt.figure(figsize=(10, 10))
plt.plot(
    np.arange(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1, 1),
    np.cumsum(pca.explained_variance_ratio_), label="Cumulative Explained Variance",)
plt.xticks(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1, 1))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.axvline(x=(np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9)[0][0] + 1, ),
            color="red", linestyle="--")
plt.axhline(y=0.9, color="black", linestyle="--")
plt.legend()
plt.show()

# Conditional Number of Original and Reduced data
print('Condition Number of Original data:', np.linalg.cond(X_smote))
print('Condition Number of Reduced data:', np.linalg.cond(X_pca))

#-----------------------------------------
#End of Exploratory Data Analysis
#-----------------------------------------








































