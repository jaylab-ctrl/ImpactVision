import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
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
    time.sleep(5)
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

time.sleep(10)

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
    time.sleep(10)
    plt.show()

# Count plot of number of vechicles affected
count_plts(accidents_vehicles_casualties, 'Number_of_Vehicles')

# Count plot of number of casualties affected
count_plts(accidents_vehicles_casualties, 'Number_of_Casualties')


# Count plot of age of the vehicles that were involved in accidents
count_plts(accidents_vehicles_casualties, 'Age_of_Vehicle')

time.sleep(10)
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