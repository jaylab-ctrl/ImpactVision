import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
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

# Feature Engineering
# Check for missing values
print("Missing Values:")
print(accidents_vehicles_casualties.isnull().sum()/accidents_vehicles_casualties.shape[0])

# Drop or impute missing values
accidents_vehicles_casualties.dropna(inplace=True)

# Check for duplicates
print("Duplicates:")
print(accidents_vehicles_casualties.duplicated().sum())

accidents_vehicles_casualties['Date'] = pd.to_datetime(accidents_vehicles_casualties['Date'])
accidents_vehicles_casualties['Year'] = accidents_vehicles_casualties['Date'].dt.year

# Drop unnecessary columns
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

# New Shape of the dataframe
print("New Shape of the dataframe: ", accidents_vehicles_casualties.shape)

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


# Define categorical and continuous columns
cat_cols = list(accidents_vehicles_casualties.select_dtypes(include=['object']).columns)
cat_cols.remove('Accident_Severity')

cont_cols = list(accidents_vehicles_casualties.select_dtypes(include=[np.number]).columns)

# Anomaly Detection
# Check for outliers
plt.figure(figsize=(20, 10))
sns.boxplot(data=accidents_vehicles_casualties, orient='h')
plt.show()

# Remove outliers
def iqr(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper

lower, upper = iqr(accidents_vehicles_casualties['Engine_Capacity_(CC)'])
accidents_vehicles_casualties = accidents_vehicles_casualties[(accidents_vehicles_casualties['Engine_Capacity_(CC)'] > lower) & (accidents_vehicles_casualties['Engine_Capacity_(CC)'] < upper)]

# One-hot the categorical columns
accidents_vehicles_casualties = pd.get_dummies(accidents_vehicles_casualties, columns=cat_cols, drop_first=True)

# Scale the continuous columns
def scale(x):
    return (x - x.min()) / (x.max() - x.min())

accidents_vehicles_casualties[cont_cols] = accidents_vehicles_casualties[cont_cols].apply(scale, axis=0)


accidents_vehicles_casualties = accidents_vehicles_casualties.iloc[: : 2]

accidents_vehicles_casualties['Accident_Severity'] = accidents_vehicles_casualties['Accident_Severity'].replace({
    'Fatal': 3,
    'Serious': 2,
    'Slight': 1
})

#Create a new column for the for the target varaible: fuel specific accident impact score
accidents_vehicles_casualties['Fuel_Specific_Accident_Impact_Score'] = accidents_vehicles_casualties['Accident_Severity'] * accidents_vehicles_casualties['Engine_Capacity_(CC)']
accidents_vehicles_casualties['Fuel_Specific_Accident_Impact_Score'] = accidents_vehicles_casualties['Fuel_Specific_Accident_Impact_Score'].astype(float)

accidents_vehicles_casualties.drop(['Accident_Severity', 'Engine_Capacity_(CC)'], axis=1, inplace=True)

print("Shape of the dataframe after feature engineering: ", accidents_vehicles_casualties.shape)

# Split the data into train and test
X = accidents_vehicles_casualties.drop(['Fuel_Specific_Accident_Impact_Score'], axis=1)
y = accidents_vehicles_casualties['Fuel_Specific_Accident_Impact_Score']

X = sm.add_constant(X)

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=5805)


#Display the AIC, BIC and Adjusted R2 as a predictive accuracy for each elimination inside the table
def backward_stepwise_with_summary_pval(X_train, y_train):
    table = pd.DataFrame(columns=["AIC", "BIC", "Adj. R2", "P-Value"])
    p_val_tab = pd.DataFrame(columns=["P-Value"])
    f_stat_tab = pd.DataFrame(columns=["F-Statistic"])
    adj_r2_tab = pd.DataFrame(columns=["Adj. R2"])
    removed_features = []
    counter = 0
    model = sm.OLS(y_train, X_train).fit()
    print(model.summary())

    while model.pvalues.max() >= 0.01:
        table.loc[model.pvalues.idxmax(), "AIC"] = model.aic
        table.loc[model.pvalues.idxmax(), "BIC"] = model.bic
        table.loc[model.pvalues.idxmax(), "Adj. R2"] = model.rsquared_adj
        table.loc[model.pvalues.idxmax(), "P-Value"] = model.pvalues.max()
        p_val_tab.loc[model.pvalues.idxmax(), "P-Value"] = model.pvalues.max()
        f_stat_tab.loc[model.pvalues.idxmax(), "F-Statistic"] = model.fvalue
        adj_r2_tab.loc[model.pvalues.idxmax(), "Adj. R2"] = model.rsquared_adj
        X_train.drop(model.pvalues.idxmax(), axis=1, inplace=True)
        X_test.drop(model.pvalues.idxmax(), axis=1, inplace=True)
        print("Removed Feature:", model.pvalues.idxmax())
        removed_features.append(model.pvalues.idxmax())
        counter += 1
        print("No. of Features Removed:", counter)
        model = sm.OLS(y_train, X_train).fit()
        print(model.summary())

    print("Removed Features:", removed_features)
    return table, p_val_tab, f_stat_tab, adj_r2_tab


table, t_stat, f_stat, adj_r2  = backward_stepwise_with_summary_pval(X_train, y_train)
print(table)
print(t_stat)
print(f_stat)
print(adj_r2)

# Fit the model with the selected features
lin_reg_final = sm.OLS(y_train, X_train).fit()
print(lin_reg_final.summary())

confidence_intervals = lin_reg_final.conf_int()
print("Confidence Intervals:\n", confidence_intervals)


def plot_backward_stepwise_summary(table):
    fig, ax = plt.subplots(1, 3, figsize=(30, 15))

    # Define the tick locations
    tick_locations = range(0, len(table.index), 15)

    # Plot AIC
    ax[0].plot(table["AIC"], marker="o")
    ax[0].set_title("AIC")
    ax[0].set_xlabel("Step")
    ax[0].set_xticks(tick_locations)
    ax[0].set_xticklabels([table.index[i] for i in tick_locations], rotation=45)

    # Plot BIC
    ax[1].plot(table["BIC"], marker="o")
    ax[1].set_title("BIC")
    ax[1].set_xlabel("Step")
    ax[1].set_xticks(tick_locations)
    ax[1].set_xticklabels([table.index[i] for i in tick_locations], rotation=45)

    # Plot Adjusted R2
    ax[2].plot(table["Adj. R2"], marker="o")
    ax[2].set_title("Adjusted R2")
    ax[2].set_xlabel("Step")
    ax[2].set_xticks(tick_locations)
    ax[2].set_xticklabels([table.index[i] for i in tick_locations], rotation=45)

    for axis in ax:
        axis.tick_params(axis='x', which='major', labelsize=8)
        axis.grid(True)

    plt.tight_layout()
    plt.show()


plot_backward_stepwise_summary(table)

# Prediciton of dependent variable
y_pred = lin_reg_final.predict(X_test)

# Plot the predicted values against the actual values (diffent colors) with line of best fit
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred, color='blue')
plt.plot(y_test, y_test, color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# Do predictions on the test set and evaluate the model
print(f"MSE:{mse(y_test, y_pred):.2f}")
print(f"MAE:{mae(y_test, y_pred):.2f}")
print(f"RMSE:{np.sqrt(mse(y_test, y_pred)): .2f}")
print(f"R2:{lin_reg_final.rsquared:.2f}")
print(f"Adjusted R2:{lin_reg_final.rsquared_adj:.2f}")








