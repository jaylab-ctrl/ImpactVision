import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split as tts
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from statsmodels.stats.outliers_influence import variance_inflation_factor
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

# After removing outliers
plt.figure(figsize=(20, 10))
sns.boxplot(data=accidents_vehicles_casualties, orient='h')
plt.show()

# Heatmap of covariance matrix
plt.figure(figsize=(20, 10))
sns.heatmap(accidents_vehicles_casualties[cont_cols].cov(), annot=True, cmap='icefire')
plt.title('Covariance Matrix')
plt.show()

# Heatmap of correlation matrix
plt.figure(figsize=(20, 10))
sns.heatmap(accidents_vehicles_casualties[cont_cols].corr(), annot=True, cmap='icefire')
plt.title('Correlation Matrix')
plt.show()

# One-hot the categorical columns
accidents_vehicles_casualties = pd.get_dummies(accidents_vehicles_casualties, columns=cat_cols, drop_first=True)

# Scale the continuous columns
def scale(x):
    return (x - x.min()) / (x.max() - x.min())

accidents_vehicles_casualties[cont_cols] = accidents_vehicles_casualties[cont_cols].apply(scale, axis=0)


accidents_vehicles_casualties = accidents_vehicles_casualties.iloc[: : 2]

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
ax[0].set_title("Genres Before SMOTE")
ax[0].set_ylabel('')
y_smote.value_counts().plot.pie(explode = [0, 0.1, 0.2], autopct = "%1.1f%%", ax = ax[1],  colors = ['#66b3ff', '#ff9999', '#964B00'],
                                shadow = True)
ax[1].set_title("Genres After SMOTE")
ax[1].set_ylabel('')

print("Overall dataset values have been increased.")
print("Original size:\n", y.value_counts())
print("New size:\n", y_smote.value_counts())
plt.show()

X_smote = sm.add_constant(X_smote)

print('New size of dataset:', X_smote.shape, y_smote.shape)

# Split the data into train and test
X_train, X_test, y_train, y_test = tts(X_smote, y_smote, test_size=0.2, random_state=5805)

print('Train size:', X_train.shape, y_train.shape)
print('Test size:', X_test.shape, y_test.shape)

#Feature Selection

# Create a table that mentions the feature selection technique and the number of features selected and removed
feature_selection = pd.DataFrame(columns=['Feature Selection Technique', 'Number of Features Selected', 'Number of Features Removed'])

# 1. Perform PCA Analysis and Conditional Number
pca = PCA(svd_solver="full", n_components=0.9, random_state=5805)
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
plt.title("Cumulative Explained Variance vs Number of Components in PCA")
plt.grid(True)
plt.axvline(x=(np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9)[0][0] + 1, ),
            color="red", linestyle="--")
plt.axhline(y=0.9, color="black", linestyle="--")
plt.legend()
plt.show()

# Conditional Number of Original and Reduced data
print('Condition Number of Original data:', np.linalg.cond(X_smote))
print('Condition Number of Reduced data:', np.linalg.cond(X_pca))

feature_selection = feature_selection.append({'Feature Selection Technique': 'PCA',
                                                'Number of Features Selected': np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9)[0][0] + 1,
                                                'Number of Features Removed': X_smote.shape[1] - np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9)[0][0] + 1,},
                                                 ignore_index=True)

# 2. Perform Random Forest Analysis
rfc = RandomForestClassifier(n_estimators=100, random_state=5805, n_jobs=-1)
rfc.fit(X_train, y_train)

def plot_feature_importance_with_elimination(importance, names, model_type, threshold=0.01):
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=fi_df["feature_importance"], y=fi_df["feature_names"])
    plt.title(f"{model_type} Feature Importance")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Names")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    removed_features = fi_df[fi_df["feature_importance"] < threshold]["feature_names"]
    print("Removed Features:", removed_features.tolist())

    selected_features = fi_df[fi_df["feature_importance"] >= threshold]["feature_names"]
    print("Selected Features:", selected_features.tolist())

    return removed_features, selected_features


removed, selected = plot_feature_importance_with_elimination(
    rfc.feature_importances_, X_train.columns, "RANDOM FOREST")

feature_selection = feature_selection.append({'Feature Selection Technique': 'Random Forest',
                                                'Number of Features Selected': len(selected),
                                                'Number of Features Removed': len(removed)},
                                                 ignore_index=True)

# 3. Singular Value Decomposition Analysis
# First SVD with full components
svd_full = TruncatedSVD(n_components=X_smote.shape[1] - 1, random_state=5805, algorithm='arpack')
X_svd_full = svd_full.fit_transform(X_smote)

cumulative_variance_ratio = np.cumsum(svd_full.explained_variance_ratio_)

n_components_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1

# Now SVD with 90% variance explained
svd_90 = TruncatedSVD(n_components=n_components_90)
X_reduced_90 = svd_90.fit_transform(X_smote)

print("Original Shape: ", X_smote.shape)
print("Reduced Shape: ", X_reduced_90.shape)

plt.figure(figsize=(10, 10))
plt.plot(np.arange(1, len(cumulative_variance_ratio) + 1, 1), cumulative_variance_ratio,
         label="Cumulative Explained Variance", marker='o')
plt.xticks(np.arange(1, len(cumulative_variance_ratio) + 1, 1))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.xticks(np.arange(1, len(cumulative_variance_ratio) + 1, 10))
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.title("Cumulative Explained Variance vs Number of Components in SVD")
plt.axvline(x=n_components_90, color="red", linestyle="--")
plt.axhline(y=0.9, color='black', linestyle="--")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# Conditional Number of Original and Reduced data
print('Condition Number of Original data:', np.linalg.cond(X_smote))
print('Condition Number of Reduced data:', np.linalg.cond(X_reduced_90))

feature_selection = feature_selection.append({'Feature Selection Technique': 'Singular Value Decomposition Analysis',
                                                'Number of Features Selected': n_components_90,
                                                'Number of Features Removed': X_smote.shape[1] - n_components_90},
                                                 ignore_index=True)

# 4. Variance Inflation Factor Analysis
# Calculate VIF and count the number of features removed
vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(X_smote, i) for i in range(X_smote.shape[1])]
vif["Features"] = X_smote.columns
print(vif)

# Drop the columns with highest VIF > 10
vif_filtered = vif[vif['vif']<10]
print(vif_filtered)

vif_iter1 = X.shape[1] - vif_filtered.shape[0]
print("Number of features removed:", vif_iter1)

# Calculate VIF again and count the number of features removed
vif_second = pd.DataFrame()
vif_second["vif"] = [variance_inflation_factor(X_smote[vif_filtered['Features']], i) for i in range(X_smote[vif_filtered['Features']].shape[1])]
vif_second["Features"] = X_smote[vif_filtered['Features']].columns
print(vif_second)

# Drop the columns with highest VIF > 10
vif_filtered_second = vif_second[vif_second['vif']<10]
print(vif_filtered_second)

vif_iter2 = vif_filtered.shape[0] - vif_filtered_second.shape[0]
print("Number of features removed:", vif_iter2)

# Calculate VIF again and count the number of features removed
vif_third = pd.DataFrame()
vif_third["vif"] = [variance_inflation_factor(X_smote[vif_filtered_second['Features']], i) for i in range(X_smote[vif_filtered_second['Features']].shape[1])]
vif_third["Features"] = X_smote[vif_filtered_second['Features']].columns
print(vif_third)

# Drop the columns with highest VIF > 10
vif_filtered_third = vif_third[vif_third['vif']<10]
print(vif_filtered_third)

vif_iter3 = vif_filtered_second.shape[0] - vif_filtered_third.shape[0]
print("Number of features removed:", vif_iter3)

# Calculate total number of features removed
vif_total = vif_iter1 + vif_iter2 + vif_iter3
print("Total number of features removed:", vif_total)

feature_selection = feature_selection.append({'Feature Selection Technique': 'Variance Inflation Factor',
                                                'Number of Features Selected': vif_filtered_third.shape[0],
                                                'Number of Features Removed': vif_total},
                                                 ignore_index=True)

# Put all the feature selection techniques in a table
def create_feature_selecction_table(df, name):
    x = PrettyTable()
    x.title = f"{name} Comparison"
    x.field_names = df.columns

    for index, row in df.iterrows():
        x.add_row(row)

    print(x)

create_feature_selecction_table(feature_selection, 'Feature Selection/Dimensionality Reduction')

# Best Feature Selection Technique is Random Forest. So, we will use the features selected by Random Forest.












