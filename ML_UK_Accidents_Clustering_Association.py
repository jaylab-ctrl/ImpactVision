import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
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

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Feature Selection
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

X_train, X_test = X_train[selected], X_test[selected]

X = X[selected]

# Clustering

# K-Means Clustering
def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k, random_state=5805).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)

    return sse

# Convert the DataFrame to a NumPy array - replace 'column1' and 'column2' with your actual column names
points_array = X.to_numpy()

k = 10
sse = calculate_WSS(points_array, k)

# Plotting code remains the same
plt.figure()
plt.plot(np.arange(1, k+1, 1), sse)
plt.xticks(np.arange(1, k+1, 1))
plt.grid()
plt.xlabel('k')
plt.ylabel('WSS')
plt.title('k selection in k-mean Elbow Algorithm')
plt.show()


# Silhouette Score
sil = []
kmax = 10

for k in range(2, kmax+1):
    kmeans = KMeans(n_clusters=k, random_state=5805).fit(points_array)
    labels = kmeans.labels_
    sil.append(silhouette_score(points_array, labels, metric='euclidean'))
plt.figure()
plt.plot(np.arange(2, k + 1, 1), sil, 'bx-')
plt.xticks(np.arange(2,k+1,1))
plt.grid()
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.title('Silhouette Method')
plt.show()

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.4, min_samples=5)
dbscan.fit(points_array)

dbscan_y = dbscan.labels_

unique_labels = np.unique(dbscan_y)

plt.figure(figsize=(12, 8))

for label in unique_labels:
    class_member_mask = (dbscan_y == label)
    if label == -1:

        plt.plot(points_array[class_member_mask, 0], points_array[class_member_mask, 1], 'x', color='black',
                 label='Noise', markersize=10)
    else:

        color = plt.cm.Spectral(float(label) / len(unique_labels))
        plt.plot(points_array[class_member_mask, 0], points_array[class_member_mask, 1], 'o', color=color,
                 label=f'Cluster {label}', markersize=8)

plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Association Rules
# Create a basket such that that the basket should groupby on accident severity agg on number of casualties, speed limit and number of vehicles
basket = accidents_vehicles_casualties.groupby('Accident_Severity').agg({'Number_of_Casualties': 'sum', 'Speed_limit': 'sum', 'Number_of_Vehicles': 'sum'}).reset_index()
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

numeric_columns = ['Number_of_Casualties', 'Speed_limit', 'Number_of_Vehicles']
basket_sets = basket[numeric_columns].applymap(encode_units)

basket_sets['Accident_Severity'] = basket['Accident_Severity']
basket_sets.set_index('Accident_Severity', inplace=True)

# Apply the Apriori algorithm
freq_items = apriori(basket_sets, min_support=0.07, use_colnames=True)
rules = association_rules(freq_items, metric="lift", min_threshold=1)
print(rules.head(5).to_string())
