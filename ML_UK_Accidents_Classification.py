import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from itertools import cycle
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, recall_score, roc_curve, precision_score, f1_score, auc
from sklearn.model_selection import StratifiedKFold
from scipy import mean
from prettytable import PrettyTable
import warnings
from sklearn.exceptions import ConvergenceWarning

pd.set_option('display.float_format', lambda x: '%.2f' % x)
sns.set_style('darkgrid')
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

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


accidents_vehicles_casualties = accidents_vehicles_casualties.iloc[: : 4]

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

# Encode target variable
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

X_train_knn = np.ascontiguousarray(X_train)
X_test_knn = np.ascontiguousarray(X_test)

print('Train size:', X_train.shape, y_train.shape)
print('Test size:', X_test.shape, y_test.shape)

# Model Building
def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        other_class = [x for x in unique_class if x != per_class]

        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict
def plot_multiclass_roc(y_test, y_pred, classes):
    y_test_binarized = label_binarize(y_test, classes=classes)
    y_pred_binarized = label_binarize(y_pred, classes=classes)

    n_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 10))

    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Extension of Receiver Operating Characteristic to Multi-class')
    plt.legend(loc="lower right")
    plt.show()

def specificity_calc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specificity_scores = {}

    for class_index in range(len(cm)):
        tn = np.sum(cm) - np.sum(cm[class_index, :]) - np.sum(cm[:, class_index]) + cm[class_index, class_index]
        fp = np.sum(cm[:, class_index]) - cm[class_index, class_index]

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_scores[class_index] = specificity

    for class_index in specificity_scores:
        print(f'Specificity of class {class_index}: {specificity_scores[class_index]:.2f}')

# Create a function tahat will create an ovo model and plot multiclass roc auc curve
def train_plot_multiclass_roc_auc_ovo(X_train, X_test, y_train, y_test, model):
    ovo = OneVsOneClassifier(model)
    ovo.fit(X_train, y_train)
    y_pred= ovo.predict(X_test)

    classes = list(set(y_train))
    y_test_binarized = label_binarize(y_test, classes=classes)
    y_pred_binarized = label_binarize(y_pred, classes=classes)

    n_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plotting
    plt.figure(figsize=(10, 10))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One-vs-One ROC AUC for Multi-class')
    plt.legend(loc="lower right")
    plt.show()

# Create a function tahat will create an ovr model and plot multiclass roc auc curve
def train_plot_multiclass_roc_auc_ovr(X_train, X_test, y_train, y_test, model):
    ovr = OneVsRestClassifier(model)
    ovr.fit(X_train, y_train)

    y_pred = ovr.predict(X_test)

    classes = list(set(y_train))
    y_test_binarized = label_binarize(y_test, classes=classes)
    y_pred_binarized = label_binarize(y_pred, classes=classes)

    # Compute ROC curve and ROC area for each class
    n_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plotting
    plt.figure(figsize=(10, 10))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One-vs-Rest ROC AUC for Multi-class')
    plt.legend(loc="lower right")
    plt.show()

def train_plot_multiclass_knn_roc_auc_ovo(X_train, X_test, y_train, y_test, model):
    X_train = np.ascontiguousarray(X_train)
    X_test = np.ascontiguousarray(X_test)

    ovo = OneVsOneClassifier(model)
    ovo.fit(X_train, y_train)

    y_pred = ovo.predict(X_test)

    classes = list(set(y_train))
    y_test_binarized = label_binarize(y_test, classes=classes)
    y_pred_binarized = label_binarize(y_pred, classes=classes)

    # Compute ROC curve and ROC area for each class
    n_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plotting
    plt.figure(figsize=(10, 10))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One-vs-One ROC AUC for Multi-class')
    plt.legend(loc="lower right")
    plt.show()

def train_plot_multiclass_knn_roc_auc_ovr(X_train, X_test, y_train, y_test, model):
    X_train = np.ascontiguousarray(X_train)
    X_test = np.ascontiguousarray(X_test)

    ovr = OneVsRestClassifier(model)
    ovr.fit(X_train, y_train)

    y_pred = ovr.predict(X_test)

    classes = list(set(y_train))
    y_test_binarized = label_binarize(y_test, classes=classes)
    y_pred_binarized = label_binarize(y_pred, classes=classes)

    n_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 10))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One-vs-Rest ROC AUC for Multi-class')
    plt.legend(loc="lower right")
    plt.show()

def kfold_cross_validation(model, X, y, k=5):
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=5805)

    scores = []
    fold = 0
    for train_index, test_index in kf.split(X, y):
        fold += 1
        X_train_kf, X_test_kf = X.iloc[train_index], X.iloc[test_index]
        y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train_kf, y_train_kf)
        y_pred_kf = model.predict(X_test_kf)

        accuracy_kf = accuracy_score(y_test_kf, y_pred_kf)
        scores.append(accuracy_kf)
        print(f'Fold {fold} accuracy: {accuracy_kf}')

    avg_scores = mean(scores)

    return avg_scores

def kfold_cross_validation_knn(model, X, y, k=5):
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=5805)

    scores = []
    fold = 0
    for train_index, test_index in kf.split(X, y):
        fold += 1
        X_train_kf, X_test_kf = X.iloc[train_index], X.iloc[test_index]
        y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]

        X_train_kf = np.ascontiguousarray(X_train_kf)
        X_test_kf = np.ascontiguousarray(X_test_kf)

        model.fit(X_train_kf, y_train_kf)
        y_pred_kf = model.predict(X_test_kf)

        accuracy_kf = accuracy_score(y_test_kf, y_pred_kf)
        scores.append(accuracy_kf)
        print(f'Fold {fold} accuracy: {accuracy_kf}')

    avg_scores = mean(scores)

    return avg_scores

def create_confusion_matrix_heatmap(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, cmap='Blues', annot=True, fmt='d', annot_kws={'size': 16})
    plt.title(f'Confusion Matrix of {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def find_best_k(X_train, y_train, X_test, y_test, k_range):
    error_rates = []
    accuracies = []

    for k in range(1, k_range + 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        error = 1 - accuracy_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        error_rates.append(error)
        accuracies.append(accuracy)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, k_range + 1), error_rates, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()

    best_k = accuracies.index(max(accuracies)) + 1
    return best_k, error_rates, accuracies


model_table = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Cross Validation Mean Score', 'AUC Score'])

# 1a. Baseline Decision Tree
dtc = DecisionTreeClassifier(random_state=5805)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

y_pred_proba = dtc.predict_proba(X_test)

print('Performance of the baseline decision tree on the test set:')
print('Accuracy of the model:', accuracy_score(y_test, y_pred).round(2))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Precision Score:', precision_score(y_test, y_pred, average='weighted').round(2))
print('Recall Score:', recall_score(y_test, y_pred, average='weighted').round(2))

specificity_calc(y_test, y_pred)


print('F1 Score:', f1_score(y_test, y_pred, average='weighted').round(2))

# Apply k-Fold Cross Validation based on accuracy
scores = kfold_cross_validation(dtc, X_smote, y_smote)

# Precision
macro_precision = precision_score(y_test, y_pred, average='macro')
micro_precision = precision_score(y_test, y_pred, average='micro')

# Recall
macro_recall = recall_score(y_test, y_pred, average='macro')
micro_recall = recall_score(y_test, y_pred, average='micro')

# F1 Score
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')

# ROC-AUC
roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred)

plot_multiclass_roc(y_test, y_pred, np.unique(y_test))

auc_mean = mean(list(roc_auc_dict.values()))
print(f'The mean of AUC for this multi-label classification is '
      f'{100*mean(list(roc_auc_dict.values())):.2f}%')

# One-vs-One
train_plot_multiclass_roc_auc_ovo(X_train, X_test, y_train, y_test, dtc)

# One-vs-Rest
train_plot_multiclass_roc_auc_ovr(X_train, X_test, y_train, y_test, dtc)

create_confusion_matrix_heatmap(y_test, y_pred, 'Baseline Decision Tree')


model_table = model_table.append({'Model': 'Baseline Decision Tree',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.round(2),
                                  'AUC Score': auc_mean.round(2)}, ignore_index=True)


# Tree Visualization
plt.figure(figsize=(10, 10))
plot_tree(dtc, filled=True, fontsize=10, rounded=True, feature_names=list(X_smote.columns), class_names=['0','1', '2'])
plt.title('Decision Tree with base parameters')
plt.show()

# 1b. Fine Tuned Decision Tree
tuned_parameters = [{'max_depth': range(1, 10),
                    'min_samples_split': range(2, 10),
                    'min_samples_leaf':range(1, 10),
                    'max_features':range(1, 5),
                    'splitter':['best','random'],
                    'criterion':['gini','entropy', 'log_loss']}]

model_dtc = DecisionTreeClassifier(random_state=5805)

clf = GridSearchCV(model_dtc, tuned_parameters, cv=5, scoring='accuracy', n_jobs=-1, verbose=False)
clf.fit(X_train, y_train)

print("Best Estimator: \n", clf.best_estimator_)
print("Best Score: \n", clf.best_score_)
print("Best Paramters: \n", clf.best_params_)

best_dtc = clf.best_estimator_
best_dtc.fit(X_train, y_train)

y_pred_grid = best_dtc.predict(X_test)

print('Performance of the fine tuned decision tree on the test set:')
print('Accuracy of the model:', accuracy_score(y_test, y_pred_grid).round(2))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_grid))
print('Precision Score:', precision_score(y_test, y_pred_grid, average='weighted').round(2))
print('Recall Score:', recall_score(y_test, y_pred_grid, average='weighted').round(2))

specificity_calc(y_test, y_pred_grid)

print('F1 Score:', f1_score(y_test, y_pred_grid, average='weighted').round(2))

# Apply k-Fold Cross Validation based on accuracy and find the average mean
scores = kfold_cross_validation(best_dtc, X_smote, y_smote)

# Precision
macro_precision = precision_score(y_test, y_pred_grid, average='macro')
micro_precision = precision_score(y_test, y_pred_grid, average='micro')

# Recall
macro_recall = recall_score(y_test, y_pred_grid, average='macro')
micro_recall = recall_score(y_test, y_pred_grid, average='micro')

# F1 Score
macro_f1 = f1_score(y_test, y_pred_grid, average='macro')
micro_f1 = f1_score(y_test, y_pred_grid, average='micro')

# ROC-AUC
roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred_grid)

plot_multiclass_roc(y_test, y_pred_grid, np.unique(y_test))

auc_mean = mean(list(roc_auc_dict.values()))
print(f'The mean of AUC for this multi-label classification is '
      f'{100*mean(list(roc_auc_dict.values())):.2f}%')

# One-vs-One
train_plot_multiclass_roc_auc_ovo(X_train, X_test, y_train, y_test, best_dtc)

# One-vs-Rest
train_plot_multiclass_roc_auc_ovr(X_train, X_test, y_train, y_test, best_dtc)

create_confusion_matrix_heatmap(y_test, y_pred_grid, 'Fine Tuned Decision Tree')

model_table = model_table.append({'Model': 'Fine Tuned Decision Tree',
                                'Accuracy': accuracy_score(y_test, y_pred_grid).round(2),
                                'Precision': precision_score(y_test, y_pred_grid, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred_grid, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred_grid, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.round(2),
                                  'AUC Score': auc_mean.round(2)}, ignore_index=True)

plt.figure(figsize=(10, 10))
plot_tree(best_dtc, filled=True, fontsize=10, rounded=True, feature_names=list(X_smote.columns), class_names=['0','1', '2'])
plt.title('Decision Tree with best parameters')
plt.show()

# 1c. Optimal Alpha and Pruned Decision Tree
path = best_dtc.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
print(ccp_alphas)
print(impurities)

accuracy_train, accuracy_test = [], []

for i in ccp_alphas:
    model = DecisionTreeClassifier(random_state=5805, ccp_alpha=i)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    accuracy_train.append(accuracy_score(y_train, y_train_pred))
    accuracy_test.append(accuracy_score(y_test, y_test_pred))

plt.figure(figsize=(5, 5))
plt.plot(ccp_alphas, accuracy_train, marker='o', label='train', drawstyle='steps-post')
plt.plot(ccp_alphas, accuracy_test, marker='o', label='test', drawstyle='steps-post')
plt.xlabel('ccp_alpha')
plt.ylabel('Accuracy')
plt.title('Accuracy vs ccp_alpha for train and test sets')
plt.legend()
plt.grid(True)
plt.show()

print('Optimal ccp_alpha:', ccp_alphas[accuracy_test.index(max(accuracy_test))])

model = DecisionTreeClassifier(random_state=5805, ccp_alpha=ccp_alphas[accuracy_test.index(max(accuracy_test))])
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print('Performance of the optimal ccp alpha decision tree on the test set:')
print('Accuracy of the model:', accuracy_score(y_test, y_test_pred).round(2))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_test_pred))
print('Precision Score:', precision_score(y_test, y_test_pred, average='weighted').round(2))
print('Recall Score:', recall_score(y_test, y_test_pred, average='weighted').round(2))

specificity_calc(y_test, y_test_pred)

print('F1 Score:', f1_score(y_test, y_test_pred, average='weighted').round(2))

# Apply k-Fold Cross Validation based on accuracy and find the average mean
scores = kfold_cross_validation(model, X_smote, y_smote)

# Precision
macro_precision = precision_score(y_test, y_test_pred, average='macro')
micro_precision = precision_score(y_test, y_test_pred, average='micro')

# Recall
macro_recall = recall_score(y_test, y_test_pred, average='macro')
micro_recall = recall_score(y_test, y_test_pred, average='micro')

# F1 Score
macro_f1 = f1_score(y_test, y_test_pred, average='macro')
micro_f1 = f1_score(y_test, y_test_pred, average='micro')

# ROC-AUC
roc_auc_dict = roc_auc_score_multiclass(y_test, y_test_pred)

plot_multiclass_roc(y_test, y_test_pred, np.unique(y_test))

auc_mean = mean(list(roc_auc_dict.values()))
print(f'The mean of AUC for this multi-label classification is '
      f'{100*mean(list(roc_auc_dict.values())):.2f}%')

# One-vs-One
train_plot_multiclass_roc_auc_ovo(X_train, X_test, y_train, y_test, model)

# One-vs-Rest
train_plot_multiclass_roc_auc_ovr(X_train, X_test, y_train, y_test, model)

create_confusion_matrix_heatmap(y_test, y_test_pred, 'Optimal Alpha: Post-Pruned Decision Tree')

model_table = model_table.append({'Model': 'Optimal Alpha: Post-Pruned Decision Tree',
                                'Accuracy': accuracy_score(y_test, y_test_pred).round(2),
                                'Precision': precision_score(y_test, y_test_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_test_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_test_pred, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.round(2),
                                  'AUC Score': auc_mean.round(2)}, ignore_index=True)

plt.figure(figsize=(10, 10))
plot_tree(model, filled=True, fontsize=10, rounded=True, feature_names=list(X_smote.columns), class_names=['0','1', '2'])
plt.title(f'Decision Tree with ccp_alpha = {ccp_alphas[accuracy_test.index(max(accuracy_test))]}')
plt.show()


# 2a. Baseline Logistic Regression
log_reg = LogisticRegression(random_state=5805)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

print('Performance of the baseline logistic regression on the test set:')
print('Accuracy of the model:', accuracy_score(y_test, y_pred).round(2))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Precision Score:', precision_score(y_test, y_pred, average='weighted').round(2))
print('Recall Score:', recall_score(y_test, y_pred, average='weighted').round(2))

specificity_calc(y_test, y_pred)

print('F1 Score:', f1_score(y_test, y_pred, average='weighted').round(2))

# Apply k-Fold Cross Validation based on accuracy and find the average mean
scores = kfold_cross_validation(log_reg, X_smote, y_smote)

# Precision
macro_precision = precision_score(y_test, y_pred, average='macro')
micro_precision = precision_score(y_test, y_pred, average='micro')

# Recall
macro_recall = recall_score(y_test, y_pred, average='macro')
micro_recall = recall_score(y_test, y_pred, average='micro')

# F1 Score
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')

# ROC-AUC
roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred)

plot_multiclass_roc(y_test, y_pred, np.unique(y_test))

auc_mean = mean(list(roc_auc_dict.values()))
print(f'The mean of AUC for this multi-label classification is '
      f'{100*mean(list(roc_auc_dict.values())):.2f}%')

# One-vs-One
train_plot_multiclass_roc_auc_ovo(X_train, X_test, y_train, y_test, log_reg)

# One-vs-Rest
train_plot_multiclass_roc_auc_ovr(X_train, X_test, y_train, y_test, log_reg)

create_confusion_matrix_heatmap(y_test, y_pred, 'Baseline Logistic Regression')

model_table = model_table.append({'Model': 'Baseline Logistic Regression',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.round(2),
                                  'AUC Score': auc_mean.round(2)}, ignore_index=True)

# 2b. Fine Tuned Logistic Regression
tuned_parameters = [{'penalty': ['l1', 'l2', 'None'],
                    'C': [0.1, 1, 10],
                    'solver': ['newton-cg', 'lbfgs'],
                    'max_iter': [2000, 3000]}]

model_log_reg = LogisticRegression(random_state=5805)

clf = GridSearchCV(model_log_reg, tuned_parameters, cv=5, scoring='accuracy', n_jobs=-1, verbose=False)
clf.fit(X_train, y_train)

print("Best Estimator: \n", clf.best_estimator_)
print("Best Score: \n", clf.best_score_)
print("Best Paramters: \n", clf.best_params_)

best_log_reg = clf.best_estimator_
best_log_reg.fit(X_train, y_train)

y_pred_grid = best_log_reg.predict(X_test)

print('Performance of the fine tuned logistic regression on the test set:')
print('Accuracy of the model:', accuracy_score(y_test, y_pred_grid).round(2))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_grid))
print('Precision Score:', precision_score(y_test, y_pred_grid, average='weighted').round(2))
print('Recall Score:', recall_score(y_test, y_pred_grid, average='weighted').round(2))

specificity_calc(y_test, y_pred_grid)

print('F1 Score:', f1_score(y_test, y_pred_grid, average='weighted').round(2))

# Apply k-Fold Cross Validation based on accuracy and find the average mean
scores = kfold_cross_validation(best_log_reg, X_smote, y_smote)

# Precision
macro_precision = precision_score(y_test, y_pred_grid, average='macro')
micro_precision = precision_score(y_test, y_pred_grid, average='micro')

# Recall
macro_recall = recall_score(y_test, y_pred_grid, average='macro')
micro_recall = recall_score(y_test, y_pred_grid, average='micro')

# F1 Score
macro_f1 = f1_score(y_test, y_pred_grid, average='macro')
micro_f1 = f1_score(y_test, y_pred_grid, average='micro')

# ROC-AUC
roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred_grid)

plot_multiclass_roc(y_test, y_pred_grid, np.unique(y_test))

auc_mean = mean(list(roc_auc_dict.values()))
print(f'The mean of AUC for this multi-label classification is '
      f'{100*mean(list(roc_auc_dict.values())):.2f}%')

# One-vs-One
train_plot_multiclass_roc_auc_ovo(X_train, X_test, y_train, y_test, best_log_reg)

# One-vs-Rest
train_plot_multiclass_roc_auc_ovr(X_train, X_test, y_train, y_test, best_log_reg)

create_confusion_matrix_heatmap(y_test, y_pred_grid, 'Fine Tuned Logistic Regression')

model_table = model_table.append({'Model': 'Fine Tuned Logistic Regression',
                                'Accuracy': accuracy_score(y_test, y_pred_grid).round(2),
                                'Precision': precision_score(y_test, y_pred_grid, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred_grid, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred_grid, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.round(2),
                                  'AUC Score': auc_mean.round(2)}, ignore_index=True)


# 3a. Baseline KNN
knn = KNeighborsClassifier()
knn.fit(X_train_knn, y_train)

y_pred = knn.predict(X_test_knn)

print('Performance of the baseline knn classifier on the test set:')
print('Accuracy of the model:', accuracy_score(y_test, y_pred).round(2))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Precision Score:', precision_score(y_test, y_pred, average='weighted').round(2))
print('Recall Score:', recall_score(y_test, y_pred, average='weighted').round(2))

specificity_calc(y_test, y_pred)

print('F1 Score:', f1_score(y_test, y_pred, average='weighted').round(2))

# Apply k-Fold Cross Validation based on accuracy and find the average mean
scores = kfold_cross_validation_knn(knn, X_smote, y_smote)

# Precision
macro_precision = precision_score(y_test, y_pred, average='macro')
micro_precision = precision_score(y_test, y_pred, average='micro')

# Recall
macro_recall = recall_score(y_test, y_pred, average='macro')
micro_recall = recall_score(y_test, y_pred, average='micro')

# F1 Score
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')

# ROC-AUC
roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred)

plot_multiclass_roc(y_test, y_pred, np.unique(y_test))

auc_mean = mean(list(roc_auc_dict.values()))
print(f'The mean of AUC for this multi-label classification is '
      f'{100*mean(list(roc_auc_dict.values())):.2f}%')

# One-vs-One
train_plot_multiclass_knn_roc_auc_ovo(X_train, X_test, y_train, y_test, knn)

# One-vs-Rest
train_plot_multiclass_knn_roc_auc_ovr(X_train, X_test, y_train, y_test, knn)

create_confusion_matrix_heatmap(y_test, y_pred, 'Baseline KNN')

model_table = model_table.append({'Model': 'Baseline KNN',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.round(2),
                                  'AUC Score': auc_mean.round(2)}, ignore_index=True)

# 3b. Fine Tuned KNN
knn = KNeighborsClassifier()
k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)

clf = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', return_train_score=True, verbose=1, n_jobs=-1)
clf.fit(X_train_knn, y_train)

print("Best Estimator: \n", clf.best_estimator_)
print("Best Score: \n", clf.best_score_)
print("Best Paramters: \n", clf.best_params_)

best_knn = clf.best_estimator_
best_knn.fit(X_train_knn, y_train)

y_pred_grid = best_knn.predict(X_test_knn)

print('Performance of the fine tuned knn classifier on the test set:')
print('Accuracy of the model:', accuracy_score(y_test, y_pred_grid).round(2))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_grid))
print('Precision Score:', precision_score(y_test, y_pred_grid, average='weighted').round(2))
print('Recall Score:', recall_score(y_test, y_pred_grid, average='weighted').round(2))

specificity_calc(y_test, y_pred_grid)

print('F1 Score:', f1_score(y_test, y_pred_grid, average='weighted').round(2))

# Apply k-Fold Cross Validation based on accuracy and find the average mean
scores = kfold_cross_validation_knn(best_knn, X_smote, y_smote)

# Precision
macro_precision = precision_score(y_test, y_pred_grid, average='macro')
micro_precision = precision_score(y_test, y_pred_grid, average='micro')

# Recall
macro_recall = recall_score(y_test, y_pred_grid, average='macro')
micro_recall = recall_score(y_test, y_pred_grid, average='micro')

# F1 Score
macro_f1 = f1_score(y_test, y_pred_grid, average='macro')
micro_f1 = f1_score(y_test, y_pred_grid, average='micro')

# ROC-AUC
roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred_grid)

plot_multiclass_roc(y_test, y_pred_grid, np.unique(y_test))

auc_mean = mean(list(roc_auc_dict.values()))
print(f'The mean of AUC for this multi-label classification is '
      f'{100*mean(list(roc_auc_dict.values())):.2f}%')

# One-vs-One
train_plot_multiclass_knn_roc_auc_ovo(X_train, X_test, y_train, y_test, best_knn)

# One-vs-Rest
train_plot_multiclass_knn_roc_auc_ovr(X_train, X_test, y_train, y_test, best_knn)

create_confusion_matrix_heatmap(y_test, y_pred_grid, 'Fine Tuned KNN')

# Selection of best k
best_k, error_rates, accuracies = find_best_k(X_train_knn, y_train, X_test_knn, y_test, 30)
print('Best k:', best_k)

model_table = model_table.append({'Model': 'Fine Tuned KNN',
                                'Accuracy': accuracy_score(y_test, y_pred_grid).round(2),
                                'Precision': precision_score(y_test, y_pred_grid, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred_grid, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred_grid, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.round(2),
                                  'AUC Score': auc_mean.round(2)}, ignore_index=True)


# 4a. Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print('Performance of the baseline gaussian naive bayes on the test set:')
print('Accuracy of the model:', accuracy_score(y_test, y_pred).round(2))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Precision Score:', precision_score(y_test, y_pred, average='weighted').round(2))
print('Recall Score:', recall_score(y_test, y_pred, average='weighted').round(2))

specificity_calc(y_test, y_pred)

print('F1 Score:', f1_score(y_test, y_pred, average='weighted').round(2))

# Apply k-Fold Cross Validation based on accuracy and find the average mean
scores = kfold_cross_validation(gnb, X_smote, y_smote)

# Precision
macro_precision = precision_score(y_test, y_pred, average='macro')
micro_precision = precision_score(y_test, y_pred, average='micro')

# Recall
macro_recall = recall_score(y_test, y_pred, average='macro')
micro_recall = recall_score(y_test, y_pred, average='micro')

# F1 Score
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')

# ROC-AUC
roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred)

plot_multiclass_roc(y_test, y_pred, np.unique(y_test))

auc_mean = mean(list(roc_auc_dict.values()))
print(f'The mean of AUC for this multi-label classification is '
      f'{100*mean(list(roc_auc_dict.values())):.2f}%')

# One-vs-One
train_plot_multiclass_roc_auc_ovo(X_train, X_test, y_train, y_test, gnb)

# One-vs-Rest
train_plot_multiclass_roc_auc_ovr(X_train, X_test, y_train, y_test, gnb)

create_confusion_matrix_heatmap(y_test, y_pred, 'Baseline Naive Bayes')


model_table = model_table.append({'Model': 'Baseline Naive Bayes',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.round(2),
                                  'AUC Score': auc_mean.round(2)}, ignore_index=True)

# 4b. Fine Tuned Naive Bayes
gnb = GaussianNB()

tuned_parameters = [{'var_smoothing': [1e-2, 1e-3]}]

clf = GridSearchCV(gnb, tuned_parameters, cv=5, scoring='accuracy', return_train_score=True, verbose=1, n_jobs=-1)
clf.fit(X_train, y_train)

print("Best Estimator: \n", clf.best_estimator_)
print("Best Score: \n", clf.best_score_)
print("Best Paramters: \n", clf.best_params_)

best_gnb = clf.best_estimator_
best_gnb.fit(X_train, y_train)

y_pred_grid = best_gnb.predict(X_test)

print('Performance of the fine tuned gaussian naive bayes on the test set:')
print('Accuracy of the model:', accuracy_score(y_test, y_pred_grid).round(2))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_grid))
print('Precision Score:', precision_score(y_test, y_pred_grid, average='weighted').round(2))
print('Recall Score:', recall_score(y_test, y_pred_grid, average='weighted').round(2))

specificity_calc(y_test, y_pred_grid)

print('F1 Score:', f1_score(y_test, y_pred_grid, average='weighted').round(2))

# Apply k-Fold Cross Validation based on accuracy and find the average mean
scores = kfold_cross_validation(best_gnb, X_smote, y_smote)

# Precision
macro_precision = precision_score(y_test, y_pred_grid, average='macro')
micro_precision = precision_score(y_test, y_pred_grid, average='micro')

# Recall
macro_recall = recall_score(y_test, y_pred_grid, average='macro')
micro_recall = recall_score(y_test, y_pred_grid, average='micro')

# F1 Score
macro_f1 = f1_score(y_test, y_pred_grid, average='macro')
micro_f1 = f1_score(y_test, y_pred_grid, average='micro')

# ROC-AUC
roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred_grid)

plot_multiclass_roc(y_test, y_pred_grid, np.unique(y_test))

auc_mean = mean(list(roc_auc_dict.values()))
print(f'The mean of AUC for this multi-label classification is '
      f'{100*mean(list(roc_auc_dict.values())):.2f}%')

# One-vs-One
train_plot_multiclass_roc_auc_ovo(X_train, X_test, y_train, y_test, best_gnb)

# One-vs-Rest
train_plot_multiclass_roc_auc_ovr(X_train, X_test, y_train, y_test, best_gnb)

create_confusion_matrix_heatmap(y_test, y_pred_grid, 'Fine Tuned Naive Bayes')

model_table = model_table.append({'Model': 'Fine Tuned Naive Bayes',
                                'Accuracy': accuracy_score(y_test, y_pred_grid).round(2),
                                'Precision': precision_score(y_test, y_pred_grid, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred_grid, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred_grid, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.round(2),
                                  'AUC Score': auc_mean.round(2)}, ignore_index=True)


# 5a. Baseline SVM
svc = SVC(random_state=5805)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

print('Performance of the baseline svm on the test set:')
print('Accuracy of the model:', accuracy_score(y_test, y_pred).round(2))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Precision Score:', precision_score(y_test, y_pred, average='weighted').round(2))
print('Recall Score:', recall_score(y_test, y_pred, average='weighted').round(2))

specificity_calc(y_test, y_pred)

print('F1 Score:', f1_score(y_test, y_pred, average='weighted').round(2))

# Apply k-Fold Cross Validation based on accuracy and find the average mean
scores = kfold_cross_validation(svc, X_smote, y_smote)

# Precision
macro_precision = precision_score(y_test, y_pred, average='macro')
micro_precision = precision_score(y_test, y_pred, average='micro')

# Recall
macro_recall = recall_score(y_test, y_pred, average='macro')
micro_recall = recall_score(y_test, y_pred, average='micro')

# F1 Score
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')

# ROC-AUC
roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred)

plot_multiclass_roc(y_test, y_pred, np.unique(y_test))

auc_mean = mean(list(roc_auc_dict.values()))
print(f'The mean of AUC for this multi-label classification is '
      f'{100*mean(list(roc_auc_dict.values())):.2f}%')

# One-vs-One
train_plot_multiclass_roc_auc_ovo(X_train, X_test, y_train, y_test, svc)

# One-vs-Rest
train_plot_multiclass_roc_auc_ovr(X_train, X_test, y_train, y_test, svc)

create_confusion_matrix_heatmap(y_test, y_pred, 'Baseline SVM')

model_table = model_table.append({'Model': 'Baseline SVM',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.round(2),
                                  'AUC Score': auc_mean.round(2)}, ignore_index=True)

# 5b. Fine Tuned SVM
svc = SVC(random_state=5805)

tuned_parameters = [{'kernel': ['rbf', 'poly'],
                    'gamma': [0.1, 1],
                    'C': [0.1, 1]}]

clf = GridSearchCV(svc, tuned_parameters, cv=5, scoring='accuracy', n_jobs=-1, verbose=False)
clf.fit(X_train, y_train)

print("Best Estimator: \n", clf.best_estimator_)
print("Best Score: \n", clf.best_score_)
print("Best Paramters: \n", clf.best_params_)

best_svc = clf.best_estimator_
best_svc.fit(X_train, y_train)

y_pred_grid = best_svc.predict(X_test)

print('Performance of the fine tuned svm on the test set:')
print('Accuracy of the model:', accuracy_score(y_test, y_pred_grid).round(2))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_grid))
print('Precision Score:', precision_score(y_test, y_pred_grid, average='weighted').round(2))
print('Recall Score:', recall_score(y_test, y_pred_grid, average='weighted').round(2))

specificity_calc(y_test, y_pred_grid)

print('F1 Score:', f1_score(y_test, y_pred_grid, average='weighted').round(2))

# Apply k-Fold Cross Validation based on accuracy and find the average mean
scores = kfold_cross_validation(best_svc, X_smote, y_smote)


# Precision
macro_precision = precision_score(y_test, y_pred_grid, average='macro')
micro_precision = precision_score(y_test, y_pred_grid, average='micro')

# Recall
macro_recall = recall_score(y_test, y_pred_grid, average='macro')
micro_recall = recall_score(y_test, y_pred_grid, average='micro')

# F1 Score
macro_f1 = f1_score(y_test, y_pred_grid, average='macro')
micro_f1 = f1_score(y_test, y_pred_grid, average='micro')

# ROC-AUC
roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred_grid)

plot_multiclass_roc(y_test, y_pred_grid, np.unique(y_test))

auc_mean = mean(list(roc_auc_dict.values()))
print(f'The mean of AUC for this multi-label classification is '
      f'{100*mean(list(roc_auc_dict.values())):.2f}%')

# One-vs-One
train_plot_multiclass_roc_auc_ovo(X_train, X_test, y_train, y_test, best_svc)

# One-vs-Rest
train_plot_multiclass_roc_auc_ovr(X_train, X_test, y_train, y_test, best_svc)

create_confusion_matrix_heatmap(y_test, y_pred_grid, 'Fine Tuned SVM')

model_table = model_table.append({'Model': 'Fine Tuned SVM',
                                'Accuracy': accuracy_score(y_test, y_pred_grid).round(2),
                                'Precision': precision_score(y_test, y_pred_grid, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred_grid, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred_grid, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.round(2),
                                  'AUC Score': auc_mean.round(2)}, ignore_index=True)


# 6a. Multi Layer Perceptron
mlp = MLPClassifier(random_state=5805)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

print('Performance of the baseline multi layer perceptron on the test set:')
print('Accuracy of the model:', accuracy_score(y_test, y_pred).round(2))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Precision Score:', precision_score(y_test, y_pred, average='weighted').round(2))
print('Recall Score:', recall_score(y_test, y_pred, average='weighted').round(2))

specificity_calc(y_test, y_pred)

print('F1 Score:', f1_score(y_test, y_pred, average='weighted').round(2))

# Apply k-Fold Cross Validation based on accuracy and find the average mean
scores = kfold_cross_validation(mlp, X_smote, y_smote)

# Precision
macro_precision = precision_score(y_test, y_pred, average='macro')
micro_precision = precision_score(y_test, y_pred, average='micro')

# Recall
macro_recall = recall_score(y_test, y_pred, average='macro')
micro_recall = recall_score(y_test, y_pred, average='micro')

# F1 Score
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')

# ROC-AUC
roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred)

plot_multiclass_roc(y_test, y_pred, np.unique(y_test))

auc_mean = mean(list(roc_auc_dict.values()))
print(f'The mean of AUC for this multi-label classification is '
      f'{100*mean(list(roc_auc_dict.values())):.2f}%')

# One-vs-One
train_plot_multiclass_roc_auc_ovo(X_train, X_test, y_train, y_test, mlp)

# One-vs-Rest
train_plot_multiclass_roc_auc_ovr(X_train, X_test, y_train, y_test, mlp)

create_confusion_matrix_heatmap(y_test, y_pred, 'Baseline Multi Layer Perceptron')


model_table = model_table.append({'Model': 'Baseline Multi Layer Perceptron',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.round(2),
                                  'AUC Score': auc_mean.round(2)}, ignore_index=True)


# 6b. Fine Tuned Multi Layer Perceptron
mlp = MLPClassifier(random_state=5805)

tuned_parameters = [{'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                    'alpha': [0.0001, 0.05],
                    'learning_rate': ['constant','adaptive']}]

clf = GridSearchCV(mlp, tuned_parameters, cv=5, scoring='accuracy', n_jobs=-1, verbose=False)
clf.fit(X_train, y_train)

print("Best Estimator: \n", clf.best_estimator_)
print("Best Score: \n", clf.best_score_)
print("Best Paramters: \n", clf.best_params_)

best_mlp = clf.best_estimator_
best_mlp.fit(X_train, y_train)

y_pred_grid = best_mlp.predict(X_test)

print('Performance of the fine tuned multi layer perceptron on the test set:')
print('Accuracy of the model:', accuracy_score(y_test, y_pred_grid).round(2))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_grid))
print('Precision Score:', precision_score(y_test, y_pred_grid, average='weighted').round(2))
print('Recall Score:', recall_score(y_test, y_pred_grid, average='weighted').round(2))

specificity_calc(y_test, y_pred_grid)

print('F1 Score:', f1_score(y_test, y_pred_grid, average='weighted').round(2))

# Apply k-Fold Cross Validation based on accuracy and find the average mean
scores = kfold_cross_validation(best_mlp, X_smote, y_smote)

# Precision
macro_precision = precision_score(y_test, y_pred_grid, average='macro')
micro_precision = precision_score(y_test, y_pred_grid, average='micro')

# Recall
macro_recall = recall_score(y_test, y_pred_grid, average='macro')
micro_recall = recall_score(y_test, y_pred_grid, average='micro')

# F1 Score
macro_f1 = f1_score(y_test, y_pred_grid, average='macro')
micro_f1 = f1_score(y_test, y_pred_grid, average='micro')

# ROC-AUC
roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred_grid)

plot_multiclass_roc(y_test, y_pred_grid, np.unique(y_test))

auc_mean = mean(list(roc_auc_dict.values()))
print(f'The mean of AUC for this multi-label classification is '
      f'{100*mean(list(roc_auc_dict.values())):.2f}%')

# One-vs-One
train_plot_multiclass_roc_auc_ovo(X_train, X_test, y_train, y_test, best_mlp)

# One-vs-Rest
train_plot_multiclass_roc_auc_ovr(X_train, X_test, y_train, y_test, best_mlp)

create_confusion_matrix_heatmap(y_test, y_pred_grid, 'Fine Tuned Multi Layer Perceptron')


model_table = model_table.append({'Model': 'Fine Tuned Multi Layer Perceptron',
                                'Accuracy': accuracy_score(y_test, y_pred_grid).round(2),
                                'Precision': precision_score(y_test, y_pred_grid, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred_grid, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred_grid, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.round(2),
                                  'AUC Score': auc_mean.round(2)}, ignore_index=True)


# 7a. Baseline Random Forest
rfc = RandomForestClassifier(random_state=5805)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

print('Performance of the baseline random forest classifier on the test set:')
print('Accuracy of the model:', accuracy_score(y_test, y_pred).round(2))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Precision Score:', precision_score(y_test, y_pred, average='weighted').round(2))
print('Recall Score:', recall_score(y_test, y_pred, average='weighted').round(2))

specificity_calc(y_test, y_pred)

print('F1 Score:', f1_score(y_test, y_pred, average='weighted').round(2))

# Apply k-Fold Cross Validation based on accuracy and find the average mean
scores = kfold_cross_validation(rfc, X_smote, y_smote)

# Precision
macro_precision = precision_score(y_test, y_pred, average='macro')
micro_precision = precision_score(y_test, y_pred, average='micro')

# Recall
macro_recall = recall_score(y_test, y_pred, average='macro')
micro_recall = recall_score(y_test, y_pred, average='micro')

# F1 Score
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')

# ROC-AUC
roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred)

plot_multiclass_roc(y_test, y_pred, np.unique(y_test))

auc_mean = mean(list(roc_auc_dict.values()))
print(f'The mean of AUC for this multi-label classification is '
      f'{100*mean(list(roc_auc_dict.values())):.2f}%')

# One-vs-One
train_plot_multiclass_roc_auc_ovo(X_train, X_test, y_train, y_test, rfc)

# One-vs-Rest
train_plot_multiclass_roc_auc_ovr(X_train, X_test, y_train, y_test, rfc)

create_confusion_matrix_heatmap(y_test, y_pred, 'Baseline Random Forest')


model_table = model_table.append({'Model': 'Baseline Random Forest',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.round(2),
                                  'AUC Score': auc_mean.round(2)}, ignore_index=True)

# 7b. Random Forest with Bagging
rfc = RandomForestClassifier(random_state=5805, n_jobs=-1)

bag = BaggingClassifier(rfc, n_estimators=100, random_state=5805, n_jobs=-1)
bag.fit(X_train, y_train)

y_pred = bag.predict(X_test)

print('Performance of the random forest classifier with bagging on the test set:')
print('Accuracy of the model:', accuracy_score(y_test, y_pred).round(2))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Precision Score:', precision_score(y_test, y_pred, average='weighted').round(2))
print('Recall Score:', recall_score(y_test, y_pred, average='weighted').round(2))

specificity_calc(y_test, y_pred)

print('F1 Score:', f1_score(y_test, y_pred, average='weighted').round(2))

# Apply k-Fold Cross Validation based on accuracy and find the average mean
scores = kfold_cross_validation(bag, X_smote, y_smote)

# Precision
macro_precision = precision_score(y_test, y_pred, average='macro')
micro_precision = precision_score(y_test, y_pred, average='micro')

# Recall
macro_recall = recall_score(y_test, y_pred, average='macro')
micro_recall = recall_score(y_test, y_pred, average='micro')

# F1 Score
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')

# ROC-AUC
roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred)

plot_multiclass_roc(y_test, y_pred, np.unique(y_test))

auc_mean = mean(list(roc_auc_dict.values()))
print(f'The mean of AUC for this multi-label classification is '
      f'{100*mean(list(roc_auc_dict.values())):.2f}%')

# One-vs-One
train_plot_multiclass_roc_auc_ovo(X_train, X_test, y_train, y_test, bag)

# One-vs-Rest
train_plot_multiclass_roc_auc_ovr(X_train, X_test, y_train, y_test, bag)

create_confusion_matrix_heatmap(y_test, y_pred, 'Random Forest with Bagging')

model_table = model_table.append({'Model': 'Random Forest with Bagging',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.round(2),
                                  'AUC Score': auc_mean.round(2)}, ignore_index=True)


# 7c. Random Forest with Stacking
rfc = RandomForestClassifier(random_state=5805, n_jobs=-1)

final_estimator = RandomForestClassifier(random_state=5805, n_jobs=-1)

stack = StackingClassifier(estimators=[('rfc', rfc)], final_estimator=final_estimator, cv=5)
stack.fit(X_train, y_train)

y_pred = stack.predict(X_test)

print('Performance of the random forest classifier with stacking on the test set:')
print('Accuracy of the model:', accuracy_score(y_test, y_pred).round(2))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Precision Score:', precision_score(y_test, y_pred, average='weighted').round(2))
print('Recall Score:', recall_score(y_test, y_pred, average='weighted').round(2))

specificity_calc(y_test, y_pred)

print('F1 Score:', f1_score(y_test, y_pred, average='weighted').round(2))

# Apply k-Fold Cross Validation based on accuracy and find the average mean
scores = kfold_cross_validation(stack, X_smote, y_smote)

# Precision
macro_precision = precision_score(y_test, y_pred, average='macro')
micro_precision = precision_score(y_test, y_pred, average='micro')

# Recall
macro_recall = recall_score(y_test, y_pred, average='macro')
micro_recall = recall_score(y_test, y_pred, average='micro')

# F1 Score
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')

# ROC-AUC
roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred)

plot_multiclass_roc(y_test, y_pred, np.unique(y_test))

auc_mean = mean(list(roc_auc_dict.values()))
print(f'The mean of AUC for this multi-label classification is '
      f'{100*mean(list(roc_auc_dict.values())):.2f}%')

# One-vs-One
train_plot_multiclass_roc_auc_ovo(X_train, X_test, y_train, y_test, stack)

# One-vs-Rest
train_plot_multiclass_roc_auc_ovr(X_train, X_test, y_train, y_test, stack)

create_confusion_matrix_heatmap(y_test, y_pred, 'Random Forest with Stacking')


model_table = model_table.append({'Model': 'Random Forest with Stacking',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.round(2),
                                  'AUC Score': auc_mean.round(2)}, ignore_index=True)

# 7d. Random Forest with Boosting
rfc = RandomForestClassifier(random_state=5805, n_jobs=-1)

boost = AdaBoostClassifier(rfc, n_estimators=100, random_state=5805)
boost.fit(X_train, y_train)

y_pred = boost.predict(X_test)

print('Performance of the random forest classifier with boosting on the test set:')
print('Accuracy of the model:', accuracy_score(y_test, y_pred).round(2))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Precision Score:', precision_score(y_test, y_pred, average='weighted').round(2))
print('Recall Score:', recall_score(y_test, y_pred, average='weighted').round(2))

specificity_calc(y_test, y_pred)

print('F1 Score:', f1_score(y_test, y_pred, average='weighted').round(2))

# Apply k-Fold Cross Validation based on accuracy and find the average mean
scores = kfold_cross_validation(boost, X_smote, y_smote)

# Precision
macro_precision = precision_score(y_test, y_pred, average='macro')
micro_precision = precision_score(y_test, y_pred, average='micro')

# Recall
macro_recall = recall_score(y_test, y_pred, average='macro')
micro_recall = recall_score(y_test, y_pred, average='micro')

# F1 Score
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')

# ROC-AUC
roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred)

plot_multiclass_roc(y_test, y_pred, np.unique(y_test))

auc_mean = mean(list(roc_auc_dict.values()))
print(f'The mean of AUC for this multi-label classification is '
      f'{100*mean(list(roc_auc_dict.values())):.2f}%')

# One-vs-One
train_plot_multiclass_roc_auc_ovo(X_train, X_test, y_train, y_test, boost)

# One-vs-Rest
train_plot_multiclass_roc_auc_ovr(X_train, X_test, y_train, y_test, boost)

create_confusion_matrix_heatmap(y_test, y_pred, 'Random Forest with Boosting')


model_table = model_table.append({'Model': 'Random Forest with Boosting',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.round(2),
                                  'AUC Score': auc_mean.round(2)}, ignore_index=True)

# 7e. Random Forest Grid Search
rfc = RandomForestClassifier(random_state=5805, n_jobs=-1)

tuned_parameters = [{'n_estimators': [100, 200],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]}]

clf = GridSearchCV(rfc, tuned_parameters, cv=5, scoring='accuracy', n_jobs=-1, verbose=False)
clf.fit(X_train, y_train)

print("Best Estimator: \n", clf.best_estimator_)

best_rfc = clf.best_estimator_

y_pred = best_rfc.predict(X_test)

print('Performance of the random forest classifier with grid search on the test set:')
print('Accuracy of the model:', accuracy_score(y_test, y_pred).round(2))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Precision Score:', precision_score(y_test, y_pred, average='weighted').round(2))
print('Recall Score:', recall_score(y_test, y_pred, average='weighted').round(2))

specificity_calc(y_test, y_pred)

print('F1 Score:', f1_score(y_test, y_pred, average='weighted').round(2))

# Apply k-Fold Cross Validation based on accuracy and find the average mean
scores = kfold_cross_validation(best_rfc, X_smote, y_smote)

# Precision
macro_precision = precision_score(y_test, y_pred, average='macro')
micro_precision = precision_score(y_test, y_pred, average='micro')

# Recall
macro_recall = recall_score(y_test, y_pred, average='macro')
micro_recall = recall_score(y_test, y_pred, average='micro')

# F1 Score
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')

# ROC-AUC
roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred)

plot_multiclass_roc(y_test, y_pred, np.unique(y_test))

auc_mean = mean(list(roc_auc_dict.values()))
print(f'The mean of AUC for this multi-label classification is '
      f'{100*mean(list(roc_auc_dict.values())):.2f}%')

# One-vs-One
train_plot_multiclass_roc_auc_ovo(X_train, X_test, y_train, y_test, best_rfc)

# One-vs-Rest
train_plot_multiclass_roc_auc_ovr(X_train, X_test, y_train, y_test, best_rfc)

create_confusion_matrix_heatmap(y_test, y_pred, 'Random Forest with Grid Search')

model_table = model_table.append({'Model': 'Random Forest with Grid Search',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                 'Cross Validation Mean Score': scores.round(2),
                                  'AUC Score': auc_mean.round(2)}, ignore_index=True)


def plot_model_comparison_table(table, name):
    table = table.sort_values(by='Accuracy', ascending=False)
    table = table.round(2)
    table = table.reset_index(drop=True)

    x = PrettyTable()
    x.title = f"{name} Comparison"
    x.field_names = table.columns

    for index, row in table.iterrows():
        x.add_row(row)

    print(x)

plot_model_comparison_table(model_table, 'Model')