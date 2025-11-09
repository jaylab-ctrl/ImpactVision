import pandas as pd
from fuzzywuzzy import process
import warnings

warnings.filterwarnings("ignore")

# Read in the data
# Read accidents data
accidents = pd.read_csv('Accidents0514.csv', low_memory=False)
print(accidents.shape)
print(accidents.head())

# Read vehicles data
vehicles = pd.read_csv('Vehicles0514.csv', low_memory=False)
print(vehicles.shape)
print(vehicles.head())

# Read casualties data
casualties = pd.read_csv('Casualties0514.csv', low_memory=False)
print(casualties.shape)
print(casualties.head())

# Merge the data
accidents_vehicles = pd.merge(accidents, vehicles, on='Accident_Index')
print(accidents_vehicles.shape)
print(accidents_vehicles.head())

accidents_vehicles_casualties = pd.merge(accidents_vehicles, casualties, on='Accident_Index')

print('Final Dataset:')
print(accidents_vehicles_casualties.shape)
print(accidents_vehicles_casualties.head())
print(accidents_vehicles_casualties.columns)

accidents_vehicles_casualties_merger = accidents_vehicles_casualties.copy()

# Read excel file
excel = pd.read_excel("Road-Accident-Safety-Data-Guide.xlsx", sheet_name=None)


# Get the list of sheets in excel file and convert into list
def get_sheet_name(excel=excel):
    sheet_name = []
    for i in excel.keys():
        sheet_name.append(i)
    return sheet_name


excel_list = get_sheet_name()
excel_list = excel_list[2:]
print(excel_list)


# Get the list of column names in the dataset and convert into list
def get_column_name(df=accidents_vehicles_casualties_merger):
    column_name = []
    for i in df.columns:
        column_name.append(i)
    return column_name


column_list = get_column_name()
print(column_list)


# Create a function that will crate a table matching score of column names and sheet names
def get_matching_score_df(column_list=column_list, excel_list=excel_list, threshold=80):
    matching_score = []
    for name_to_find in column_list:
        match = process.extractOne(name_to_find, excel_list)
        if match[1] > threshold:
            row = {
                "Actual Column Name": name_to_find,
                "Actual Sheet Name": match[0],
                "Similarity Score": match[1],
            }
            matching_score.append(row)
            print(row)
    matching_table = pd.DataFrame(matching_score)
    return (
        matching_table,
        matching_table["Actual Column Name"].to_list(),
        matching_table["Actual Sheet Name"].to_list(),
    )


match, actual_columns, actual_excel = get_matching_score_df()
print(match)

# Create a function that will map the excel sheet to the dataset after converting to dictionary
def excel_to_dict_map(
    df=accidents_vehicles_casualties_merger,
    excel_sheet="Road-Accident-Safety-Data-Guide.xlsx",
    df_column_name=None,
    excel_column_name=None,
):
    for i in range(len(df_column_name)):
        subset_data = pd.read_excel(excel_sheet, sheet_name=excel_column_name[i])
        dict_data = subset_data.to_dict(orient="records")
        dict_code = {item["code"]: item["label"] for item in dict_data}
        df[df_column_name[i]] = df[df_column_name[i]].map(dict_code)
    return df

accidents_vehicles_casualties_merger = excel_to_dict_map(df_column_name=actual_columns,
                                              excel_column_name=actual_excel)

print(accidents_vehicles_casualties_merger.head())

# Save the data
# accidents_vehicles_casualties_merger.to_csv('UK_Accidents_Merger.csv', index=False)

