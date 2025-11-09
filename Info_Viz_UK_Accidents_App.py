import pandas as pd
import time
import plotly.express as px
from dash import Dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate
import warnings

pd.set_option('display.float_format', lambda x: '%.2f' % x)
load_figure_template('CYBORG')
warnings.filterwarnings("ignore")

# Read in the data
start_time = time.time()
accidents_vehicles_casualties = pd.DataFrame()

for chunk in pd.read_csv('UK_Accidents_Merger.csv', chunksize=200000, low_memory=False):
    print('Number of chunks read: ', chunk.shape)
    accidents_vehicles_casualties = pd.concat([accidents_vehicles_casualties, chunk])

print("Time taken to read the data: ", time.time() - start_time)

# Drop or impute missing values
accidents_vehicles_casualties.dropna(inplace=True)

# Viusalizations using Plotly

# Convert date to year
accidents_vehicles_casualties['Date'] = pd.to_datetime(accidents_vehicles_casualties['Date'])
accidents_vehicles_casualties['Year'] = accidents_vehicles_casualties['Date'].dt.year

# Remove dataset features
accidents_vehicles_casualties.drop(['Location_Easting_OSGR', 'Location_Northing_OSGR','Local_Authority_(District)',
                                    'Local_Authority_(Highway)', '1st_Road_Number', '2nd_Road_Number',
                                    'Pedestrian_Crossing-Human_Control',
                                    'Pedestrian_Crossing-Physical_Facilities', 'Vehicle_Reference_x', 'Vehicle_Reference_y',
                                    'Pedestrian_Location', 'Pedestrian_Movement', 'Pedestrian_Road_Maintenance_Worker',
                                    'Casualty_Home_Area_Type', 'Age_Band_of_Driver', 'Age_Band_of_Casualty'], axis=1, inplace=True)

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

# Create a dash app
uk_accidents_app = Dash("UK Accident Dashboard", external_stylesheets=[dbc.themes.CYBORG])

# Layout of the app
uk_accidents_app.layout = html.Div(style={'padding': '10px 10px 10px 10px'},
    children=[

    html.H1("UK Accidents Dashboard", style={'textAlign': 'center', 'color': '#503D36'}),

    html.Div("Explore the relationships between UK Accidents, Casualties, and Vehicles...", style={'textAlign': 'center', 'color': '#F57241'}),

    html.Br(),

    # Put tabs in left side not in middle
    dcc.Tabs(id='Tabs', value='about-tool', children=[
        dcc.Tab(label='About Project', id='About tool', value='about-tool'),
        dcc.Tab(label='Subplots', id='Subplot tool', value='subplots-tool'),
        dcc.Tab(label='Individual Line Plots', id='Scatter tool', value='line-plot-tool'),
        dcc.Tab(label='Individual Bar Plots', id='Bar tool', value='bar-plot-tool'),
        dcc.Tab(label='Individual Pie Plots', id='Pie tool', value='pie-plot-tool'),
        dcc.Tab(label='Individual Count Plots', id='Count tool', value='count-plot-tool'),
        dcc.Tab(label='Individual Histograms', id='Histogram tool', value='histogram-tool'),
        dcc.Tab(label="Other Miscellaneous Plots", id="Misc tool", value="misc-plot-tool"),
        dcc.Tab(label='Geospatial Plots', id='Geospatial tool', value='geospatial-tool')
        ],
    colors={
        "border": "black",
        "primary": "gold",
        "background": "black"
    },
    ),
    html.Div(id='tabs-content')
    ]
)

@uk_accidents_app.callback(
    Output(component_id='tabs-content', component_property='children'),
    [Input(component_id='Tabs', component_property='value')])

def render_content(tab):

    if tab == 'about-tool':
        return tab_about

    elif tab == 'subplots-tool':
        return sub_plots

    elif tab == 'line-plot-tool':
        return line_plots

    elif tab == 'bar-plot-tool':
        return bar_plots

    elif tab == 'pie-plot-tool':
        return pie_plots

    elif tab == 'count-plot-tool':
        return count_plots

    elif tab == 'histogram-tool':
        return histogram_plots

    elif tab == 'misc-plot-tool':
        return misc_plots

    elif tab == 'geospatial-tool':
        return geo_plots

#--------------------------------
# About Project
#--------------------------------

# Create about project tab
tab_about = html.Div(
    children=[

    html.H4('About Project', style={'textAlign': 'center', 'color': '#503D36', 'font-size': '40px'}),

    html.Img(src='https://t3.ftcdn.net/jpg/03/72/46/46/360_F_372464646_Ks082AREONEjY5XYhWSexdDGFQ9tHr8S.jpg', style={'width': '70%',
                                                                                                                     'display': 'block',
                                                                                                                     'margin-left': 'auto',
                                                                                                                     'margin-right': 'auto',
                                                                                                                     'padding': '10px'}),
    html.P(
        'This project is about exploring the relationships between UK Accidents, Vehicles, and Casualties. The data is from the UK government website.',
        style={'margin': '10px 0', 'color': 'white'}
    ),
    html.P(
        'The United Kingdom Police Forces collects data on every vehicle collision in the UK on a form called Stats19.',
        style={'margin': '10px 0', 'color': 'white'}
    ),
    html.P([
        'Data from this form ends up at the DFT and is published at ',
        html.A('https://data.gov.uk/dataset/road-accidents-safety-data',
               href='https://data.gov.uk/dataset/road-accidents-safety-data',
               target='_blank',
               style={'color': 'lightblue'}),
    ], style={'margin': '10px 0', 'color': 'white'}),
    html.P(
        'This project aims to delve into the intricate relationships and patterns that exist among accidents, vehicles, and casualties in the UK, spanning a substantial period from 2005 to 2014.',
        style={'margin': '10px 0', 'color': 'white'}
    ),
    html.P(
        'The analysis of such a dataset can reveal multi-faceted insights and correlations that are crucial for understanding the dynamics of road safety and accident causation in the UK.',
        style={'margin': '10px 0', 'color': 'white'}
    ),
    html.P([
        'The dataset was found on Kaggle at ',
        html.A('https://www.kaggle.com/datasets/benoit72/uk-accidents-10-years-history-with-many-variables',
               href='https://www.kaggle.com/datasets/benoit72/uk-accidents-10-years-history-with-many-variables',
               target='_blank',
               style={'color': 'lightblue'}),
    ], style={'margin': '10px 0', 'color': 'white'}),
  ], style={'padding': '20px'}
)

@uk_accidents_app.callback(
    Output(component_id='tab_about', component_property='children'),
    [Input(component_id='tab_about', component_property='figure')])

def render_content(tab_about):

       return tab_about

#--------------------------------
# Subplots
#--------------------------------

# Create a dropdown of subplots:
# 1. Number of Accidents and Casualties per Year
# 2. Conditions that lead to an Accident
# 3. Vehicles affected in Accidents

# Allow only one subplot to be selected at a time

accidents_by_surface = accidents_vehicles_casualties.groupby('Road_Surface_Conditions').count()['Accident_Index'].reset_index()
accidents_by_surface_severity = accidents_vehicles_casualties.groupby(['Road_Surface_Conditions', 'Accident_Severity']).count()['Accident_Index'].reset_index()

accidents_by_weather = accidents_vehicles_casualties.groupby('Weather_Conditions').count()['Accident_Index'].reset_index()
accidents_by_weather_severity = accidents_vehicles_casualties.groupby(['Weather_Conditions', 'Accident_Severity']).count()['Accident_Index'].reset_index()

accidents_by_light = accidents_vehicles_casualties.groupby('Light_Conditions').count()['Accident_Index'].reset_index()
accidents_by_light_severity = accidents_vehicles_casualties.groupby(['Light_Conditions', 'Accident_Severity']).count()['Accident_Index'].reset_index()

accidents_by_age = accidents_vehicles_casualties.groupby('Age_of_Vehicle').count()['Accident_Index'].reset_index()

accidents_by_vehicle_affected = accidents_vehicles_casualties.groupby('Vehicle_Type').count()['Accident_Index'].reset_index()

accidents_by_impact = accidents_vehicles_casualties.groupby('1st_Point_of_Impact').count()['Accident_Index'].reset_index()

sub_plots = html.Div(
    children=[

    html.Header('The Sub-Plots fpr the given Data', style={'color': '#F57241'}),

    html.Label('Select a Subplot', style={'color': '#F57241'}),

    html.H4('UK Accidents Subplots', style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),

    dcc.Dropdown(id='dropdown',
                 options=[
                     {'label': 'Number of Accidents and Casualties per Year', 'value': 'accidents_casualties'},
                     {'label': 'Conditions that lead to an Accident', 'value': 'conditions'},
                     {'label': 'Vehicles affected in Accidents', 'value': 'vehicles'}
                 ],
                 value='accidents_casualties',
                 style={'width': '50%'},
                 multi = False,
                 ),

    html.Br(),

    #Add loading
    dcc.Loading(
        id="loading-1",
        type="default",
        children=html.Div(id='output_container', children=[])
    ),
    html.Br(),

    dcc.Loading(
        id="loading-2",
        type="default",
        children=html.Div(id='graph_container', children=[])
    ),
  ]
)

@uk_accidents_app.callback(
    [Output('output_container', 'children'),
     Output('graph_container', 'children')],
    [Input('dropdown', 'value')]
)

def select_dropdown(value):

    container = "The selected value was: {}".format(value)

    graphs = html.Div()

    if value == 'accidents_casualties':

        fig1 = px.line(accidents_by_year_severity, x='Year', y='Accident_Index', color='Accident_Severity', height=500,
                          title='Number of accidents per year based on Severity', template='plotly_dark').update_layout(
                {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        fig1_bar = px.bar(year_accidents, x='Year', y='Accident_Index', color='Accident_Index', height=500,
                        title='Number of accidents per year', template='plotly_dark').update_layout(
                {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        fig2 = px.line(casualties_by_year_severity, x='Year', y='Number_of_Casualties', color='Accident_Severity', height=500,
                            title='Number of casulties per year based on Severity', template='plotly_dark').update_layout(
                {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        fig2_bar = px.bar(casualties_by_year, x='Year', y='Number_of_Casualties', color='Number_of_Casualties', height=500,
                            title='Number of casulties per year', template='plotly_dark').update_layout(
                {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        graphs = html.Div(
            children=[
                dcc.Graph(
                    figure=fig1
                ),
                dcc.Graph(
                    figure=fig1_bar
                ),
                dcc.Graph(
                    figure=fig2
                ),
                dcc.Graph(
                    figure=fig2_bar
                ),
            ]
        )


    elif value == 'conditions':

        fig1 = px.bar(accidents_by_surface_severity, x='Road_Surface_Conditions', y='Accident_Index', color='Accident_Severity', height=500,
                      title='Number of accidents based on Road Surface Conditions and Severity', template='plotly_dark').update_layout(
            {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        fig1_pie = px.pie(accidents_by_surface, values='Accident_Index', names='Road_Surface_Conditions',
                        title='Pie Chart of Road Surface Conditions',
                        template='plotly_dark').update_layout(
            {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'showlegend': False})

        fig2 = px.bar(accidents_by_weather_severity, x='Weather_Conditions', y='Accident_Index', color='Accident_Severity', height=500,
                      title='Number of accidents based on Weather Conditions and Severity', template='plotly_dark').update_layout(
            {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        fig2_pie = px.pie(accidents_by_weather, values='Accident_Index', names='Weather_Conditions',
                        title='Pie Chart of Weather Conditions',
                        template='plotly_dark').update_layout(
            {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'showlegend': False})

        fig3 = px.bar(accidents_by_light_severity, x='Light_Conditions', y='Accident_Index', color='Accident_Severity', height=500,
                      title='Number of accidents based on Light Conditions and Severity', template='plotly_dark').update_layout(
            {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        fig3_pie = px.pie(accidents_by_light, values='Accident_Index', names='Light_Conditions',
                        title='Pie Chart of Light Conditions',
                        template='plotly_dark').update_layout(
            {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'showlegend': False})

        graphs = html.Div(
            children=[
                dcc.Graph(
                    figure=fig1
                ),
                dcc.Graph(
                    figure=fig1_pie
                ),
                dcc.Graph(
                    figure=fig2
                ),
                dcc.Graph(
                    figure=fig2_pie
                ),
                dcc.Graph(
                    figure=fig3
                ),
                dcc.Graph(
                    figure=fig3_pie
                ),
            ]
        )
        return container, graphs

    elif value == 'vehicles':

        fig1 = px.bar(accidents_by_age, x='Age_of_Vehicle', y='Accident_Index', color='Accident_Index', height=500,
                      title='Number of accidents based on Age of Vehicle', template='plotly_dark').update_layout(
            {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        fig2 = px.bar(accidents_by_vehicle_affected, x='Vehicle_Type', y='Accident_Index', color='Accident_Index', height=500,
                      title='Number of accidents based on Vehicle Type', template='plotly_dark').update_layout(
            {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        fig3 = px.bar(accidents_by_impact, x='1st_Point_of_Impact', y='Accident_Index', color='Accident_Index', height=500,
                      title='Number of accidents based on First Point of Impact', template='plotly_dark').update_layout(
            {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        graphs = html.Div(
            children=[
                dcc.Graph(
                    figure=fig1
                ),
                dcc.Graph(
                    figure=fig2
                ),
                dcc.Graph(
                    figure=fig3
                ),
            ]
        )

    return container, graphs


#--------------------------------
# Line Plots
#--------------------------------

# Create line graphs such that there should be a dropdown with a download button to all plots:
# 1. Number of Accidents per Year based on Severity
# 2. Number of Casualties per Year based on Severity

#
year_accidents = accidents_vehicles_casualties.groupby('Year')['Accident_Index'].count().reset_index()
accidents_by_year_severity = accidents_vehicles_casualties.groupby(['Year', 'Accident_Severity']).count()['Accident_Index'].reset_index()

casualties_by_year = accidents_vehicles_casualties.groupby('Year').sum()['Number_of_Casualties'].reset_index()
casualties_by_year_severity = accidents_vehicles_casualties.groupby(['Year', 'Accident_Severity']).sum()['Number_of_Casualties'].reset_index()

line_plots = html.Div(
    children=[

    html.Header('The Line Plots for the given Data', style={'color': '#F57241'}),

    html.Label('Select a Line Plot', style={'color': '#F57241'}),

    html.H4('UK Accidents Line Plots', style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),

    dcc.Dropdown(
        id='graph-dropdown',
        options=[
            {'label': 'Accidents per Year by Severity', 'value': 'accidents'},
            {'label': 'Casualties per Year by Severity', 'value': 'casualties'}
        ],
        style={'width': '50%'},
        multi=False,
        value='accidents',
    ),

    dcc.Graph(id='display-graph'),

    html.Button('Download Graph', id='download-graph-button'),

    dcc.Download(id='download1'),

    html.Br(),
    ]
)

@uk_accidents_app.callback(
    Output('display-graph', 'figure'),
    [Input('graph-dropdown', 'value')])

def update_graph(graph_type):

    if graph_type == 'accidents':

        fig1 = px.line(accidents_by_year_severity, x='Year', y='Accident_Index', color='Accident_Severity', height=500,
                            title='Number of accidents per year based on Severity', template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        return fig1

    else:

        fig2 = px.line(casualties_by_year_severity, x='Year', y='Number_of_Casualties', color='Accident_Severity', height=500,
                            title='Number of casulties per year based on Severity', template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        return fig2

@uk_accidents_app.callback(
    Output('download-graph', 'data'),
    [Input('download-graph-button', 'n_clicks'),
     Input('graph-dropdown', 'value')],prevent_initial_call=True)

def download_graph(n_clicks, graph_type):

    if not n_clicks:
        raise PreventUpdate

    fig = update_graph(graph_type)
    return dcc.send_bytes(fig.to_image(format="PNG"), filename=f"{graph_type}_graph.png")


#--------------------------------
# Bar Plots
#--------------------------------

# Create bar graphs such that there should be a dropdown:
# 1. Number of Accidents per Year
# 2. Number of Accidents per Year based on Severity
# 3. Number of Casualties per Year
# 4. Number of Casualties per Year based on Severity
# 5. Number of Vehicles involved in Accidents
# 6. Number of Accidents based on Settlement
# 7. Number of Accidents based on Settlement and Severity
# 7. Number of Accidents based on Police Attending Scene



accidents_by_vehicles = accidents_vehicles_casualties.groupby('Number_of_Vehicles').count()['Accident_Index'].reset_index()

accidents_by_settlement = accidents_vehicles_casualties.groupby('Urban_or_Rural_Area').count()['Accident_Index'].reset_index()
accidents_by_settlement_severity = accidents_vehicles_casualties.groupby(['Urban_or_Rural_Area', 'Accident_Severity']).count()['Accident_Index'].reset_index()

accidents_by_police = accidents_vehicles_casualties.groupby('Did_Police_Officer_Attend_Scene_of_Accident').count()['Accident_Index'].reset_index()


bar_plots = html.Div(
    children=[

    html.Header('The Bar Plots for the given Data', style={'color': '#F57241'}),

    html.Label('Select a Bar Plot', style={'color': '#F57241'}),

    html.H4('UK Accidents Bar Plots', style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),

        dcc.Dropdown(
            id='graph-bar-dropdown',
            options=[
                {'label': 'Accidents per Year', 'value': 'accidents'},
                {'label': 'Accidents per Year by Severity', 'value': 'accidents_severity'},
                {'label': 'Casualties per Year', 'value': 'casualties'},
                {'label': 'Casualties per Year by Severity', 'value': 'casualties_severity'},
                {'label': 'Vehicles involved in Accidents', 'value': 'vehicles'},
                {'label': 'Accidents based on Settlement', 'value': 'settlement'},
                {'label': 'Accidents based on Settlement and Severity', 'value': 'settlement_severity'},
                {'label': 'Accidents based on Police Attending Scene', 'value': 'police'}
            ],
            style={'width': '50%'},
            multi=False,
            value='accidents',
        ),
        dcc.Graph(id='display-bar-graph'),
        html.Br(),
    ]
)


@uk_accidents_app.callback(
    Output('display-bar-graph', 'figure'),
    [Input('graph-bar-dropdown', 'value')])

def update_graph(graph_type):

    if graph_type == 'accidents':

        fig1 = px.bar(year_accidents, x='Year', y='Accident_Index', color='Accident_Index', height=500,
                            title='Number of accidents per year', template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        return fig1

    elif graph_type == 'accidents_severity':

        fig2 = px.bar(accidents_by_year_severity, x='Year', y='Accident_Index', color='Accident_Severity', height=500,
                            title='Number of accidents per year based on Severity', template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        return fig2

    elif graph_type == 'casualties':

        fig3 = px.bar(casualties_by_year, x='Year', y='Number_of_Casualties', color='Number_of_Casualties', height=500,
                            title='Number of casulties per year', template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        return fig3

    elif graph_type == 'casualties_severity':

        fig4 = px.bar(casualties_by_year_severity, x='Year', y='Number_of_Casualties', color='Accident_Severity', height=500,
                            title='Number of casulties per year based on Severity', template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        return fig4

    elif graph_type == 'vehicles':

        fig5 = px.bar(accidents_by_vehicles, x='Number_of_Vehicles', y='Accident_Index', color='Accident_Index', height=500,
                            title='Number of vehicles involved in accidents', template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        return fig5

    elif graph_type == 'settlement':

        fig6 = px.bar(accidents_by_settlement, x='Urban_or_Rural_Area', y='Accident_Index', color='Accident_Index', height=500,
                            title='Number of accidents based on Settlement', template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        return fig6

    elif graph_type == 'settlement_severity':

        fig7 = px.bar(accidents_by_settlement_severity, x='Urban_or_Rural_Area', y='Accident_Index', color='Accident_Severity', height=500,
                            title='Number of accidents based on Settlement and Severity', template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        return fig7

    elif graph_type == 'police':

        fig8 = px.bar(accidents_by_police, x='Did_Police_Officer_Attend_Scene_of_Accident', y='Accident_Index', color='Accident_Index', height=500,
                            title='Number of accidents based on Police Attending Scene', template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        return fig8

#--------------------------------
# Pie Plots
#--------------------------------

# Create pie graphs such that there should be a dropdown:
# 1. Number of Accidents by Road Type
# 2. Number of Accidents by Speed Limit
# 3. Number of Accidents by Light Conditions
# 4. Number of Accidents by Weather Conditions
# 5. Number of Accidents by Road Surface Conditions
# 6. Number of Accidents by Special Conditions
# 7. Number of Accidents by Carriageway Hazards

accidents_by_road_type = accidents_vehicles_casualties.groupby('Road_Type').count()['Accident_Index'].reset_index()

accidents_by_speed_limit = accidents_vehicles_casualties.groupby('Speed_limit').count()['Accident_Index'].reset_index()

accidents_by_light_conditions = accidents_vehicles_casualties.groupby('Light_Conditions').count()['Accident_Index'].reset_index()

accidents_by_weather_conditions = accidents_vehicles_casualties.groupby('Weather_Conditions').count()['Accident_Index'].reset_index()

accidents_by_road_surface = accidents_vehicles_casualties.groupby('Road_Surface_Conditions').count()['Accident_Index'].reset_index()

accidents_by_special_conditions = accidents_vehicles_casualties.groupby('Special_Conditions_at_Site').count()['Accident_Index'].reset_index()

accidents_by_carriageway_hazards = accidents_vehicles_casualties.groupby('Carriageway_Hazards').count()['Accident_Index'].reset_index()

pie_plots = html.Div(
    children=[

    html.Header('The Pie Plots for the given Data', style={'color': '#F57241'}),

    html.Label('Select a Pie Plot', style={'color': '#F57241'}),

    html.H4('UK Accidents Pie Plots', style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),

    dcc.Dropdown(
        id='graph-pie-dropdown',
        options=[
            {'label': 'Accidents by Road Type', 'value': 'road_type'},
            {'label': 'Accidents by Speed Limit', 'value': 'speed_limit'},
            {'label': 'Accidents by Light Conditions', 'value': 'light_conditions'},
            {'label': 'Accidents by Weather Conditions', 'value': 'weather_conditions'},
            {'label': 'Accidents by Road Surface Conditions', 'value': 'road_surface'},
            {'label': 'Accidents by Special Conditions', 'value': 'special_conditions'},
            {'label': 'Accidents by Carriageway Hazards', 'value': 'carriageway_hazards'}
        ],
        style={'width': '50%'},
        multi=False,
        value='road_type',
    ),

    dcc.Graph(id='display-pie-graph'),

    html.Br(),
    ]
)

@uk_accidents_app.callback(
    Output('display-pie-graph', 'figure'),
    [Input('graph-pie-dropdown', 'value')])

def update_graph(graph_type):

    if graph_type == 'road_type':

        fig1 = px.pie(accidents_by_road_type, values='Accident_Index', names='Road_Type',
                            title='Pie Chart of Road Type',
                            template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'showlegend': False})

        return fig1

    elif graph_type == 'speed_limit':

        fig2 = px.pie(accidents_by_speed_limit, values='Accident_Index', names='Speed_limit',
                            title='Pie Chart of Speed Limit',
                            template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'showlegend': False})

        return fig2

    elif graph_type == 'light_conditions':

        fig3 = px.pie(accidents_by_light_conditions, values='Accident_Index', names='Light_Conditions',
                            title='Pie Chart of Light Conditions',
                            template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'showlegend': False})

        return fig3

    elif graph_type == 'weather_conditions':

        fig4 = px.pie(accidents_by_weather_conditions, values='Accident_Index', names='Weather_Conditions',
                            title='Pie Chart of Weather Conditions',
                            template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'showlegend': False})

        return fig4

    elif graph_type == 'road_surface':

        fig5 = px.pie(accidents_by_road_surface, values='Accident_Index', names='Road_Surface_Conditions',
                            title='Pie Chart of Road Surface Conditions',
                            template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'showlegend': False})

        return fig5

    elif graph_type == 'special_conditions':

        fig6 = px.pie(accidents_by_special_conditions, values='Accident_Index', names='Special_Conditions_at_Site',
                            title='Pie Chart of Special Conditions at Site',
                            template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'showlegend': False})

        return fig6

    elif graph_type == 'carriageway_hazards':

        fig7 = px.pie(accidents_by_carriageway_hazards, values='Accident_Index', names='Carriageway_Hazards',
                            title='Pie Chart of Carriageway Hazards',
                            template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'showlegend': False})

        return fig7


#--------------------------------
# Count Plots
#--------------------------------

# Create count plots such that there should be a dropdown:
# 1. Number of Accidents by Road Type
# 2. Number of Accidents by Speed Limit
# 3. Number of Accidents by Light Conditions
# 4. Number of Accidents by Weather Conditions
# 5. Number of Accidents by Road Surface Conditions
# 6. Number of Accidents by Special Conditions
# 7. Number of Accidents by Carriageway Hazards

count_plots = html.Div(
    children=[

    html.Header('The Count Plots for the given Data', style={'color': '#F57241'}),

    html.Label('Select a Count Plot', style={'color': '#F57241'}),

    html.H4('UK Accidents Count Plots', style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),

    dcc.Dropdown(
        id='graph-count-dropdown',
        options=[
            {'label': 'Accidents by Road Type', 'value': 'road_type'},
            {'label': 'Accidents by Speed Limit', 'value': 'speed_limit'},
            {'label': 'Accidents by Light Conditions', 'value': 'light_conditions'},
            {'label': 'Accidents by Weather Conditions', 'value': 'weather_conditions'},
            {'label': 'Accidents by Road Surface Conditions', 'value': 'road_surface'},
            {'label': 'Accidents by Special Conditions', 'value': 'special_conditions'},
            {'label': 'Accidents by Carriageway Hazards', 'value': 'carriageway_hazards'}
        ],
        style={'width': '50%'},
        multi=False,
        value='road_type',
    ),

    dcc.Graph(id='display-count-graph'),

    html.Br(),

    ]
)

@uk_accidents_app.callback(
    Output('display-count-graph', 'figure'),
    [Input('graph-count-dropdown', 'value')])

def update_graph(graph_type):

    if graph_type == 'road_type':

        fig1 = px.bar(accidents_by_road_type, x='Road_Type', y='Accident_Index', color='Accident_Index', height=500,
                            title='Number of accidents based on Road Type', template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0,0,0,0)'})

        return fig1

    elif graph_type == 'speed_limit':

        fig2 = px.bar(accidents_by_speed_limit, x='Speed_limit', y='Accident_Index', color='Accident_Index', height=500,
                            title='Number of accidents based on Speed Limit', template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0,0,0,0)'})

        return fig2

    elif graph_type == 'light_conditions':

        fig3 = px.bar(accidents_by_light_conditions, x='Light_Conditions', y='Accident_Index', color='Accident_Index', height=500,
                            title='Number of accidents based on Light Conditions', template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0,0,0,0)'})

        return fig3

    elif graph_type == 'weather_conditions':

        fig4 = px.bar(accidents_by_weather_conditions, x='Weather_Conditions', y='Accident_Index', color='Accident_Index', height=500,
                            title='Number of accidents based on Weather Conditions', template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0,0,0,0)'})

        return fig4

    elif graph_type == 'road_surface':

        fig5 = px.bar(accidents_by_road_surface, x='Road_Surface_Conditions', y='Accident_Index', color='Accident_Index', height=500,
                            title='Number of accidents based on Road Surface Conditions', template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0,0,0,0)'})

        return fig5

    elif graph_type == 'special_conditions':

        fig6 = px.bar(accidents_by_special_conditions, x='Special_Conditions_at_Site', y='Accident_Index', color='Accident_Index', height=500,
                            title='Number of accidents based on Special Conditions at Site', template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0,0,0,0)'})

        return fig6

    elif graph_type == 'carriageway_hazards':

        fig7 = px.bar(accidents_by_carriageway_hazards, x='Carriageway_Hazards', y='Accident_Index', color='Accident_Index', height=500,
                            title='Number of accidents based on Carriageway Hazards', template='plotly_dark').update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0,0,0,0)'})

        return fig7


#--------------------------------
# Histogram Plots
#--------------------------------

# Create histogram plots such that there should be a dropdown:
# 1. Number of Vehicles
# 2. Number of Casualties
# 3. Speed Limit
# 4. Age of Vehicle
# 5. Engine Capacity
# 6. Age of Driver
# 7. Number of Casualties by Age of Driver

histogram_plots = html.Div(
    children=[

    html.Header('The Histogram Plots for the given Data', style={'color': '#F57241'}),

    html.Label('Select a Histogram Plot', style={'color': '#F57241'}),

    html.H4('UK Accidents Histogram Plots', style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),

    dcc.Dropdown(
        id='graph-hist-dropdown',
        options=[
            {'label': 'Number of Vehicles', 'value': 'vehicles'},
            {'label': 'Number of Casualties', 'value': 'casualties'},
            {'label': 'Speed Limit', 'value': 'speed_limit'},
            {'label': 'Age of Vehicle', 'value': 'age_vehicle'},
            {'label': 'Engine Capacity', 'value': 'engine_capacity'},
            {'label': 'Age of Driver', 'value': 'age_driver'},
            {'label': 'Number of Casualties by Age of Driver', 'value': 'casualties_age_driver'}
        ],
        style={'width': '50%'},
        multi=False,
        value='vehicles',
    ),

    dcc.Graph(id='display-hist-graph'),

    html.Br(),

    ]
)

@uk_accidents_app.callback(
    Output('display-hist-graph', 'figure'),
    [Input('graph-hist-dropdown', 'value')])

def update_graph(graph_type):

    if graph_type == 'vehicles':

        fig1 = px.histogram(accidents_vehicles_casualties, x='Number_of_Vehicles', color='Number_of_Vehicles',
                            title='Histogram of Number of Vehicles',
                            template='plotly_dark', nbins=20).update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        return fig1

    elif graph_type == 'casualties':

        fig2 = px.histogram(accidents_vehicles_casualties, x='Number_of_Casualties', color='Number_of_Casualties',
                            title='Histogram of Number of Casualties',
                            template='plotly_dark', nbins=20).update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        return fig2

    elif graph_type == 'speed_limit':

        fig3 = px.histogram(accidents_vehicles_casualties, x='Speed_limit', color='Speed_limit',
                            title='Histogram of Speed Limit',
                            template='plotly_dark', nbins=20).update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        return fig3

    elif graph_type == 'age_vehicle':

        fig4 = px.histogram(accidents_vehicles_casualties, x='Age_of_Vehicle', color='Age_of_Vehicle',
                            title='Histogram of Age of Vehicle',
                            template='plotly_dark', nbins=20).update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        return fig4

    elif graph_type == 'engine_capacity':

        fig5 = px.histogram(accidents_vehicles_casualties, x='Engine_Capacity_(CC)', color='Engine_Capacity_(CC)',
                            title='Histogram of Engine Capacity',
                            template='plotly_dark', nbins=20).update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'showlegend': False})

        return fig5

    elif graph_type == 'age_driver':

        fig6 = px.histogram(accidents_vehicles_casualties, x='Age_of_Driver', color='Age_of_Driver',
                            title='Histogram of Age of Driver',
                            template='plotly_dark', nbins=20).update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'showlegend': False})

        return fig6

    elif graph_type == 'casualties_age_driver':

        fig7 = px.histogram(accidents_vehicles_casualties, x='Age_of_Driver', color='Number_of_Casualties',
                            title='Histogram of Number of Casualties by Age of Driver',
                            template='plotly_dark', nbins=20).update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'showlegend': False})

        return fig7


#--------------------------------
# Miscellaneous Plots
#--------------------------------

# Create a check list to select the type of violin plot to be displayed:
# 1. Area plot of Number of Accidents by year
# 1. Area plot of Number of Casualties by year
# 2. Area plot of Number of Vehicles affected by year

accidents_by_year_severity = accidents_vehicles_casualties.groupby(['Year', 'Accident_Severity']).count()['Accident_Index'].reset_index()
casualties_by_year_severity = accidents_vehicles_casualties.groupby(['Year', 'Accident_Severity']).sum()['Number_of_Casualties'].reset_index()
accidents_by_vehicles_severity = accidents_vehicles_casualties.groupby(['Number_of_Vehicles', 'Accident_Severity']).count()['Accident_Index'].reset_index()

misc_plots = html.Div(
    children=[

    html.Header('The respective Plots for the given Data', style={'color': '#F57241'}),

    html.Label('Select the type of Miscellaneous Plots to be displayed:'),

    dcc.Checklist(
        id='misc_plots',
        options=[
            {'label': 'Area Plot - Number of Accidents per Year', 'value': 'area_accidents_by_year'},
            {'label': 'Area Plot - Number of Casualties per Year', 'value': 'area_casualties_by_year'},
            {'label': 'Area Plot - Number of Vehicles affected per Year', 'value': 'area_vehicles_by_year'},
        ],
        value=[],
        labelStyle={'display': 'inline-block', 'margin-right': '20px'}
    ),

    html.Div(id='misc_plots_output'),

    ]
)

@uk_accidents_app.callback(
    Output(component_id='misc_plots_output', component_property='children'),
    [Input(component_id='misc_plots', component_property='value')])

def update_misc_plots(plot_types):
    graphs = []

    if 'area_accidents_by_year' in plot_types:
        graphs.append(dcc.Graph(
            id='area_accidents_by_year',
            figure=px.area(accidents_by_year_severity, x='Year', y='Accident_Index', color='Accident_Severity', height=500,
                           title='Area plot of Number of Accidents by year', template='plotly_dark').update_layout(
                {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
            ))

    if 'area_casualties_by_year' in plot_types:
        graphs.append(dcc.Graph(
            id='area_casualties_by_year',
            figure=px.area(casualties_by_year_severity, x='Year', y='Number_of_Casualties', color='Accident_Severity', height=500,
                           title='Area plot of Number of Casualties by year', template='plotly_dark').update_layout(
                {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
            ))

    if 'area_vehicles_by_year' in plot_types:
        graphs.append(dcc.Graph(
            id='area_vehicles_by_year',
            figure=px.area(accidents_by_vehicles_severity, x='Number_of_Vehicles', y='Accident_Index', color='Accident_Severity', height=500,
                           title='Area plot of Number of Vehicles affected by year', template='plotly_dark').update_layout(
                {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
            ))

    return graphs



#--------------------------------
# Geospatial Plots
#--------------------------------

# Create a radio button to select the type of geospatial plot to be displayed, but not multiple plots at the same time:
# 1. Density Heatmap of accidents in the UK in year slider
# 2. Bubble Map of accidents in the UK in year slider
# 3. 3D Scatter Plot of accidents in the UK in year slider
# 4. Cluster Map of accidents in the UK in year slider
# 5. Line Map of accidents where accidents occurred in the UK in year slider


geo_plots = html.Div(
    children=[

    html.Header('The Geospatial Plots for the given Data', style={'color': '#F57241'}),

    html.Label('Select the type of Geospatial Plot to be displayed:'),

    html.H4('UK Accidents Geospatial Plots', style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),

    dcc.RadioItems(
        id='geospatial_plots',
        options=[
            {'label': 'Density Heatmap', 'value': 'density_heatmap'},
            {'label': 'Bubble Map', 'value': 'bubble_map'},
            {'label': '3D Scatter Plot', 'value': '3d_scatter_plot'},
            {'label': 'Cluster Map', 'value': 'cluster_map'},
            {'label': 'Line Map', 'value': 'line_map'},
        ],
        value=[],
        labelStyle={'display': 'inline-block', 'margin-right': '20px'}
    ),

    html.Br(),

    dcc.Slider(
        id='year_slider',
        min=2005,
        max=2014,
        value=2005,
        marks={str(year): str(year) for year in range(2005, 2015)},
        step=None
    ),


    html.Div(id='geospatial_plots_output'),


    dcc.Textarea(
        id='geospatial_comments',
        placeholder='Enter your comments or descriptions here...',
        style={'width': '100%', 'height': 100},
    ),

    html.Button('Submit Comment', id='submit_comment_button', n_clicks=0),

    html.Div(id='comments_output'),
    ]
)

@uk_accidents_app.callback(
    Output(component_id='geospatial_plots_output', component_property='children'),
    [Input(component_id='geospatial_plots', component_property='value'),
     Input(component_id='year_slider', component_property='value')])

def update_slider(plot_type, year):

    filtered_data = accidents_vehicles_casualties[accidents_vehicles_casualties['Year'] == year]
    plot = render_plot(plot_type, filtered_data)

    if plot:
        return dcc.Graph(figure=plot)

    return "Select plot type and year to display."

def render_plot(plot_type, filtered_data):

    if plot_type == 'density_heatmap':
        return px.density_mapbox(filtered_data, lat='Latitude', lon='Longitude', radius=10, zoom=5, height=800,
                                 hover_data=['Accident_Index'],
                                 hover_name='Accident_Index',
                                 title='Density Heatmap of Accidents in the UK in {}'.format(filtered_data['Year'].unique()[0]),
                                 template='plotly_dark').update_layout(
            mapbox_style="open-street-map", mapbox_center_lon=0, mapbox_center_lat=52)

    elif plot_type == 'bubble_map':
        return px.scatter_mapbox(filtered_data, lat='Latitude', lon='Longitude', color='Accident_Severity', size='Number_of_Casualties',
                                 hover_data=['Accident_Index'],
                                 hover_name='Accident_Index',
                                 zoom=5, height=800, title='Bubble Map of Accidents in the UK in {}'.format(filtered_data['Year'].unique()[0]),
                                 template='plotly_dark').update_layout(
            mapbox_style="open-street-map", mapbox_center_lon=0, mapbox_center_lat=52)

    elif plot_type == '3d_scatter_plot':
        return px.scatter_3d(filtered_data, x='Longitude', y='Latitude', z='Number_of_Casualties', color='Accident_Severity',
                             hover_data=['Accident_Index'],
                             hover_name='Accident_Index',
                             height=800, title='3D Scatter Plot of Accidents in the UK in {}'.format(filtered_data['Year'].unique()[0]),
                             template='plotly_dark')

    elif plot_type == 'cluster_map':
        return px.scatter_mapbox(filtered_data, lat='Latitude', lon='Longitude', color='Accident_Severity', zoom=5, height=800,
                                 hover_data=['Accident_Index'],
                                 hover_name='Accident_Index',
                                 title='Cluster Map of Accidents in the UK in {}'.format(filtered_data['Year'].unique()[0]),
                                 template='plotly_dark').update_layout(
            mapbox_style="open-street-map", mapbox_center_lon=0, mapbox_center_lat=52, mapbox_zoom=5)

    elif plot_type == 'line_map':
        return px.line_mapbox(filtered_data, lat='Latitude', lon='Longitude', color='Accident_Severity', zoom=5, height=800,
                              hover_data=['Accident_Index'],
                              hover_name='Accident_Index',
                              title='Line Map of Accidents in the UK in {}'.format(filtered_data['Year'].unique()[0]),
                              template='plotly_dark').update_layout(
                mapbox_style="open-street-map", mapbox_center_lon=0, mapbox_center_lat=52, mapbox_zoom=5)

    else:
        return None

@uk_accidents_app.callback(
    Output(component_id='comments_output', component_property='children'),
    [Input(component_id='submit_comment_button', component_property='n_clicks')],
    [State(component_id='geospatial_comments', component_property='value')])

def update_comments(n_clicks, comments):

    if n_clicks > 0:
        return html.Div([
            html.Hr(),
            html.H5('Comments:'),
            html.Br(),
            html.Div(comments)
        ])

    return None

uk_accidents_app.run_server(port = 8050, host='0.0.0.0')








