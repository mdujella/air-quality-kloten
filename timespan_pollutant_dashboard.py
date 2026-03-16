import pandas as pd
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output
import plotly.express as px


# Load pre-cleaned dataset
path_df_cleaned = 'Data/meteo_pivoted_cleaned.parquet'
df_cleaned = pd.read_parquet(path_df_cleaned)

df_cleaned['day_of_week_name'] = df_cleaned.index.day_name()


# List of pollutants
pollutants = ['NOx', 'PM2.5','CO2', 'PN[5-100nm]', 'SO2','eBC2.5']
timespans = ['Day of week', 'Day of year', 'Season']

# color scheme
main_color = 'steelblue'
dark_color = 'navy'
accent_color = 'orange'

# Start Dash app
app = dash.Dash(__name__)
app.title = "Pollutant Dashboard"

app.layout = html.Div([
    html.H1("Interactive Pollutant Dashboard"),
    
    html.Div([
        html.Label("Choose timespan:"),
        dcc.Dropdown(timespans, 'Day of week', id='timespan-dropdown')
    ], style={'width': '40%', 'display': 'inline-block'}),

    html.Div([
        html.Label("Choose pollutant:"),
        dcc.Dropdown(pollutants, 'NOx', id='pollutant-dropdown')
    ], style={'width': '40%', 'display': 'inline-block', 'marginLeft': '5%'}),
    
    dcc.Graph(id='pollutant-graph')
])

@app.callback(
    Output('pollutant-graph', 'figure'),
    Input('timespan-dropdown', 'value'),
    Input('pollutant-dropdown', 'value')
)
def update_graph(timespan, pollutant):
    if timespan == 'Day of week':
        group_col = 'day_of_week_name'
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    elif timespan == 'Season':
        group_col = 'season'
        order = ['Winter', 'Spring', 'Summer', 'Autumn']
    else:
        group_col = 'day_of_year'
        order = sorted(df_cleaned[group_col].unique())

    df_grouped = df_cleaned.groupby(group_col)[pollutant].mean().reindex(order)

    fig = px.bar(df_grouped, x=df_grouped.index, y=pollutant, color_discrete_sequence=[main_color],
                 labels={'x': timespan, pollutant: f"{pollutant} mean concentration"},
                 title=f"Average {pollutant} concentration by {timespan.lower()}")

    fig.update_layout(xaxis_tickangle=45, hovermode="x unified", 
                      bargap=0.3 if len(df_grouped) < 10 else 0.05  # wider spacing for few bars
                      )

    return fig

if __name__ == '__main__':
    app.run(debug=True)