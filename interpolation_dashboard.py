import pandas as pd
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output

# Load pre-cleaned dataset
path_df_cleaned = 'Data/meteo_pivoted_cleaned.parquet'
df_cleaned = pd.read_parquet(path_df_cleaned)

# color scheme
main_color = 'steelblue'
dark_color = 'navy'
accent_color = 'orange'

pollutants_columns = ['CO2', 'NO', 'NO2', 'NOx', 'PM2.5', 'PN[5-100nm]', 'SO2',  'eBC2.5']
weather_columns = ['Hr', 'Hr_Trockner', 'RainDur', 'StrGlo', 'T', 'T_Trockner', 'WD', 'WVv']
pollutants_remove_days = ['NO', 'NO2', 'NOx', 'PM2.5']
pollutants_interpolate = ['NO', 'NO2', 'NOx', 'PM2.5','eBC2.5', 'PN[5-100nm]']
pollutants_ignore = ['CO2', 'SO2']

# Launch app
app = dash.Dash(__name__)
server = app.server  # for deployment if needed

# Choose variables to include
all_vars = pollutants_interpolate + weather_columns

# Layout
app.layout = html.Div([
    html.H1("Interpolated Values Dashboard"),
    html.Label("Select variable:"),
    dcc.Dropdown(all_vars, value='PM2.5', id='var-dropdown'),
    dcc.Graph(id='interpolation-plot'),
])

# Callback to update plot
@app.callback(
    Output('interpolation-plot', 'figure'),
    Input('var-dropdown', 'value')
)
def update_plot(var):
    fig = go.Figure()

    # Real values
    real = df_cleaned[df_cleaned[f'{var}_was_interpolated'] == 0]
    fig.add_trace(go.Scatter(x=real.index, y=real[var], mode='lines',
                             name='Observed', line=dict(color=main_color)))

    # Interpolated values
    interp = df_cleaned[df_cleaned[f'{var}_was_interpolated'] == 1]
    fig.add_trace(go.Scatter(x=interp.index, y=interp[var], mode='markers',
                             name='Interpolated', marker=dict(color=accent_color, size=5)))

    # Gaps too long
    gap = df_cleaned[df_cleaned[f'{var}_gap_too_long'] == 1]
    fig.add_trace(go.Scatter(x=gap.index, y=gap[var], mode='markers',
                             name='Gap too long', marker=dict(color='red', size=6, symbol='x')))

    fig.update_layout(title=f"{var} - Interpolation Overview",
                      xaxis_title='Time', yaxis_title=var,
                      template='plotly_white')
    return fig

# Run app
if __name__ == '__main__':
    app.run(debug=True)