import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from dash import dash_table

# color scheme
main_color = 'steelblue'
dark_color = 'navy'
accent_color = 'orange'

# Load preprocessed dataframe
path_df_cleaned = 'Data/meteo_pivoted_cleaned.parquet'
df = pd.read_parquet(path_df_cleaned)

# Load original df
path_df_original = 'Data/meteo_pivoted_original.parquet'
df_original = pd.read_parquet(path_df_original)

pollutants = ['PM2.5', 'NOx', 'NO2', 'NO', 'eBC2.5', 'PN[5-100nm]', 'SO2', 'CO2']
weather_columns = ['Hr', 'Hr_Trockner', 'RainDur', 'StrGlo', 'T', 'T_Trockner', 'WD', 'WVv']
# pollutants_columns = ['CO2', 'NO', 'NO2', 'NOx', 'PM2.5', 'PN[5-100nm]', 'SO2',  'eBC2.5']
# weather_columns = ['Hr', 'Hr_Trockner', 'RainDur', 'StrGlo', 'T', 'T_Trockner', 'WD', 'WVv']
# pollutants_remove_days = ['NO', 'NO2', 'NOx', 'PM2.5']
pollutants_interpolate = ['NO', 'NO2', 'NOx', 'PM2.5','eBC2.5', 'PN[5-100nm]']
# pollutants_ignore = ['CO2', 'SO2']

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    html.H1("Air Pollution Project Dashboard"),
    dcc.Tabs(id='tabs', value='tab-eda', children=[
        dcc.Tab(label='0. Overview', value='tab-overview'),
        dcc.Tab(label='1. Initial EDA', value='tab-eda'),
        dcc.Tab(label='2. Missing Data', value='tab-missing'),
        dcc.Tab(label='3. EDA and correlations', value='tab-correlations'),
        dcc.Tab(label='4. Predictive Modeling', value='tab-model'),
        dcc.Tab(label='5. COVID Lockdown Analysis', value='tab-covid')
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-overview':
        return dcc.Markdown('''
### Project Overview

This project explores air pollution patterns in Kloten using 10-minute interval environmental data from 2020 to 2023.

Using data from https://www.zh.ch/de/umwelt-tiere/luft-strahlung/luftqualitaet-auswirkungen.html

**Main goals:**
- Explore pollution patterns and missing data
- Model NOx and PM2.5 levels from weather and time features
- Understand which variables drive pollution behavior
- Assess impact of COVID lockdowns on pollution

**Data summary:**
- Time range: 5.3.2020–31.12.2023
- Resolution: 10-minute intervals
- Pollutants: PM2.5, NOx, NO2, NO, CO2, SO2, eBC2.5, PN[5-100nm]
- Weather: 
    - T: ambient air temperature
    - StrGlo: ambient global radiation
    - Hr: ambient air relative humidity
    - RainDur: rain duration
    - WD: wind direction
    - WVv: vector wind velocity
    - T_Trockner: UFP aerosol sample inlet air temperature (measured after dryer)
    - Hr_Trockner: UFP aerosol sample inlet relative humidity (measured after dryer)

Use the tabs above to explore:
- Exploratory plots (EDA)
- Missing data and interpolation
- Data correlations and aggregation
- Predictive modeling results
- COVID lockdown comparisons
''')
    elif tab == 'tab-eda':
        return html.Div([
            html.Label("Select variable:"),
            dcc.Dropdown(pollutants + weather_columns, 'PM2.5', id='eda-var'),
            dcc.Graph(id='eda-time-series'),
            dcc.Graph(id='eda-distribution'),
            html.Div(id='eda-summary')
        ])

    elif tab == 'tab-missing':
        return html.Div([
            html.Label("Select variable:"),
            dcc.Dropdown(pollutants_interpolate + weather_columns, 'PM2.5', id='missing-var'),
            dcc.Graph(id='interpolation-plot'),
            html.Div(id='missing-stats')
        ])

    elif tab == 'tab-correlations':
        df['day_of_week_name'] = df.index.day_name()
        timespans = ['Day of week', 'Day of year', 'Season', 'Hour']
        return html.Div([
            html.Div([
                html.Label("Choose timespan:", style={'marginRight': '10px'}),
                dcc.Dropdown(timespans, 'Day of week', id='timespan-dropdown', style={'width': '100%'})
            ], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '48%'}),

            html.Div([
                html.Label("Choose pollutant:", style={'marginRight': '10px'}),
                dcc.Dropdown(pollutants, 'NOx', id='pollutant-dropdown', style={'width': '100%'})
            ], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '48%', 'marginLeft': '4%'}),

            dcc.Graph(id='pollutant-graph')
        ])

    elif tab == 'tab-model':
        return html.Div([
            html.Label("Select Target Variable:"),
            dcc.Dropdown(
                id='model-target-dropdown',
                options=[{'label': 'NOx', 'value': 'NOx'}, {'label': 'PM2.5', 'value': 'PM2.5'}],
                value='NOx',
                clearable=False
            ),
            html.Label("Select Model:"),
            dcc.Dropdown(
                id='model-type-dropdown',
                options=[{'label': 'HistGradientBoosting', 'value': 'HGBR'}, {'label': 'Random Forest', 'value': 'RF'}],
                value='HGBR',
                clearable=False
            ),
            html.Div(id='model-metrics'),
            dcc.Graph(id='model-pred-vs-actual'),
            html.Div(id='model-top-features')
        ])

    elif tab == 'tab-covid':
        return html.Div([
            html.Label("Select pollutant to compare:"),
            dcc.Dropdown(
                options=[{'label': col, 'value': col} for col in pollutants if col not in ['SO2', 'eBC2.5']],
                value='PN[5-100nm]',
                multi=True,
                id='covid-pollutants'
            ),
            html.Div(id='covid-plots')
        ])

# Callbacks for basic EDA
@app.callback(
    Output('eda-time-series', 'figure'),
    Output('eda-distribution', 'figure'),
    Output('eda-summary', 'children'),
    Input('eda-var', 'value')
)
def update_eda(var):
    fig1 = px.line(df, x=df.index, y=var, title=f"Time Series of {var}", color_discrete_sequence=[main_color])
    # fig2 = px.histogram(df, x=var, nbins=100, title=f"Distribution of {var}", color_discrete_sequence=[main_color])

    hist_data = df[var].dropna()
    hist_y, hist_x = np.histogram(hist_data, bins=50, density=True)
    kde_x = np.linspace(hist_data.min(), hist_data.max(), 500)
    kde = gaussian_kde(hist_data, bw_method='scott')
    kde_y = kde.evaluate(kde_x)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=hist_x[:-1], y=hist_y, name='Histogram', marker_color=main_color, opacity=0.7))
    fig2.add_trace(go.Scatter(x=kde_x, y=kde_y, name='KDE', line=dict(color=dark_color)))
    fig2.update_layout(title=f"Distribution and KDE of {var}", xaxis_title=var, yaxis_title='Density')

    summary = df[var].describe().round(2).to_frame(name='Value')
    summary.reset_index(inplace=True)
    summary.columns = ['Statistic', 'Value']
    summary_table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in summary.columns],
        data=summary.to_dict('records'),
        style_table={'width': '50%', 'margin': '20px 0'},
        style_cell={'textAlign': 'left'},
    )
    return fig1, fig2, summary_table

# Missing data tab callback
@app.callback(
    Output('interpolation-plot', 'figure'),
    Output('missing-stats', 'children'),
    Input('missing-var', 'value')
)
def update_missing_plot(var):
    fig = go.Figure()
    real = df[df[f'{var}_was_interpolated'] == 0]
    interp = df[df[f'{var}_was_interpolated'] == 1]
    gap = df[df[f'{var}_gap_too_long'] == 1]

    fig.add_trace(go.Scatter(x=real.index, y=real[var], mode='lines', name='Observed', line=dict(color=main_color)))
    fig.add_trace(go.Scatter(x=interp.index, y=interp[var], mode='markers', name='Interpolated', marker=dict(color=accent_color, size=5)))
    fig.add_trace(go.Scatter(x=gap.index, y=gap[var], mode='markers', name='Gap too long', marker=dict(color='red', size=6, symbol='x')))

    fig.update_layout(title=f"{var} - Interpolation Overview", xaxis_title='Time', yaxis_title=var, template='plotly_white')

    total = len(df_original)
    missing = df_original[var].isna().sum()
    interpolated = df[f'{var}_was_interpolated'].sum()
    too_long = df[f'{var}_gap_too_long'].sum()

    longest_gap_before = (df_original[var].isna().astype(int)
                   .groupby((df_original[var].notna()).cumsum())
                   .sum().max())
    
    longest_gap_after = (df[var].isna().astype(int)
                   .groupby((df[var].notna()).cumsum())
                   .sum().max())

    stats = pd.DataFrame({
        'Statistic': ['Total points', 'Missing before interpolation', 'Interpolated points', 'Gaps too long', 'Longest missing gap before interpolation (in days)', 'Longest missing gap after interpolation (in days)'],
        'Value': [total, missing, interpolated, too_long, round(longest_gap_before/144, 2), round(longest_gap_after/144, 2)]
    })

    stats_table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in stats.columns],
        data=stats.to_dict('records'),
        style_table={'width': '60%', 'margin': '20px 0'},
        style_cell={'textAlign': 'left'},
    )

    return fig, stats_table

# Callback for advanced EDA

@app.callback(
    Output('pollutant-graph', 'figure'),
    Input('timespan-dropdown', 'value'),
    Input('pollutant-dropdown', 'value')
)
def update_advanced_graph(timespan, pollutant):
    if timespan == 'Day of week':
        group_col = 'day_of_week_name'
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    elif timespan == 'Season':
        group_col = 'season'
        order = ['Winter', 'Spring', 'Summer', 'Autumn']
    elif timespan == 'Day of year':
        group_col = 'day_of_year'
        order = sorted(df[group_col].unique())
    else:
        group_col = 'hour'
        order = sorted(df[group_col].unique())
    

    df_grouped = df.groupby(group_col)[pollutant].mean().reindex(order)

    fig = px.bar(
        df_grouped,
        x=df_grouped.index,
        y=pollutant,
        color_discrete_sequence=[main_color],
        labels={'x': timespan, pollutant: f"{pollutant} mean concentration"},
        title=f"Average {pollutant} concentration by {timespan.lower()}"
    )
    fig.update_layout(xaxis_tickangle=45, hovermode="x unified", bargap=0.3 if len(df_grouped) < 10 else 0.05)
    return fig
# @app.callback(
#     Output('adv-hour-box', 'figure'),
#     Output('adv-weekday-box', 'figure'),
#     Output('adv-corr-heatmap', 'figure'),
#     Input('adv-var', 'value')
# )
# def update_advanced(var):
#     df['hour'] = df.index.hour
#     df['weekday'] = df.index.day_name()
#     fig1 = px.box(df, x='hour', y=var, title=f"{var} by Hour", color_discrete_sequence=[main_color])
#     fig2 = px.box(df, x='weekday', y=var, title=f"{var} by Weekday", color_discrete_sequence=[main_color])
#     corr = df[pollutants + weather_columns].corr()
#     fig3 = px.imshow(corr, text_auto=True, title="Correlation Matrix", color_discrete_sequence=[main_color])
#     return fig1, fig2, fig3

# Callback for models tab
@app.callback(
    Output('model-metrics', 'children'),
    Output('model-pred-vs-actual', 'figure'),
    Output('model-top-features', 'children'),
    Input('model-target-dropdown', 'value'),
    Input('model-type-dropdown', 'value')
)
def update_model_tab(target, model_type):
    import json
    import joblib
    import os

    # Load metrics
    metrics_path = f"results/{model_type}_{target}_weather_only_metrics.json"
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

    metrics = html.Ul([
        html.Li(f"R²: {metrics_data['r2']:.3f}"),
        html.Li(f"RMSE: {metrics_data['rmse']:.3f}"),
        html.Li(f"MAE: {metrics_data['mae']:.3f}")
    ])

    # Load predictions
    pred_path = f"results/{model_type}_{target}_weather_only_predictions_.parquet"
    pred_df = pd.read_parquet(pred_path)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=pred_df['y_true'], mode='lines', name='Actual', line=dict(color=main_color)))
    fig.add_trace(go.Scatter(y=pred_df['y_pred'], mode='lines', name='Predicted', line=dict(color=accent_color)))
    fig.update_layout(title=f"{target} Prediction – {model_type}", xaxis_title='Time Step', yaxis_title=target)

    feature_fig = html.Div([html.Img(src=f"/assets/perm_importance_{model_type}_{target}.png", style={"width": "50%", "margin": "20px auto", "display": "block"})])


    # feature_table = dash_table.DataTable(
    #     columns=[{"name": i, "id": i} for i in top_perm.columns],
    #     data=top_perm.to_dict("records"),
    #     style_table={'width': '60%', 'margin': '20px 0'},
    #     style_cell={'textAlign': 'left'},
    # )

    return metrics, fig, feature_fig

    # return metrics, fig, feature_table

# Callback for COVID tab
@app.callback(
    Output('covid-plots', 'children'),
    Input('covid-pollutants', 'value')
)
def update_covid_plots(selected_pollutants):
    plots = []
    start_day = 65
    end_day = 228

    non_2020 = df[df['year'] != 2020]
    just_2020 = df[df['year'] == 2020]

    for col in selected_pollutants:
        mean_all_years = (
            non_2020.groupby('day_of_year')[col]
            .mean()
            .loc[start_day:end_day]
        )

        values_2020 = (
            just_2020.groupby('day_of_year')[col]
            .mean()
            .loc[start_day:end_day]
        )

        common_days = mean_all_years.index.intersection(values_2020.index)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=common_days, y=mean_all_years.loc[common_days],
                             name='All years (excl. 2020)', marker_color=main_color))
        fig.add_trace(go.Bar(x=common_days, y=values_2020.loc[common_days],
                             name='2020 (lockdown)', marker_color=accent_color, opacity=0.7))
        fig.add_shape(type="line", x0=76, x1=76, y0=0, y1=max(mean_all_years.max(), values_2020.max()),
                      line=dict(color="red", dash="dash"))

        fig.update_layout(
            title=f"{col} during lockdown (2020 vs previous years)",
            xaxis_title="Day of Year",
            yaxis_title=f"Mean {col}",
            template='plotly_white',
            height=400
        )
        plots.append(dcc.Graph(figure=fig))

    return plots
if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=7860, debug=False)

