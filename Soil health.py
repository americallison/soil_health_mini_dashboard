import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pykrige.ok import OrdinaryKriging
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Generate soil data
np.random.seed(42)
data = {
    'sample_id': range(1, 51),
    'x': np.random.uniform(0, 100, 50),
    'y': np.random.uniform(0, 100, 50),
    'pH': np.round(np.random.normal(6.5, 0.8, 50), 1),
    'organic_matter_pct': np.round(np.random.uniform(1.5, 4.5, 50), 1),
    'nitrogen_ppm': np.random.randint(20, 150, 50),
    'phosphorus_ppm': np.random.randint(10, 80, 50),
    'potassium_ppm': np.random.randint(100, 500, 50),
    'sand_pct': np.random.randint(40, 80, 50),
    'bulk_density_gcm3': np.round(np.random.uniform(1.1, 1.6, 50), 2),
}

soil_df = pd.DataFrame(data)
soil_df['health_index'] = (
    0.2*soil_df['organic_matter_pct'] + 
    0.2*(7 - np.abs(soil_df['pH'] - 6.5)) + 
    0.2*soil_df['nitrogen_ppm']/50 +
    0.1*soil_df['phosphorus_ppm']/30 +
    0.1*soil_df['potassium_ppm']/200 +
    0.1*soil_df['sand_pct']/60 +
    0.1*soil_df['bulk_density_gcm3']
).round(2)


# Prepare features for PCA
features = ['pH', 'organic_matter_pct', 'nitrogen_ppm', 
            'phosphorus_ppm', 'potassium_ppm', 'sand_pct', 'bulk_density_gcm3']

# Perform PCA
X = soil_df[features]
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
soil_df['PC1'] = principal_components[:, 0]
soil_df['PC2'] = principal_components[:, 1]

# Prepare kriging data
x = soil_df['x'].values.astype(float)
y = soil_df['y'].values.astype(float)
z = soil_df['health_index'].values.astype(float)

# Create grid
grid_x = np.linspace(0, 100, 50)
grid_y = np.linspace(0, 100, 50)

# Perform Ordinary Kriging
ok = OrdinaryKriging(
    x, y, z,
    variogram_model='spherical',
    verbose=False,
    enable_plotting=False
)
krige_grid, variance = ok.execute('grid', grid_x, grid_y)

# Create Plotly figure for kriging results
krige_fig = go.Figure(data=go.Heatmap(
    z=krige_grid,
    x=grid_x,
    y=grid_y,
    colorscale='Viridis',
    colorbar=dict(title='Health Index')
))
krige_fig.update_layout(
    title='Kriging Prediction',
    xaxis_title='X Coordinate (m)',
    yaxis_title='Y Coordinate (m)',
    height=600
)

# Create Plotly figure for kriging variance
krige_var = go.Figure(data=go.Heatmap(
    z=variance,
    x=grid_x,
    y=grid_y,
    colorscale='Reds',
    colorbar=dict(title='Health Index')
))
krige_var.update_layout(
    title='Kriging Variance',
    xaxis_title='X Coordinate (m)',
    yaxis_title='Y Coordinate (m)',
    height=600
)

# Create initial figures for the other plots
sample_fig = px.scatter(
    soil_df, 
    x='x', 
    y='y', 
    color='health_index',
    size='organic_matter_pct',
    hover_name='sample_id',
    hover_data=features,
    title="Field Sampling Locations",
    labels={
        'x': 'X Coordinate (m)',
        'y': 'Y Coordinate (m)',
        'health_index': 'Health Index',
        'organic_matter_pct': 'Organic Matter %'
    },
    color_continuous_scale='Viridis'
)
sample_fig.update_layout(
    coloraxis_colorbar=dict(title='Health Index'),
    title_x=0.5,
    height=500
)

pca_fig = px.scatter(
    soil_df, 
    x='PC1', 
    y='PC2', 
    color='organic_matter_pct',
    hover_name='sample_id',
    title="PCA Biplot",
    labels={
        'PC1': 'Principal Component 1 (Soil Fertility)',
        'PC2': 'Principal Component 2 (Nutrient Availability)',
        'organic_matter_pct': 'Organic Matter %'
    },
    color_continuous_scale='Viridis'
)
pca_fig.update_layout(title_x=0.5, height=500)

# Initialize Dash app
app = dash.Dash(__name__)

# CSS to prevent scrolling and stabilize layout
app.layout = html.Div([
    html.Div(style={
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'right': 0,
        'bottom': 0,
        'overflow': 'scroll',
        'padding': '20px',
        'backgroundColor': '#f8f9fa'
    }, children=[
        html.H1("Advanced Soil Analysis Dashboard", style={
            'textAlign': 'center', 
            'color': '#2c3e50',
            'marginBottom': '30px'
        }),
        
    
        # First row with two columns
        html.Div(style={
            'display': 'flex',
            'height': '90vh',  # Fixed height for top row
            'marginBottom': '20px',
            'gap': '20px'  # Space between columns
        }, children=[
            # Soil sample data
            html.Div(style={
                'flex': '1',
                'backgroundColor': 'white',
                'borderRadius': '8px',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                'padding': '15px',
                'overflow': 'hidden'
            }, children=[
                html.H3("Soil Sample Data", style={'textAlign': 'center'}),
                dcc.Graph(
                    id='sample-map',
                    figure=sample_fig,
                    style={'height': 'calc(100% - 40px)'}  # Fill available space
                )
            ]),
            
            # PCA results
            html.Div(style={
                'flex': '1',
                'backgroundColor': 'white',
                'borderRadius': '8px',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                'padding': '15px',
                'overflow': 'hidden'
            }, children=[
                html.H3("PCA Results", style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id='pca-color',
                    options=[{'label': col, 'value': col} for col in features],
                    value='organic_matter_pct',
                    clearable=False,
                    style={'marginBottom': '15px', 'zIndex': 100}  # Ensure dropdown is above
                ),
                dcc.Graph(
                    id='pca-plot',
                    figure=pca_fig,
                    style={'height': 'calc(100% - 70px)'}  # Account for header and dropdown
                )
            ])
        ]),
        
        # Spatial interpolation (full width)
        html.Div(style={
            'height': '100vh',  # Fixed height for bottom section
            'backgroundColor': 'white',
            'borderRadius': '8px',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
            'padding': '15px',
            'margin-bottom': '10px',
            'overflow': 'scroll'
        }, children=[
            html.H3("Spatial Interpolation", style={'textAlign': 'center'}),
            dcc.Graph(
                id='kriging-map',
                figure=krige_fig,
                style={'height': 'calc(100% - 60px)'}
            )
        ]),

        html.Div(style={
            'height': '100vh',  # Fixed height for bottom section
            'backgroundColor': 'white',
            'borderRadius': '8px',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
            'padding': '15px',
            'overflow': 'scroll'
        }, children=[
            html.H3("Kriging Variance", style={'textAlign': 'center'}),
            dcc.Graph(
                id='kriging-var',
                figure=krige_var,
                style={'height': 'calc(100% - 60px)'}
            )
        ])
    ])
])

@app.callback(
    Output('pca-plot', 'figure'),
    Input('pca-color', 'value')
)
def update_pca(color_var):
    fig = px.scatter(
        soil_df, 
        x='PC1', 
        y='PC2', 
        color=color_var,
        hover_name='sample_id',
        title="PCA Biplot",
        labels={
            'PC1': 'Principal Component 1 (Soil Fertility)',
            'PC2': 'Principal Component 2 (Nutrient Availability)',
            color_var: color_var.replace('_', ' ').title()
        },
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        title_x=0.5,
        height=500
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)