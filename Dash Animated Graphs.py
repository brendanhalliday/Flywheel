import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import random
import plotly.express as px

# initial values. Populate the dataframe with the starting point of your data
df = pd.DataFrame({
    'x': [0],
    'y': [0]})

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the application
app.layout = html.Div(style={'backgroundColor':'#141414', 'color':'#E8E8E8'}, children = [
    #create the graph object. the 'id' should match the 'Output' part of the callback. this output is the function's output
    dcc.Graph(id='animated-graph'),
    #define the updater. This object tells dash to update the graph every 'interval' duration of time. every time an 'interval' amount of time elapses, n_intervals increases by 1. 
    #Since ;n_intervals' is the input for our function, every time this changes, the function is called upon in order to create a new output graph. we do not care about the actual value of 'n_intervals' we only care that each time it changes, it makes our function get called up and create a new graph. 
    dcc.Interval(
        id='iterator',
        interval=90, # 100 is stable and consistent
        #how many intervals we should start at. Should be kept at 0 for the purposes of specifying how many iterations we want our function to go through later.
        n_intervals=0
    )
])

# Define callback to update the graph
@app.callback(
    #this output is what the function creates each time 'interval' amount of time passes
    Output('animated-graph', 'figure'),
    #this is the input for our function (in particular, n_intervals. it tells out function to draw its input 'n_intervals' through the 'iterator' object). it increases by 1 every time 'interval' amount of time passes, calling our function.
    [Input('iterator', 'n_intervals')]
)

#callback function. This creates all the obects that will go in the website. It takes inputs from the input objects defined in the '@app.callback' thing above, and it returns outputs which go into the '@app.callback' thing. Then this callback thing puts them in the website's objects (the are linked through the 'id=' name.)
def update_graph(n_intervals):

    #define how many iterations we want this program to go through. 
    if n_intervals <=500:
        # every time the function is called, it automatically creates the desired new data value corresponding to the given iteration.
        update = {
            'x': df['x'].iloc[-1] + 0.01,
            'y': np.sin(np.power(df['x'].iloc[-1] + 0.01, 2))
        }

        #this appends the above value to our original dataframe defined in the begining of the script
        df.loc[len(df)] = update

        #this figure object is our actual line. (plotly.express.line for documentation)
        fig = px.line(df, 'x', 'y')

        # Customize layout (self explanatory)
        fig.update_layout({"plot_bgcolor":'#141414', 'paper_bgcolor': "rgba(0,0,0,0)"},
            title='Animated Data',
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False, visible=False),
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            yaxis_range = [-3, 3],
            xaxis_range = [0, 7],
            font=dict(color='#E8E8E8')
        )
        return fig
    
    #stop updating the graph when we reach our required iterations
    else:
        return dash.no_update

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
