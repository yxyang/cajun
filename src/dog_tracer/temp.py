import numpy as np
import plotly.graph_objs as go
import plotly.express as px

# Generate some example data
timesteps = 100
t = np.linspace(0, 10, timesteps)
x = np.sin(t)
y = np.cos(t)

# Create a scatter plot for the point mass
scatter = go.Scatter(x=[x[0]], y=[y[0]], mode='markers', marker=dict(size=10))

# Create a line plot for the trajectory
line = go.Scatter(x=x[:1], y=y[:1], mode='lines')

# Create a slider component
slider = go.layout.Slider(steps=[dict(method="animate", args=[[f"frame_{i}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]) for i in range(timesteps)])

# Create the layout for the figure
layout = go.Layout(updatemenus=[dict(type="buttons", showactive=False,
                                     buttons=[dict(label="Play",
                                                   method="animate",
                                                   args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}])])],
                   sliders=[slider],
                   xaxis=dict(range=[-1.5, 1.5]),
                   yaxis=dict(range=[-1.5, 1.5]),
                   width=600, height=600)

# Create the frames for the animation
frames = [go.Frame(name=f"frame_{i}", data=[go.Scatter(x=x[:i+1], y=y[:i+1], mode='lines'),
                          go.Scatter(x=x[i:i+1], y=y[i:i+1], mode='markers', marker=dict(size=10))]) for i in range(1, timesteps)]

# Create the figure with the layout, data, and frames
fig = go.Figure(data=[line, scatter], layout=layout, frames=frames)

# Show the figure
fig.show()
