# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import base64
import os
import pickle

from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

from isaacgym.torch_utils import to_torch
import torch

app = Dash(__name__)

list(pio.templates)  # Available templates
pio.templates.default = "seaborn"


def load_data(content):
  # logs = pickle.load(open(data_dir, 'rb'))
  content_type, content_string = content.split(',')
  decoded = base64.b64decode(content_string)
  logs = pickle.loads(decoded)
  for frame in logs:
    for key in frame:
      if type(frame[key]) == torch.Tensor:
        frame[key] = frame[key].cpu().numpy()
        if frame[key].shape[0] != 1:
          raise ValueError("Only 1 robot is supported at this time.")
        frame[key] = frame[key][0]
  return logs


def create_upload_div():
  upload_div = html.Div(
      [
          html.H3("Upload Data"),
          dcc.Upload(
              id='upload_data',
              children=html.Div(['Drag and Drop or ',
                                 html.A('Select Files')]),
              style={
                  'width': '80%',
                  'height': '60px',
                  'lineHeight': '60px',
                  'borderWidth': '4px',
                  'borderStyle': 'dashed',
                  'borderRadius': '5px',
                  'textAlign': 'center',
                  'margin': '10px'
              },
              # Allow multiple files to be uploaded
              multiple=True),
          html.Button('Clear All Data',
                      id='clear_logs',
                      style={
                          'justifyContent': 'center',
                          'alignItems': 'center',
                          'margin': '10px'
                      })
      ],
      style={'width': '35%'},
  )
  selection_div = html.Div(
      [
          html.H3("Select Data to Plot"),
          dcc.Checklist(id="traj_name_selector", value=list(all_data.keys()))
      ],
      style={
          'width': '49%',
          'height': '10vw'
      },
  )
  return html.Div(
      [upload_div, selection_div],
      style={
          'width': '100vw',
          'display': 'flex',
      },
  )


@app.callback(Output('traj_name_selector', 'options', allow_duplicate=True),
              [Input('clear_logs', 'n_clicks')],
              prevent_initial_call=True)
def clear_data(_):
  global all_data
  all_data = {}
  return []


@app.callback(Output('traj_name_selector', 'options', allow_duplicate=True),
              Input('upload_data', 'contents'),
              State('upload_data', 'filename'),
              prevent_initial_call=True)
def file_upload_callback(list_of_contents, list_of_names):
  if list_of_contents is None:
    return []
  global all_data
  for name, content in zip(list_of_names, list_of_contents):
    filename = os.path.splitext(name)[0]
    all_data[filename] = load_data(content)
  return list(all_data.keys())


def generate_timeseries_plot(traj_names: str,
                             attr_name="base_velocity",
                             dim=0,
                             title="Base Velocity: X",
                             xlabel="Time/s",
                             ylabel="m/s"):
  fig = go.Figure()
  for traj_name in traj_names:
    data = all_data[traj_name]
    ts = np.array([frame["timestamp"] for frame in data])
    y = np.array([frame[attr_name][dim] for frame in data])
    fig.add_scatter(x=ts, y=y, name=traj_name)
  fig.update_layout(
      title=title,
      xaxis_title=xlabel,
      yaxis_title=ylabel,
      title_x=0.5,  # This centers the title
      title_y=1.,  # This adjusts the vertical position of the title
      margin=dict(t=20, b=40, l=20, r=20),
      legend=dict(
          x=0,  # This sets the horizontal position of the legend
          y=1,  # This sets the vertical position of the legend
          bordercolor='rgba(255, 255, 255, 0.)',
          bgcolor='rgba(255, 255, 255, 0.5)',
          borderwidth=1),
      paper_bgcolor='rgba(255, 255, 255, 1.)',
  )
  return fig


def create_base_velocity_div():
  base_vel_div = html.Div(
      children=[
          html.H3('Base Velocity', style={'height': '4px'}),
          html.Div(
              children=[
                  dcc.Graph(id="base_vel_x", className="plot"),
                  dcc.Graph(id="base_vel_y", className='plot'),
                  dcc.Graph(id="base_vel_z", className='plot')
              ],
              className='container',
          )
      ],
      style={
          'width': '100vw',
          'height': '20vh',
          # 'overflow': 'hidden'
      },
  )
  return base_vel_div


@app.callback([
    Output('base_vel_x', 'figure'),
    Output('base_vel_y', 'figure'),
    Output('base_vel_z', 'figure')
], [Input('traj_name_selector', 'value')])
def update_base_velocity_figs(selected_traj_names):
  if selected_traj_names is None:
    selected_traj_names = []
  vel_x_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="base_velocity",
                                       dim=0,
                                       title="X")
  vel_y_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="base_velocity",
                                       dim=1,
                                       title="Y")
  vel_z_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="base_velocity",
                                       dim=2,
                                       title="Z")
  return vel_x_fig, vel_y_fig, vel_z_fig


def create_base_position_div():
  base_pos_div = html.Div(
      children=[
          html.H3('Base Position', style={'height': '4px'}),
          html.Div(
              children=[
                  dcc.Graph(id="base_pos_xy", className="plot"),
                  dcc.Graph(id="base_pos_z", className='plot')
              ],
              className='container',
          )
      ],
      style={
          'width': '100vw',
          'height': '20vh',
          # 'overflow': 'hidden'
      },
  )
  return base_pos_div


@app.callback(
    [Output('base_pos_xy', 'figure'),
     Output('base_pos_z', 'figure')], [Input('traj_name_selector', 'value')])
def update_base_position_figs(selected_traj_names):
  if selected_traj_names is None:
    selected_traj_names = []

  scatters = []
  for traj_name in selected_traj_names:
    data = all_data[traj_name]
    base_pos = np.array([frame['base_position'] for frame in data])
    scatters.append(
        go.Scatter3d(x=base_pos[:, 0],
                     y=base_pos[:, 1],
                     z=base_pos[:, 2],
                     name=traj_name,
                     mode='lines'))

  layout = go.Layout(
      title_text="Space Trajectory",
      title_x=0.5,  # This centers the title
      title_y=1.,  # This adjusts the vertical position of the title
      margin=dict(t=20, b=40, l=20, r=20),
      legend=dict(
          x=0,  # This sets the horizontal position of the legend
          y=1,  # This sets the vertical position of the legend
          bgcolor='rgba(255, 255, 255, 0.5)',
          bordercolor='rgba(0, 0, 0, 0.1)',
          borderwidth=1),
      paper_bgcolor='rgba(255, 255, 255, 1.)',
      scene=dict(
          xaxis_title='x',
          yaxis_title='y',
          zaxis_title='z',
          aspectmode='manual',
          aspectratio=dict(x=1, y=1, z=1.),
      ))
  pos_fig = go.Figure(data=scatters, layout=layout)
  # pos_fig.update_layout()
  # pos_fig.update_yaxes(
  #     scaleanchor="x",
  #     scaleratio=1,
  # )
  # pos_fig.update_zaxes(
  #     scaleanchor="x",
  #     scaleratio=1,
  # )

  pos_z_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="base_position",
                                       dim=2,
                                       title="Z",
                                       ylabel="Height/m")
  return pos_fig, pos_z_fig


def create_base_position_div():
  base_pos_div = html.Div(
      children=[
          html.H3('Base Position', style={'height': '4px'}),
          html.Div(
              children=[
                  dcc.Graph(id="base_pos_xy", className="plot"),
                  dcc.Graph(id="base_pos_z", className='plot')
              ],
              className='container',
          )
      ],
      style={
          'width': '100vw',
          'height': '20vh',
          # 'overflow': 'hidden'
      },
  )
  return base_pos_div


def create_base_orientation_div():
  base_vel_div = html.Div(
      children=[
          html.H3('Base Orientation', style={'height': '4px'}),
          html.Div(
              children=[
                  dcc.Graph(id="base_roll", className="plot"),
                  dcc.Graph(id="base_pitch", className='plot'),
                  dcc.Graph(id="base_yaw", className='plot')
              ],
              className='container',
          )
      ],
      style={
          'width': '100vw',
          'height': '20vh',
          # 'overflow': 'hidden'
      },
  )
  return base_vel_div


@app.callback([
    Output('base_roll', 'figure'),
    Output('base_pitch', 'figure'),
    Output('base_yaw', 'figure')
], [Input('traj_name_selector', 'value')])
def update_base_orientation_figs(selected_traj_names):
  if selected_traj_names is None:
    selected_traj_names = []
  vel_x_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="base_orientation_rpy",
                                       dim=0,
                                       title="Roll",
                                       ylabel="rad")
  vel_y_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="base_orientation_rpy",
                                       dim=1,
                                       title="Pitch",
                                       ylabel="rad")
  vel_z_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="base_orientation_rpy",
                                       dim=2,
                                       title="Yaw",
                                       ylabel="rad")
  return vel_x_fig, vel_y_fig, vel_z_fig


def create_base_angvel_div():
  base_vel_div = html.Div(
      children=[
          html.H3('Base Angular Velocity', style={'height': '4px'}),
          html.Div(
              children=[
                  dcc.Graph(id="base_angvel_x", className="plot"),
                  dcc.Graph(id="base_angvel_y", className='plot'),
                  dcc.Graph(id="base_angvel_z", className='plot')
              ],
              className='container',
          )
      ],
      style={
          'width': '100vw',
          'height': '20vh',
          # 'overflow': 'hidden'
      },
  )
  return base_vel_div


@app.callback([
    Output('base_angvel_x', 'figure'),
    Output('base_angvel_y', 'figure'),
    Output('base_angvel_z', 'figure')
], [Input('traj_name_selector', 'value')])
def update_base_angvel_figs(selected_traj_names):
  if selected_traj_names is None:
    selected_traj_names = []
  vel_x_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="base_angular_velocity",
                                       dim=0,
                                       title="X",
                                       ylabel="rad/s")
  vel_y_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="base_angular_velocity",
                                       dim=1,
                                       title="Y",
                                       ylabel="rad/s")
  vel_z_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="base_angular_velocity",
                                       dim=2,
                                       title="Z",
                                       ylabel="rad/s")
  return vel_x_fig, vel_y_fig, vel_z_fig


def create_foot_contact_div():
  foot_contact_div = html.Div(
      children=[
          html.H3('Foot Contact', style={'height': '4px'}),
          html.Div(
              children=[
                  dcc.Graph(id="foot_contact", className="plot"),
              ],
              className='container',
          )
      ],
      style={
          'width': '100vw',
          'height': '20vh',
          # 'overflow': 'hidden'
      },
  )
  return foot_contact_div


@app.callback([
    Output('foot_contact', 'figure'),
], [Input('traj_name_selector', 'value')])
def update_base_foot_contact_figs(selected_traj_names):
  if selected_traj_names is None:
    selected_traj_names = []

  active_template = pio.templates[pio.templates.default]
  # Get the color sequence from the active template
  color_sequence = active_template.layout.colorway

  plots = []
  for idx, traj_name in enumerate(selected_traj_names):
    data = all_data[traj_name]
    ts = np.array([frame['timestamp'] for frame in data])
    foot_contact = np.array([frame['foot_contact_state'] for frame in data])
    t1 = go.Scatter(
        x=ts,
        y=foot_contact[:, 0] * 0.8 + 3,
        mode='lines',
        name=traj_name,
        legendgroup=traj_name,
        # legendgrouptitle_text=traj_name,
        line=dict(color=color_sequence[idx]))
    t2 = go.Scatter(x=ts,
                    y=foot_contact[:, 1] * 0.8 + 2,
                    mode='lines',
                    name="FL",
                    legendgroup=traj_name,
                    line=dict(color=color_sequence[idx]),
                    showlegend=False)
    t3 = go.Scatter(x=ts,
                    y=foot_contact[:, 2] * 0.8 + 1,
                    mode='lines',
                    name="RR",
                    legendgroup=traj_name,
                    line=dict(color=color_sequence[idx]),
                    showlegend=False)
    t4 = go.Scatter(x=ts,
                    y=foot_contact[:, 3] * 0.8 + 0,
                    mode='lines',
                    name="RL",
                    legendgroup=traj_name,
                    line=dict(color=color_sequence[idx]),
                    showlegend=False)
    plots.extend([t1, t2, t3, t4])

  foot_contact_fig = go.Figure(data=plots)
  foot_contact_fig.update_layout(
      title="Foot Contact",
      xaxis_title="time/s",
      yaxis_title="",
      title_x=0.5,  # This centers the title
      title_y=1.,  # This adjusts the vertical position of the title
      margin=dict(t=20, b=40, l=20, r=20),
      legend=dict(
          x=0,  # This sets the horizontal position of the legend
          y=1,  # This sets the vertical position of the legend
          bgcolor='rgba(255, 255, 255, 0.5)',
          bordercolor='rgba(0, 0, 0, 0.1)',
          borderwidth=1),
      paper_bgcolor='rgba(255, 255, 255, 1.)',
  )
  return [foot_contact_fig]


def create_desired_acc_div():
  desired_acc_div = html.Div(
      children=[
          html.H3('Desired Base Acceleration', style={'height': '4px'}),
          html.Div(
              children=[
                  dcc.Graph(id="desired_acc_lin_x", className="plot"),
                  dcc.Graph(id="desired_acc_lin_y", className='plot'),
                  dcc.Graph(id="desired_acc_lin_z", className='plot'),
              ],
              className='container',
              style={
                  'width': '100vw',
                  'height': '50%',
                  # 'overflow': 'hidden'
              }),
          html.Div(
              children=[
                  dcc.Graph(id="desired_acc_ang_x", className="plot"),
                  dcc.Graph(id="desired_acc_ang_y", className='plot'),
                  dcc.Graph(id="desired_acc_ang_z", className='plot'),
              ],
              className='container',
              style={
                  'width': '100vw',
                  'height': '50%',
                  # 'overflow': 'hidden'
              })
      ],
      style={
          'width': '100vw',
          'height': '40vh',
          # 'overflow': 'hidden'
      },
  )
  return desired_acc_div


@app.callback([
    Output('desired_acc_lin_x', 'figure'),
    Output('desired_acc_lin_y', 'figure'),
    Output('desired_acc_lin_z', 'figure'),
    Output('desired_acc_ang_x', 'figure'),
    Output('desired_acc_ang_y', 'figure'),
    Output('desired_acc_ang_z', 'figure'),
], [Input('traj_name_selector', 'value')])
def update_desired_acc_figs(selected_traj_names):
  if selected_traj_names is None:
    selected_traj_names = []
  acc_lin_x_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="desired_acc_body_frame",
                                           dim=0,
                                           title="Lin X",
                                           ylabel="m/s^2")
  acc_lin_y_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="desired_acc_body_frame",
                                           dim=1,
                                           title="Lin Y",
                                           ylabel="m/s^2")
  acc_lin_z_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="desired_acc_body_frame",
                                           dim=2,
                                           title="Lin Z",
                                           ylabel="m/s^2")
  acc_ang_x_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="desired_acc_body_frame",
                                           dim=3,
                                           title="Ang X",
                                           ylabel="Rad/s^2")
  acc_ang_y_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="desired_acc_body_frame",
                                           dim=4,
                                           title="Ang Y",
                                           ylabel="Rad/s^2")
  acc_ang_z_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="desired_acc_body_frame",
                                           dim=5,
                                           title="Ang Z",
                                           ylabel="Rad/s^2")
  return (
      acc_lin_x_fig,
      acc_lin_y_fig,
      acc_lin_z_fig,
      acc_ang_x_fig,
      acc_ang_y_fig,
      acc_ang_z_fig,
  )


def create_solved_acc_div():
  desired_acc_div = html.Div(
      children=[
          html.H3('Solved Base Acceleration', style={'height': '4px'}),
          html.Div(
              children=[
                  dcc.Graph(id="solved_acc_lin_x", className="plot"),
                  dcc.Graph(id="solved_acc_lin_y", className='plot'),
                  dcc.Graph(id="solved_acc_lin_z", className='plot')
              ],
              className='container',
              style={
                  'width': '100vw',
                  'height': '50%',
                  # 'overflow': 'hidden'
              }),
          html.Div(
              children=[
                  dcc.Graph(id="solved_acc_ang_x", className="plot"),
                  dcc.Graph(id="solved_acc_ang_y", className='plot'),
                  dcc.Graph(id="solved_acc_ang_z", className='plot')
              ],
              className='container',
              style={
                  'width': '100vw',
                  'height': '50%',
                  # 'overflow': 'hidden'
              })
      ],
      style={
          'width': '100vw',
          'height': '40vh',
          # 'overflow': 'hidden'
      },
  )
  return desired_acc_div


@app.callback([
    Output('solved_acc_lin_x', 'figure'),
    Output('solved_acc_lin_y', 'figure'),
    Output('solved_acc_lin_z', 'figure'),
    Output('solved_acc_ang_x', 'figure'),
    Output('solved_acc_ang_y', 'figure'),
    Output('solved_acc_ang_z', 'figure')
], [Input('traj_name_selector', 'value')])
def update_solved_acc_figs(selected_traj_names):
  if selected_traj_names is None:
    selected_traj_names = []
  acc_lin_x_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="solved_acc_body_frame",
                                           dim=0,
                                           title="Lin X",
                                           ylabel="m/s^2")
  acc_lin_y_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="solved_acc_body_frame",
                                           dim=1,
                                           title="Lin Y",
                                           ylabel="m/s^2")
  acc_lin_z_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="solved_acc_body_frame",
                                           dim=2,
                                           title="Lin Z",
                                           ylabel="m/s^2")
  acc_ang_x_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="solved_acc_body_frame",
                                           dim=3,
                                           title="Ang X",
                                           ylabel="m/s^2")
  acc_ang_y_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="solved_acc_body_frame",
                                           dim=4,
                                           title="Ang Y",
                                           ylabel="m/s^2")
  acc_ang_z_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="solved_acc_body_frame",
                                           dim=5,
                                           title="Ang Z",
                                           ylabel="m/s^2")
  return (
      acc_lin_x_fig,
      acc_lin_y_fig,
      acc_lin_z_fig,
      acc_ang_x_fig,
      acc_ang_y_fig,
      acc_ang_z_fig,
  )


def create_app(app):
  app.layout = html.Div(
      children=[
          html.H1(children='Dog Tracer', style={'textAlign': 'center'}),
          create_upload_div(),
          create_base_position_div(),
          create_base_velocity_div(),
          create_base_orientation_div(),
          create_base_angvel_div(),
          create_foot_contact_div(),
          create_desired_acc_div(),
          create_solved_acc_div()
      ],
      style={
          'width': '100vw',
          # 'height': '100vh',
          # 'overflow': 'hidden'
      },
  )

  # app.update_layout()  # This line is required to correctly render the CSS styles
  app.css.append_css({'external_url': 'app.css'})
  return app


if __name__ == '__main__':
  all_data = {}
  create_app(app)
  app.run_server(debug=False)
