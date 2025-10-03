import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
from train_env import TrainTrafficEnv
from stable_baselines3 import PPO
import os
import time

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

# --- Load the environment and the trained RL model ---
env = TrainTrafficEnv()
state, info = env.reset()

# --- Load the trained RL agent ---
MODEL_PATH = "models/PPO/200000.zip"
if os.path.exists(MODEL_PATH):
    print("Loading trained RL agent...")
    rl_agent = PPO.load(MODEL_PATH, env=env)
else:
    print("WARNING: Trained RL model not found. Using dummy agent.")
    rl_agent = None

STATIONS_LIST = [
    "Aluva", "Pulincha", "Companypady", "Ambattukavu", "Muttom", "Kalamassery",
    "Cochin University", "Pathadipalam", "Edapally", "Changampuzha Park",
    "Palarivattom", "JLN Stadium", "Kaloor", "Town Hall", "MG Road",
    "Maharajas College", "Ernakulam South", "Kadavanthra", "Elamkulam",
    "Vyttila", "Thykoodam", "Pettah", "Vadakkekotta", "SN Junction"
]
POSITIONS = {"DEPOT_1": (-1, 0.5), "DEPOT_2": (len(STATIONS_LIST), 0.5)}
for i, station_name in enumerate(STATIONS_LIST):
    POSITIONS[station_name] = (i, 0)

WHATIF_RESULTS = {}

app.layout = html.Div(style={'backgroundColor': '#222831', 'color': '#EEEEEE', 'fontFamily': 'Roboto, sans-serif'}, children=[
    dcc.Store(id='historical-kpi-store', data={'time': [], 'passengers': [], 'headway': []}),
    
    html.Div(style={'backgroundColor': '#00ADB5', 'padding': '15px', 'textAlign': 'center'}, children=[
        html.H2("KMRL AI Control Center", style={'color': 'white', 'margin': '0px'}),
        html.P("Smart Train Induction Planning & Scheduling System (Dual Depot Ready)", style={'color': '#EEEEEE', 'fontSize': '14px', 'margin': '5px 0 0 0'})
    ]),
    
    html.Div(style={'backgroundColor': '#393E46', 'padding': '10px 20px', 'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}, children=[
        html.Div([
            html.Span("Current Time: ", style={'fontSize': '16px', 'marginRight': '5px'}),
            html.Strong(id='current-time-display', style={'color': '#FFD700'}),
            html.Span(" | Forecast Multiplier: ", style={'fontSize': '16px', 'marginLeft': '20px', 'marginRight': '5px'}),
            html.Strong(id='forecast-multiplier-display', style={'color': '#00ADB5'}),
        ]),
        html.Div([
            html.Button('PAUSE', id='pause-button', n_clicks=0, style={'backgroundColor': '#FF6347', 'color': 'white', 'marginRight': '5px', 'border': 'none', 'padding': '10px 15px', 'cursor': 'pointer'}),
            html.Button('PLAY', id='play-button', n_clicks=1, style={'backgroundColor': '#32CD32', 'color': 'white', 'marginRight': '5px', 'border': 'none', 'padding': '10px 15px', 'cursor': 'pointer'}),
            html.Button('FFWD (1 Day)', id='ffwd-button', n_clicks=0, style={'backgroundColor': '#FFD700', 'color': 'black', 'border': 'none', 'padding': '10px 15px', 'cursor': 'pointer'}),
        ])
    ]),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0, disabled=False), # Slower default interval
    
    html.Div(style={'display': 'flex', 'justifyContent': 'space-around', 'padding': '20px'}, children=[
        html.Div(className="three columns card", style={'backgroundColor': '#393E46', 'padding': '15px', 'borderRadius': '8px', 'margin': '0 10px', 'textAlign': 'center'}, children=[html.P("Active Trains", style={'fontSize': '16px', 'color': '#00ADB5'}), html.H3(id='active-trains-card')]),
        html.Div(className="three columns card", style={'backgroundColor': '#393E46', 'padding': '15px', 'borderRadius': '8px', 'margin': '0 10px', 'textAlign': 'center'}, children=[html.P("Standby Trains", style={'fontSize': '16px', 'color': '#00ADB5'}), html.H3(id='standby-trains-card')]),
        html.Div(className="three columns card", style={'backgroundColor': '#393E46', 'padding': '15px', 'borderRadius': '8px', 'margin': '0 10px', 'textAlign': 'center'}, children=[html.P("IBL (Maintenance)", style={'fontSize': '16px', 'color': '#FF6347'}), html.H3(id='ibl-trains-card')]),
        html.Div(className="three columns card", style={'backgroundColor': '#393E46', 'padding': '15px', 'borderRadius': '8px', 'margin': '0 10px', 'textAlign': 'center'}, children=[html.P("Current Headway", style={'fontSize': '16px', 'color': '#00ADB5'}), html.H3(id='headway-card')]),
        html.Div(className="three columns card", style={'backgroundColor': '#393E46', 'padding': '15px', 'borderRadius': '8px', 'margin': '0 10px', 'textAlign': 'center'}, children=[html.P("System Status", style={'fontSize': '16px', 'color': '#00ADB5'}), html.H3(id='status-card')]),
        html.Div(className="three columns card", style={'backgroundColor': '#393E46', 'padding': '15px', 'borderRadius': '8px', 'margin': '0 10px', 'textAlign': 'center'}, children=[html.P("Total Waiting Psg", style={'fontSize': '16px', 'color': '#00ADB5'}), html.H3(id='total-passengers-card')]),
    ]),
    html.Div(style={'display': 'flex', 'padding': '0 20px 20px 20px'}, children=[
        html.Div(style={'flex': '65%', 'backgroundColor': '#393E46', 'borderRadius': '8px', 'padding': '15px', 'marginRight': '20px'}, children=[
            html.H4("Live Metro Map - Real-Time Crowd & Train Positions", style={'color': '#00ADB5'}),
            dcc.Graph(id='network-graph', style={'height': '400px'}, config={'scrollZoom': True, 'displaylogo': False}), # TOOLBAR ENABLED
            html.H4("System Performance Over Time", style={'color': '#00ADB5', 'marginTop': '20px'}),
            dcc.Graph(id='kpi-trend-chart', style={'height': '250px'}),
            html.H5("Service Audit Trail (Last 7 Days)", style={'color': '#EEEEEE', 'marginTop': '15px'}),
            html.Div(id='audit-log-table')
        ]),
        html.Div(style={'flex': '35%', 'backgroundColor': '#393E46', 'borderRadius': '8px', 'padding': '15px'}, children=[
            html.H4("Fleet Induction & Analytics (XAI)", style={'color': '#00ADB5'}),
            html.Div(style={'backgroundColor': '#2D4059', 'padding': '10px', 'borderRadius': '6px', 'marginBottom': '10px'}, children=[
                html.P("Resilience Test: Inject Scenario", style={'fontSize': '14px', 'color': '#FF6347', 'margin': '0 0 5px 0'}),
                dcc.Dropdown(id='scenario-dropdown', options=[
                        {'label': 'Train Failure (Random Active Train)', 'value': 'train_failure'},
                        {'label': 'Track Blockage (Edapally <-> Palarivattom)', 'value': 'track_blockage'},
                        {'label': 'Passenger Surge (JLN Stadium)', 'value': 'passenger_surge'},
                        {'label': 'Clear All Disruptions', 'value': 'clear_all'}],
                    placeholder="Select a disruption scenario...", style={'marginBottom': '5px'}),
                html.Button('INJECT SCENARIO', id='inject-scenario-button', n_clicks=0, style={'width': '100%', 'backgroundColor': '#FF6347', 'color': 'white', 'cursor':'pointer'})
            ]),
            html.Div(style={'backgroundColor': '#2D4059', 'padding': '10px', 'borderRadius': '6px', 'marginBottom': '10px'}, children=[
                html.P("AI Plan Performance (Predicted 24h)", style={'fontSize': '14px', 'color': '#00ADB5', 'margin': '0'}),
                html.Div(id='ai-performance-summary', style={'display': 'flex', 'justifyContent': 'space-between', 'fontSize': '12px', 'marginTop': '5px'})
            ]),
            html.Div(style={'backgroundColor': '#2D4059', 'padding': '10px', 'borderRadius': '6px', 'marginBottom': '10px'}, children=[
                html.P("What-If Simulation (Manual Override)", style={'fontSize': '14px', 'color': '#FFD700', 'margin': '0'}),
                html.Div(style={'display': 'flex'}, children=[
                    html.Div(style={'flex': '70%'}, children=[
                        dcc.Checklist(id='manual-induction-checklist', value=env.induction_plan, inline=False,
                            style={'color': '#EEEEEE', 'fontSize': '12px', 'maxHeight': '100px', 'overflowY': 'auto', 'padding': '5px'},
                            labelStyle={'display': 'block'})
                    ]),
                    html.Div(style={'flex': '30%', 'textAlign': 'right', 'marginLeft': '5px'}, children=[
                        html.Button('RUN WHAT-IF', id='run-whatif-button', n_clicks=0, style={'backgroundColor': '#00ADB5', 'color': 'white', 'border': 'none', 'padding': '5px', 'borderRadius': '4px', 'cursor': 'pointer', 'width': '100%'}),
                        html.Div(id='whatif-performance-summary', style={'fontSize': '11px', 'marginTop': '5px', 'color': '#32CD32'})
                    ])
                ])
            ]),
            html.Div(id='station-table-div', style={'maxHeight': '200px', 'overflowY': 'auto'})
        ])
    ])
])

@app.callback(Output('interval-component', 'disabled'), [Input('pause-button', 'n_clicks'), Input('play-button', 'n_clicks')], [State('interval-component', 'disabled')])
def toggle_interval(pause_n, play_n, is_disabled):
    ctx = dash.callback_context
    if not ctx.triggered: return is_disabled
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'pause-button': return True
    return False

@app.callback(Output('inject-scenario-button', 'n_clicks'), [Input('inject-scenario-button', 'n_clicks')], [State('scenario-dropdown', 'value')], prevent_initial_call=True)
def handle_scenario_injection(n_clicks, scenario):
    if n_clicks > 0 and scenario:
        env.inject_scenario(scenario)
    return 0

@app.callback(
    [Output('active-trains-card', 'children'), Output('standby-trains-card', 'children'),
     Output('ibl-trains-card', 'children'), Output('headway-card', 'children'),
     Output('status-card', 'children'), Output('total-passengers-card', 'children'),
     Output('network-graph', 'figure'), Output('station-table-div', 'children'),
     Output('ai-performance-summary', 'children'),
     Output('whatif-performance-summary', 'children'),
     Output('manual-induction-checklist', 'options'),
     Output('manual-induction-checklist', 'value'),
     Output('current-time-display', 'children'),
     Output('forecast-multiplier-display', 'children'),
     Output('audit-log-table', 'children'),
     Output('historical-kpi-store', 'data')],
    [Input('interval-component', 'n_intervals'),
     Input('run-whatif-button', 'n_clicks'),
     Input('ffwd-button', 'n_clicks')],
    [State('manual-induction-checklist', 'value'),
     State('historical-kpi-store', 'data')]
)
def update_view(n, n_clicks_whatif, n_clicks_ffwd, manual_induction_list, kpi_data):
    global state, info, WHATIF_RESULTS
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    terminated = False

    # --- FFWD FIX & SIMULATION LOGIC ---
    if triggered_id == 'ffwd-button':
        # Non-blocking loop for fast-forward
        end_time = info.get('current_time', 0) + 1440 # 1 day in minutes
        while info.get('current_time', 0) < end_time and not terminated:
            if rl_agent:
                action, _ = rl_agent.predict(state, deterministic=True)
            else:
                action = 0 # Default action
            state, _, terminated, _, info = env.step(action)
    elif triggered_id == 'interval-component':
        # Normal single step for interval
        if rl_agent:
            action, _ = rl_agent.predict(state, deterministic=True)
        else:
            action = 0
        state, _, terminated, _, info = env.step(action)
    
    if terminated:
        state, info = env.reset()
        WHATIF_RESULTS = {}
        kpi_data = {'time': [], 'passengers': [], 'headway': []}
    
    if triggered_id == 'run-whatif-button':
        valid_manual_list = [tid for tid in manual_induction_list if env.trains.get(tid) and env._check_hard_constraints(env.trains.get(tid)) == "READY"]
        WHATIF_RESULTS = env._simulate_day_performance(valid_manual_list)

    # --- UI UPDATE LOGIC ---
    active_trains_count = info.get('active_trains_count', 0)
    standby_trains_count = info.get('standby_trains_count', 0)
    ibl_trains_count = info.get('ibl_trains_count', 0)
    headway_text = f"{info.get('current_headway_min', 0.0):.1f} min"
    status_text = info.get('system_status', 'UNKNOWN')
    total_passengers_waiting = info.get('total_passengers_waiting', 0)
    status_color = '#FF6347' if status_text == 'CRITICAL' else '#FFD700' if status_text == 'ALERT' else '#32CD32'
    
    current_time_min = int(info.get('current_time', 0))
    time_display = f"{current_time_min // 60:02d}:{current_time_min % 60:02d} IST"
    forecast_display = f"x{info.get('demand_forecast', 1.0):.2f}"

    if triggered_id == 'interval-component' or triggered_id == 'ffwd-button':
        kpi_data['time'].append(time_display)
        kpi_data['passengers'].append(total_passengers_waiting)
        kpi_data['headway'].append(info.get('current_headway_min', 0.0))
        
    ai_perf = info.get('ai_plan_performance', {})
    perf_summary = [html.Span(f"Cost Penalty: {ai_perf.get('predicted_maintenance_cost', 0.0):.2f}", style={'color': '#FFD700', 'marginRight': '20px'}),
                    html.Span(f"Readiness: {ai_perf.get('predicted_service_readiness', 0.0)*100:.0f}%", style={'color': '#32CD32'})]

    whatif_summary = []
    if WHATIF_RESULTS:
        w_cost = WHATIF_RESULTS.get('predicted_maintenance_cost', 0)
        w_ready = WHATIF_RESULTS.get('predicted_service_readiness', 0)
        ai_cost = ai_perf.get('predicted_maintenance_cost', 0.1)
        ai_ready = ai_perf.get('predicted_service_readiness', 0.0)
        w_cost_color = '#FF6347' if w_cost > ai_cost * 1.05 else '#32CD32'
        w_readiness_color = '#FF6347' if w_ready < ai_ready * 0.95 else '#32CD32'
        whatif_summary = [html.Span(f"Cost: {w_cost:.2f} ", style={'color': w_cost_color, 'marginRight': '5px', 'fontWeight': 'bold'}),
                          html.Span(f"Ready: {w_ready*100:.0f}%", style={'color': w_readiness_color, 'fontWeight': 'bold'})]
    
    checklist_options = []
    for t in env.trains.values():
        is_ready = env._check_hard_constraints(t) == "READY"
        label_style = {'textDecoration': 'line-through', 'color': '#FF6347'} if not is_ready else {}
        checklist_options.append({'label': html.Span(f' Train {t.id} ({t.bay_id})', style=label_style), 'value': t.id, 'disabled': not is_ready})

    checklist_value = manual_induction_list
    if terminated or (info.get('current_time', 0) >= 1260 and (info.get('current_time', 0) - env.time_step_seconds / 60) < 1260):
        checklist_value = info.get('induction_plan', [])

    # --- ENHANCED NETWORK VISUALIZATION ---
    G = info.get('rail_network', env.rail_network)
    
    # DUAL TRACK VISUALIZATION
    up_track_y, down_track_y = 0.05, -0.05
    edge_traces = [
        go.Scatter(x=[], y=[], line=dict(width=1, color='#555555'), mode='lines', hoverinfo='none'), # UP Track
        go.Scatter(x=[], y=[], line=dict(width=1, color='#555555'), mode='lines', hoverinfo='none')  # DOWN Track
    ]
    for u, v in env.original_rail_network.edges():
        if u in POSITIONS and v in POSITIONS:
            edge_traces[0]['x'] += (POSITIONS[u][0], POSITIONS[v][0], None)
            edge_traces[0]['y'] += (up_track_y, up_track_y, None)
            edge_traces[1]['x'] += (POSITIONS[u][0], POSITIONS[v][0], None)
            edge_traces[1]['y'] += (down_track_y, down_track_y, None)

    node_traces = []
    for station_name in STATIONS_LIST:
        passengers_waiting = info['stations'].get(station_name, {}).get('passengers_waiting', 0)
        color = '#FF6347' if passengers_waiting > 200 else '#FFD700' if passengers_waiting > 50 else '#32CD32'
        # STRAIGHT ALIGNMENT
        node_traces.append(go.Scatter(x=[POSITIONS[station_name][0]], y=[0], mode='markers+text',
                                      text=f"<b>{station_name}</b><br>{passengers_waiting} psg", textposition="bottom center",
                                      textfont=dict(size=10),
                                      marker=dict(size=12, color=color, line=dict(width=1, color='#EEEEEE'))))

    node_traces.append(go.Scatter(x=[POSITIONS["DEPOT_1"][0]], y=[POSITIONS["DEPOT_1"][1]], mode='markers+text', text=f"<b>DEPOT 1</b><br>IBL: {ibl_trains_count}", textposition="top center", marker=dict(size=20, color='#00ADB5')))
    node_traces.append(go.Scatter(x=[POSITIONS["DEPOT_2"][0]], y=[POSITIONS["DEPOT_2"][1]], mode='markers+text', text=f"<b>DEPOT 2</b><br>Standby: {standby_trains_count}", textposition="top center", marker=dict(size=20, color='#00ADB5')))

    train_traces = []
    for train_id, train_data in info.get("trains", {}).items():
        if train_data['state'] not in ['in_depot', 'standby', 'IBL']:
            x, y_base = POSITIONS.get(train_data['current_node'], (0,0))
            
            # Place train on correct track based on direction
            y_track = up_track_y if train_data['direction'] == 'forward' else down_track_y

            if train_data['state'] == 'moving' and train_data['path'] and len(train_data['path']) > 1:
                u, v = train_data['path'][0], train_data['path'][1]
                if u in POSITIONS and v in POSITIONS:
                    edge_len = env.original_rail_network.get_edge_data(u, v)['weight']
                    progress = train_data['distance_on_edge'] / edge_len if edge_len > 0 else 0
                    x0, x1 = POSITIONS[u][0], POSITIONS[v][0]
                    x = x0 + (x1 - x0) * progress
            
            if train_data['state'] == 'failed':
                 train_traces.append(go.Scatter(x=[x], y=[y_track], mode='text', text="⚠️", textfont=dict(size=30, color='red'), hoverinfo='text', hovertext=f"Train {train_id} FAILED"))
            else:
                train_color = '#FFD700' if train_data.get('is_induction_plan') else '#32CD32'
                hover_text = f"Train {train_id}<br>Pass: {len(train_data['passengers'])}<br>Mileage: {train_data['total_mileage_km']:.0f} km"
                train_traces.append(go.Scatter(x=[x], y=[y_track], mode='text', text="🚆", textfont=dict(size=24, color=train_color), hoverinfo='text', hovertext=hover_text))
    
    disrupted_edge_trace = go.Scatter(x=[], y=[], line=dict(width=5, color='rgba(255, 99, 71, 0.8)', dash='dot'), hoverinfo='none', mode='lines')
    for u, v in info.get("disrupted_edges", []):
        if u in POSITIONS and v in POSITIONS:
            disrupted_edge_trace['x'] += (POSITIONS[u][0], POSITIONS[v][0], None)
            disrupted_edge_trace['y'] += (0, 0, None) # Show disruption on the center line

    fig = go.Figure(data=edge_traces + [disrupted_edge_trace] + node_traces + train_traces,
                    layout=go.Layout(showlegend=False, margin=dict(b=20, l=20, r=20, t=10),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 1]),
                                     plot_bgcolor='#393E46', paper_bgcolor='#393E46', font_color='#EEEEEE', height=400))
    
    audit_df = pd.DataFrame.from_records(info.get('audit_history', []))
    if not audit_df.empty:
        audit_table = html.Table([html.Thead(html.Tr([html.Th(c) for c in ['Time', 'Trains', 'Cost Penalty']]))] +
                                 [html.Tbody([html.Tr([html.Td(f"{audit_df.iloc[i][c]:.2f}" if c == 'cost_penalty' else audit_df.iloc[i][c]) for c in ['time', 'active_trains', 'cost_penalty']]) for i in range(len(audit_df))])],
                                 className='table', style={'color': '#EEEEEE', 'fontSize': '12px', 'maxHeight': '150px', 'overflowY': 'auto'})
    else:
        audit_table = html.P("No audit records yet.", style={'fontSize': '12px', 'color': '#AAAAAA'})
    
    table_header = [html.Thead(html.Tr([html.Th("ID"), html.Th("Status"), html.Th("Decision & Reasoning"), html.Th("Mileage (km)"), html.Th("Bay")]))]
    table_rows = []
    reasoning = info.get('induction_reasoning', {})
    for train_id, data in sorted(info.get('trains', {}).items()):
        reason_data = reasoning.get(train_id, {})
        decision_key = reason_data.get('decision', 'UNKNOWN')
        if 'EXCLUDED' in decision_key:
            row_style, decision_text = {'backgroundColor': 'rgba(255, 99, 71, 0.2)'}, html.Span(decision_key, style={'color': '#FF6347', 'fontWeight': 'bold'})
        elif 'INCLUDED' in decision_key:
            row_style, decision_text = {'backgroundColor': 'rgba(0, 173, 181, 0.2)'}, html.Div([html.Span(f"INCLUDED (P:{reason_data.get('penalty', 0.0):.1f})", style={'color': '#00ADB5', 'fontWeight': 'bold'}), html.Br(), html.Small(f"Why: Risk({reason_data.get('predicted_risk', 0.0):.1f}) | Brand({reason_data.get('branding_cost', 0.0):.1f}) | Shunt({reason_data.get('shunting_cost', 0.0):.1f})", style={'color': '#BBBBBB'})])
        else:
            row_style, decision_text = {'backgroundColor': 'rgba(255, 215, 0, 0.1)'}, html.Div([html.Span(f"STANDBY (P:{reason_data.get('penalty', 0.0):.1f})", style={'color': '#FFD700', 'fontWeight': 'bold'}), html.Br(), html.Small("Low priority.", style={'color': '#BBBBBB'})])
        table_rows.append(html.Tr([html.Td(train_id), html.Td(data.get('state', 'N/A')),
                                   html.Td(decision_text, style={'fontSize': '10px'}),
                                   html.Td(f"{data.get('total_mileage_km', 0.0):.0f}"),
                                   html.Td(f"{data.get('bay_id', 'N/A')} ({data.get('assigned_depot', 'D?')[-1]})")], style=row_style))
    station_table = html.Table(table_header + [html.Tbody(table_rows)], className='table', style={'color': '#EEEEEE', 'fontSize': '12px'})

    return (active_trains_count, standby_trains_count, ibl_trains_count, headway_text,
            html.Span(status_text, style={'color': status_color}), total_passengers_waiting,
            fig, station_table, perf_summary, whatif_summary, checklist_options, checklist_value,
            time_display, forecast_display, audit_table, kpi_data)

@app.callback(Output('kpi-trend-chart', 'figure'), [Input('historical-kpi-store', 'data')])
def update_kpi_chart(kpi_data):
    if not kpi_data or not kpi_data['time']: return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kpi_data['time'], y=kpi_data['passengers'], name='Total Waiting Passengers', line=dict(color='#FFD700')))
    fig.add_trace(go.Scatter(x=kpi_data['time'], y=kpi_data['headway'], name='Avg. Headway (min)', yaxis='y2', line=dict(color='#00ADB5', dash='dot')))
    fig.update_layout(
        plot_bgcolor='#393E46', paper_bgcolor='#393E46', font_color='#EEEEEE',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=20, b=20),
        xaxis=dict(showgrid=False),
        yaxis=dict(title='Passengers', showgrid=True, gridcolor='#444'),
        yaxis2=dict(title='Headway (min)', overlaying='y', side='right', showgrid=False)
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)