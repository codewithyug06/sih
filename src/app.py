# app.py
import dash
from dash import dcc, html, Input, Output, State 
import plotly.graph_objects as go
import networkx as nx
from train_env import TrainTrafficEnv

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server
env = TrainTrafficEnv()
state, info = env.reset()

# --- FIX: Ensure only pure English names are used in the dashboard display ---
STATIONS_LIST = [
    "Aluva", "Pulincha", "Companypady", "Ambattukavu", "Muttom", "Kalamassery",
    "Cochin University", "Pathadipalam", "Edapally", "Changampuzha Park",
    "Palarivattom", "JLN Stadium", "Kaloor", "Town Hall", "MG Road", # Fixed "M. G. Road" for plotting stability
    "Maharajas College", "Ernakulam South", "Kadavanthra", "Elamkulam",
    "Vyttila", "Thykoodam", "Pettah", "Vadakkekotta", "SN Junction"
]
POSITIONS = {"DEPOT": (-1, 0.5)}
for i, station_name in enumerate(STATIONS_LIST):
    POSITIONS[station_name] = (i, 0)

# Global variable to store What-If results, accessible by all callbacks
WHATIF_RESULTS = {}


app.layout = html.Div(style={'backgroundColor': '#222831', 'color': '#EEEEEE', 'fontFamily': 'Roboto, sans-serif'}, children=[
    html.Div(style={'backgroundColor': '#00ADB5', 'padding': '15px', 'textAlign': 'center'}, children=[
        html.H2("KMRL AI Control Center", style={'color': 'white', 'margin': '0px'}),
        html.P("Smart Train Induction Planning & Scheduling System", style={'color': '#EEEEEE', 'fontSize': '14px', 'margin': '5px 0 0 0'})
    ]),
    dcc.Interval(id='interval-component', interval=100, n_intervals=0),
    html.Div(style={'display': 'flex', 'justifyContent': 'space-around', 'padding': '20px'}, children=[
        html.Div(className="three columns card", style={'backgroundColor': '#393E46', 'padding': '15px', 'borderRadius': '8px', 'margin': '0 10px', 'textAlign': 'center'}, children=[
            html.P("Active Trains", style={'fontSize': '16px', 'color': '#00ADB5'}),
            html.H3(id='active-trains-card')
        ]),
        html.Div(className="three columns card", style={'backgroundColor': '#393E46', 'padding': '15px', 'borderRadius': '8px', 'margin': '0 10px', 'textAlign': 'center'}, children=[
            html.P("Standby Trains", style={'fontSize': '16px', 'color': '#00ADB5'}),
            html.H3(id='standby-trains-card')
        ]),
        html.Div(className="three columns card", style={'backgroundColor': '#393E46', 'padding': '15px', 'borderRadius': '8px', 'margin': '0 10px', 'textAlign': 'center'}, children=[
            html.P("IBL (Maintenance)", style={'fontSize': '16px', 'color': '#FF6347'}),
            html.H3(id='ibl-trains-card')
        ]),
        html.Div(className="three columns card", style={'backgroundColor': '#393E46', 'padding': '15px', 'borderRadius': '8px', 'margin': '0 10px', 'textAlign': 'center'}, children=[
            html.P("Current Headway", style={'fontSize': '16px', 'color': '#00ADB5'}),
            html.H3(id='headway-card')
        ]),
        html.Div(className="three columns card", style={'backgroundColor': '#393E46', 'padding': '15px', 'borderRadius': '8px', 'margin': '0 10px', 'textAlign': 'center'}, children=[
            html.P("System Status", style={'fontSize': '16px', 'color': '#00ADB5'}),
            html.H3(id='status-card')
        ]),
        html.Div(className="three columns card", style={'backgroundColor': '#393E46', 'padding': '15px', 'borderRadius': '8px', 'margin': '0 10px', 'textAlign': 'center'}, children=[
            html.P("Total Waiting Psg", style={'fontSize': '16px', 'color': '#00ADB5'}), # Changed symbol to text for stability
            html.H3(id='total-passengers-card')
        ]),
    ]),
    html.Div(style={'display': 'flex', 'padding': '0 20px 20px 20px'}, children=[
        html.Div(style={'flex': '65%', 'backgroundColor': '#393E46', 'borderRadius': '8px', 'padding': '15px', 'marginRight': '20px'}, children=[
            html.H4("Live Metro Map - Real-Time Crowd & Train Positions", style={'color': '#00ADB5'}),
            dcc.Graph(id='network-graph', style={'height': '500px'})
        ]),
        html.Div(style={'flex': '35%', 'backgroundColor': '#393E46', 'borderRadius': '8px', 'padding': '15px'}, children=[
            html.H4("Fleet Induction & Analytics (XAI)", style={'color': '#00ADB5'}),
            
            # --- AI Plan Performance Card ---
            html.Div(style={'backgroundColor': '#2D4059', 'padding': '10px', 'borderRadius': '6px', 'marginBottom': '10px'}, children=[
                html.P("AI Plan Performance (Predicted 24h)", style={'fontSize': '14px', 'color': '#00ADB5', 'margin': '0'}),
                html.Div(id='ai-performance-summary', style={'display': 'flex', 'justifyContent': 'space-between', 'fontSize': '12px', 'marginTop': '5px'})
            ]),
            
            # --- WHAT-IF SIMULATION PANEL ---
            html.Div(style={'backgroundColor': '#2D4059', 'padding': '10px', 'borderRadius': '6px', 'marginBottom': '10px'}, children=[
                html.P("What-If Simulation (Manual Override)", style={'fontSize': '14px', 'color': '#FFD700', 'margin': '0'}),
                html.Div(style={'display': 'flex'}, children=[
                    html.Div(style={'flex': '70%'}, children=[
                        dcc.Checklist(
                            id='manual-induction-checklist',
                            value=env.induction_plan, 
                            inline=False,
                            style={'color': '#EEEEEE', 'fontSize': '12px', 'maxHeight': '100px', 'overflowY': 'auto', 'padding': '5px'},
                            labelStyle={'display': 'block'}
                        )
                    ]),
                    html.Div(style={'flex': '30%', 'textAlign': 'right', 'marginLeft': '5px'}, children=[
                        html.Button('RUN WHAT-IF', id='run-whatif-button', n_clicks=0, 
                                    style={'backgroundColor': '#00ADB5', 'color': 'white', 'border': 'none', 'padding': '5px', 'borderRadius': '4px', 'cursor': 'pointer', 'width': '100%'}),
                        html.Div(id='whatif-performance-summary', style={'fontSize': '11px', 'marginTop': '5px', 'color': '#32CD32'})
                    ])
                ])
            ]),

            html.Div(id='station-table-div', style={'maxHeight': '200px', 'overflowY': 'auto'}) 
        ])
    ])
])


@app.callback(
    [Output('active-trains-card', 'children'), Output('standby-trains-card', 'children'),
     Output('ibl-trains-card', 'children'), Output('headway-card', 'children'), 
     Output('status-card', 'children'), Output('total-passengers-card', 'children'), 
     Output('network-graph', 'figure'), Output('station-table-div', 'children'),
     Output('ai-performance-summary', 'children'),
     Output('whatif-performance-summary', 'children'), 
     Output('manual-induction-checklist', 'options'),
     Output('manual-induction-checklist', 'value')],
    [Input('interval-component', 'n_intervals'),
     Input('run-whatif-button', 'n_clicks')],
    [State('manual-induction-checklist', 'value')] 
)
def update_view(n, n_clicks, manual_induction_list):
    global state, info, WHATIF_RESULTS

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # 1. Handle Core Simulation Step
    terminated = False # Initialize terminated state
    if triggered_id == 'interval-component' and n % 10 == 0:
        active_count = info.get('active_trains_count', 0)
        waiting_count = info.get('total_passengers_waiting', 0)
        
        if waiting_count > 500 and active_count < 8: action = 1
        elif waiting_count < 100 and active_count > 4: action = 2
        else: action = 0
        
        state, reward, terminated, _, info = env.step(action)
        if terminated: state, info = env.reset()
        
        # Reset What-If on Nightly Planning Run or Reset
        if terminated or (info.get('current_time', 0) >= 1260 and (info.get('current_time', 0) - env.time_step_seconds / 60) < 1260):
            WHATIF_RESULTS = {}

    elif triggered_id == 'interval-component':
        # Update train positions
        for train in env.trains.values(): train.update(env.time_step_seconds / 10, env.rail_network, info.get('current_time', 0))
        info = env._get_info()
    
    # 2. Handle What-If Button Click 
    elif triggered_id == 'run-whatif-button' and n_clicks > 0:
        # NOTE: Using .get(tid) safely in case list contains stale IDs
        valid_manual_list = [tid for tid in manual_induction_list if env.trains.get(tid) and env._check_hard_constraints(env.trains.get(tid)) == "READY"]
        WHATIF_RESULTS = env._simulate_day_performance(valid_manual_list)


    # --- 3. Dynamic UI Updates and Calculations ---
    active_trains_count = info.get('active_trains_count', 0)
    standby_trains_count = info.get('standby_trains_count', 0)
    ibl_trains_count = info.get('ibl_trains_count', 0)
    headway_text = f"{info.get('current_headway_min', 0.0):.1f} min"
    status_text = info.get('system_status', 'UNKNOWN')
    total_passengers_waiting = info.get('total_passengers_waiting', 0)
    status_color = '#FF6347' if status_text == 'CRITICAL' else '#FFD700' if status_text == 'ALERT' else '#32CD32'

    # --- AI Performance Summary ---
    ai_perf = info.get('ai_plan_performance', {})
    # Use 0.1 as a sensible default cost threshold
    cost_color = '#FF6347' if ai_perf.get('predicted_maintenance_cost', 0.1) > 0.1 else '#FFD700'
    readiness_color = '#FF6347' if ai_perf.get('predicted_service_readiness', 0.0) < 0.8 else '#32CD32'

    perf_summary = [
        html.Span(f"Cost Penalty: {ai_perf.get('predicted_maintenance_cost', 0.0):.2f}", 
                  style={'color': cost_color, 'marginRight': '20px'}),
        html.Span(f"Readiness: {ai_perf.get('predicted_service_readiness', 0.0)*100:.0f}%", 
                  style={'color': readiness_color})
    ]
    
    # --- What-If Performance Summary (Comparison) ---
    whatif_summary = []
    if WHATIF_RESULTS:
        w_cost = WHATIF_RESULTS.get('predicted_maintenance_cost', 0)
        w_ready = WHATIF_RESULTS.get('predicted_service_readiness', 0)
        
        # Compare against AI's plan (Red if significantly worse)
        ai_cost = ai_perf.get('predicted_maintenance_cost', 0.1)
        ai_ready = ai_perf.get('predicted_service_readiness', 0.0)
        
        w_cost_color = '#FF6347' if w_cost > ai_cost * 1.05 else '#32CD32'
        w_readiness_color = '#FF6347' if w_ready < ai_ready * 0.95 else '#32CD32'

        whatif_summary = [
            html.Span(f"Cost: {w_cost:.2f} ", 
                      style={'color': w_cost_color, 'marginRight': '5px', 'fontWeight': 'bold'}),
            html.Span(f"Ready: {w_ready*100:.0f}%", 
                      style={'color': w_readiness_color, 'fontWeight': 'bold'})
        ]
    
    # --- Checklist Options and Value Update ---
    checklist_options = []
    for t in env.trains.values():
        is_ready = env._check_hard_constraints(t) == "READY"
        label_style = {'textDecoration': 'none'}
        if not is_ready:
            label_style['textDecoration'] = 'line-through'
            label_style['color'] = '#FF6347'

        checklist_options.append({
            'label': html.Span(f' Train {t.id} ({t.bay_id})', style=label_style),
            'value': t.id,
            'disabled': not is_ready 
        })
        
    # Reset checklist value to AI's plan if a new nightly plan was run
    if triggered_id == 'interval-component' and (terminated or info.get('current_time', 0) >= 1260 and (info.get('current_time', 0) - env.time_step_seconds / 60) < 1260):
        checklist_value = info.get('induction_plan', [])
    else:
        checklist_value = manual_induction_list


    # --- Create Network Visualization ---
    G = env.rail_network
    pos = POSITIONS
    edge_trace = go.Scatter(x=[], y=[], line=dict(width=4, color='#666666'), hoverinfo='none', mode='lines')
    for edge in G.edges():
        if edge[0] in pos and edge[1] in pos:
            edge_trace['x'] += (pos[edge[0]][0], pos[edge[1]][0], None)
            edge_trace['y'] += (pos[edge[0]][1], pos[edge[1]][1], None)

    node_traces = []
    
    # Use STATIONS_LIST for clean, guaranteed English names
    for station_name in STATIONS_LIST: 
        station_data = info['stations'].get(station_name, {}) # Defensive dictionary lookup
        passengers_waiting = station_data.get('passengers_waiting', 0)
        color = '#FF6347' if passengers_waiting > 200 else '#FFD700' if passengers_waiting > 50 else '#32CD32'
        
        # FIX: Changed label to only include clean English names and passenger count
        node_traces.append(go.Scatter(x=[pos[station_name][0]], y=[pos[station_name][1]], mode='markers+text',
                                      text=f"{station_name}<br>{passengers_waiting} Psg", 
                                      textposition="bottom center",
                                      marker=dict(size=12, color=color, line=dict(width=1, color='#EEEEEE'))))
    
    depot_text = f"DEPOT<br>Standby: {standby_trains_count}<br>IBL: {ibl_trains_count}"
    node_traces.append(go.Scatter(x=[pos["DEPOT"][0]], y=[pos["DEPOT"][1]], mode='markers+text', text=depot_text,
                                  textposition="top center", marker=dict(size=20, color='#00ADB5')))

    train_traces = []
    for train_id, train_data in info.get("trains", {}).items():
        if train_data['state'] not in ['in_depot', 'standby', 'IBL']:
            x, y = pos.get(train_data['current_node'], (0,0))
            if train_data['state'] == 'moving' and train_data['path'] and len(train_data['path']) > 1:
                u, v = train_data['path'][0], train_data['path'][1]
                if G.has_edge(u, v):
                    edge_len = G.get_edge_data(u, v)['weight']
                    progress = train_data['distance_on_edge'] / edge_len if edge_len > 0 else 0
                    x0, y0 = pos.get(u, (0,0))
                    x1, y1 = pos.get(v, (0,0))
                    x = x0 + (x1 - x0) * progress
                    y = y0 + (y1 - y0) * progress + 0.1
            
            train_color = '#FFD700' if train_data.get('is_induction_plan') else '#32CD32' 
            hover_text = f"Train {train_id}<br>Pass: {len(train_data['passengers'])}<br>Mileage: {train_data['total_mileage_km']:.0f} km"
            train_traces.append(go.Scatter(x=[x], y=[y], mode='text', text="🚆", # Use simple ASCII emoji
                                           textfont=dict(size=24, color=train_color), hoverinfo='text', hovertext=hover_text))
            
    fig = go.Figure(data=[edge_trace] + node_traces + train_traces,
                    layout=go.Layout(showlegend=False, margin=dict(b=20, l=20, r=20, t=10),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 1]),
                                     plot_bgcolor='#393E46', paper_bgcolor='#393E46', font_color='#EEEEEE', height=500))

    # --- Assemble Fleet Induction Table (XAI) ---
    table_header = [html.Thead(html.Tr([
        html.Th("ID"), html.Th("Status"), html.Th("Decision & Reasoning"), 
        html.Th("Mileage (km)"), html.Th("Bay") 
    ]))]
    
    sorted_trains = sorted(info.get('trains', {}).items(), key=lambda item: item[0])
    table_rows = []
    
    reasoning = info.get('induction_reasoning', {}) 
    
    for train_id, data in sorted_trains:
        reason_data = reasoning.get(train_id, {})
        
        row_style = {}
        decision_key = reason_data.get('decision', 'UNKNOWN')
        
        if 'EXCLUDED' in decision_key:
            row_style = {'backgroundColor': 'rgba(255, 99, 71, 0.2)'} 
            decision_text = html.Span(decision_key, style={'color': '#FF6347', 'fontWeight': 'bold'})
        elif 'INCLUDED' in decision_key: 
            row_style = {'backgroundColor': 'rgba(0, 173, 181, 0.2)'} 
            decision_text = html.Div([
                html.Span(f"INCLUDED (P:{reason_data.get('penalty', 0.0):.1f})", style={'color': '#00ADB5', 'fontWeight': 'bold'}),
                html.Br(),
                html.Small(f"Why: Brand({reason_data.get('branding_cost', 0.0):.1f}) | Shunt({reason_data.get('shunting_cost', 0.0):.1f}) | Mile({reason_data.get('mileage_deviation', 0.0):.1f})", style={'color': '#BBBBBB'})
            ])
        else: # STANDBY
            row_style = {'backgroundColor': 'rgba(255, 215, 0, 0.1)'}
            decision_text = html.Div([
                html.Span(f"STANDBY (P:{reason_data.get('penalty', 0.0):.1f})", style={'color': '#FFD700', 'fontWeight': 'bold'}),
                html.Br(),
                html.Small("Low priority, available for real-time induction.", style={'color': '#BBBBBB'})
            ])

        table_rows.append(html.Tr([
            html.Td(train_id), 
            html.Td(data.get('state', 'N/A')),
            html.Td(decision_text, style={'fontSize': '10px'}),
            html.Td(f"{data.get('total_mileage_km', 0.0):.0f}"), 
            html.Td(data.get('bay_id', 'N/A'))
        ], style=row_style))
        
    station_table = html.Table(table_header + [html.Tbody(table_rows)], className='table', style={'color': '#EEEEEE', 'fontSize': '12px'})

    return (
        active_trains_count,
        standby_trains_count, 
        ibl_trains_count, 
        headway_text,
        html.Span(status_text, style={'color': status_color}),
        total_passengers_waiting,
        fig,
        station_table,
        perf_summary,
        whatif_summary,
        checklist_options,
        checklist_value
    )

if __name__ == '__main__':
    app.run(debug=True)