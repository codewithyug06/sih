import networkx as nx

def create_railway_network():
    """
    Creates a realistic railway network graph for the Kochi Metro Line 1, 
    with a second depot for scalability testing.
    """
    G = nx.DiGraph()
    
    # UPDATED: The list of stations has been modified to match your app.py file.
    stations = [
        "Aluva", "Pulincha", "Companypady", "Ambattukavu", "Muttom", "Kalamassery",
        "Cochin University", "Pathadipalam", "Edapally", "Changampuzha Park",
        "Palarivattom", "JLN Stadium", "Kaloor", "Town Hall", "MG Road",
        "Vyttila", "Thykoodam", "Pettah", "Vadakkekotta", "SN Junction"
    ]
    
    # Dual Depot Definitions
    nodes = {
        "DEPOT_1": {"type": "depot", "stabling_bays": 5, "ibl_slots": 3},
        "DEPOT_2": {"type": "depot", "stabling_bays": 5, "ibl_slots": 3},
    }
    for station in stations:
        nodes[station] = {"type": "station", "passengers_waiting": 0, "avg_wait_time": 0}
        
    for node, attrs in nodes.items():
        G.add_node(node, **attrs)

    # --- Define Edges ---
    G.add_edge("DEPOT_1", "Aluva", weight=2)
    G.add_edge("Aluva", "DEPOT_1", weight=2)
    
    G.add_edge("DEPOT_2", "SN Junction", weight=5)
    G.add_edge("SN Junction", "DEPOT_2", weight=5)
    
    # Connections are built based on the updated stations list
    for i in range(len(stations) - 1):
        u, v = stations[i], stations[i+1]
        G.add_edge(u, v, weight=3) 
        G.add_edge(v, u, weight=3)
    
    return G
