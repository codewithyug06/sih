# track_layout.py
import networkx as nx

def create_railway_network():
    """
    Creates a realistic railway network graph for the full Kochi Metro Line 1.
    """
    G = nx.DiGraph()
    
    # All 24 stations of Kochi Metro Line 1 + Depot
    stations = [
        "Aluva", "Pulincha", "Companypady", "Ambattukavu", "Muttom", "Kalamassery",
        "Cochin University", "Pathadipalam", "Edapally", "Changampuzha Park",
        "Palarivattom", "JLN Stadium", "Kaloor", "Town Hall", "MG Road", 
        "Maharajas College", "Ernakulam South", "Kadavanthra", "Elamkulam",
        "Vyttila", "Thykoodam", "Pettah", "Vadakkekotta", "SN Junction"
    ]
    
    nodes = {"DEPOT": {"type": "depot", "stabling_bays": 5, "ibl_slots": 3}}
    for station in stations:
        nodes[station] = {"type": "station", "passengers_waiting": 0, "avg_wait_time": 0}
        
    for node, attrs in nodes.items():
        G.add_node(node, **attrs)

    # --- Define Edges ---
    G.add_edge("DEPOT", "Aluva", weight=2)
    
    for i in range(len(stations) - 1):
        u, v = stations[i], stations[i+1]
        G.add_edge(u, v, weight=3) # Assign an arbitrary weight/distance
        G.add_edge(v, u, weight=3) # Reverse path
    
    G.add_edge("Aluva", "DEPOT", weight=2)
    G.add_edge("SN Junction", "DEPOT", weight=5)
    
    return G