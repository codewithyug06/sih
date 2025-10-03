import gymnasium as gym
import numpy as np
import networkx as nx
from track_layout import create_railway_network
import collections
import random
import copy
import pandas as pd

class Passenger:
    def __init__(self, entry_time):
        self.entry_time = entry_time
        self.wait_time = 0

class Train:
    def __init__(self, train_id, capacity=150, speed=80, initial_depot="DEPOT_1"):
        self.id = train_id
        self.capacity = capacity
        self.passengers = []
        self.speed_kph = speed
        self.state = "in_depot"
        self.current_node = initial_depot
        self.assigned_depot = initial_depot
        self.rs_clearance_valid = random.choice([True, True, True, False])
        self.snt_clearance_valid = random.choice([True, True, True, False])
        self.open_job_cards = random.choice([0, 0, 0, 1, 1, 2])
        self.branding_exposure_hours = random.randint(0, 100)
        self.branding_target_hours = 250
        self.total_mileage_km = random.randint(1000, 10000)
        self.days_since_deep_clean = random.randint(0, 10)
        self.bay_id = f"BAY_{random.randint(1, 5)}"
        self.component_health_score = random.randint(60, 100)
        self.predicted_failure_risk = 0.0
        self.next_maintenance_due_min = random.randint(1440, 7200)
        self.path = []
        self.distance_on_edge = 0
        self.last_station_arrival_time = 0
        self.direction = None

    def assign_path(self, path, direction):
        self.path = list(path)
        self.direction = direction
        if len(self.path) > 1:
            self.state = "moving"
            self.distance_on_edge = 0
        elif len(self.path) == 1:
            self.state = "at_station" if self.path[0] not in ["DEPOT_1", "DEPOT_2"] else "in_depot"
            self.current_node = self.path[0]
            if self.state == "in_depot": self.bay_id = f"BAY_{random.randint(1, 5)}"

    def update(self, time_step_seconds, graph, current_time_minutes):
        if self.state == "failed":
            return
            
        if self.state == "moving":
            wear = (self.speed_kph / 3600) * time_step_seconds * 0.0001
            self.component_health_score = max(0, self.component_health_score - wear)
        
        self.next_maintenance_due_min -= (time_step_seconds / 60)
        
        if self.state == "moving" and self.path and len(self.path) > 1:
            dist_to_cover = (self.speed_kph / 3600) * time_step_seconds
            self.distance_on_edge += dist_to_cover
            self.total_mileage_km += dist_to_cover
            self.branding_exposure_hours += time_step_seconds / 3600

            u, v = self.path[0], self.path[1]
            if graph.has_edge(u, v):
                edge_len = graph.get_edge_data(u, v)['weight']
                if self.distance_on_edge >= edge_len:
                    self.current_node = self.path.pop(1)
                    self.distance_on_edge = 0
                    if self.current_node not in ["DEPOT_1", "DEPOT_2"]:
                        self.state = "at_station"
                        self.last_station_arrival_time = current_time_minutes
                    else:
                        self.state = "in_depot"
                        self.path = []
            else:
                self.state = "at_station"
                self.path = [self.current_node]

class TrainTrafficEnv(gym.Env):
    DEPOT_TOPOLOGY = {"BAY_5": ["BAY_4", "BAY_3", "BAY_2", "BAY_1"], "BAY_4": ["BAY_3", "BAY_2", "BAY_1"], "BAY_3": ["BAY_2", "BAY_1"], "BAY_2": ["BAY_1"], "BAY_1": []}
    
    def __init__(self):
        super().__init__()
        self.original_rail_network = create_railway_network()
        self.rail_network = self.original_rail_network.copy()
        self.disrupted_edges = set()
        self.failed_trains = set()

        self.time_step_seconds = 60
        self.current_time_minutes = 0
        self.passenger_demand_profile = { 0: 50, 420: 300, 600: 100, 1020: 400, 1200: 50, 1380: 20 }
        
        # UPDATED: Removed demand multipliers for stations that no longer exist.
        self.station_demand_multipliers = {"Aluva": 1.2, "JLN Stadium": 1.5, "Edapally": 1.3, "SN Junction": 1.2}
        
        default_multiplier = 0.8
        self.stations_list = [n for n, d in self.rail_network.nodes(data=True) if d['type'] == 'station']
        for s in self.stations_list:
            if s not in self.station_demand_multipliers: self.station_demand_multipliers[s] = default_multiplier
        self.trains = {}
        self.next_train_id = 0
        self.station_last_train_arrival = {s: -np.inf for s in self.stations_list}
        self.station_headways = {s: collections.deque(maxlen=5) for s in self.stations_list}
        self.action_space = gym.spaces.Discrete(3)
        
        self.observation_space = gym.spaces.Box(low=0, high=1.0, shape=(6,), dtype=np.float32)

        self.TOTAL_FLEET_SIZE = 12
        self.TARGET_FLEET_SIZE = 4
        self.induction_plan = []
        self.induction_reasoning = {}
        self.ai_plan_performance = {}
        self.history_df = pd.DataFrame(columns=['time', 'waiting_psg', 'active_trains', 'induction_plan', 'cost_penalty'])
        self.demand_forecast_multiplier = 1.0
        
        self._initialize_fleet()
        self._run_nightly_planning()

    def inject_scenario(self, scenario_type, details=None):
        if scenario_type == 'clear_all':
            self.rail_network = self.original_rail_network.copy()
            self.disrupted_edges.clear()
            for train_id in self.failed_trains:
                if train_id in self.trains:
                    if self.trains[train_id].current_node in self.stations_list:
                         self.trains[train_id].state = "at_station"
                    else:
                         self.trains[train_id].state = "IBL"
            self.failed_trains.clear()
            print("All disruptions cleared.")
            
        elif scenario_type == 'train_failure':
            active_trains = [t for t in self.trains.values() if t.state not in ['in_depot', 'standby', 'IBL', 'failed']]
            if active_trains:
                failed_train = random.choice(active_trains)
                failed_train.state = "failed"
                self.failed_trains.add(failed_train.id)
                print(f"SCENARIO: Train {failed_train.id} has failed.")

        elif scenario_type == 'track_blockage':
            u, v = "Edapally", "Palarivattom"
            if self.rail_network.has_edge(u, v):
                self.rail_network.remove_edge(u, v)
                self.disrupted_edges.add((u, v))
            if self.rail_network.has_edge(v, u):
                self.rail_network.remove_edge(v, u)
                self.disrupted_edges.add((v, u))
            print(f"SCENARIO: Track between {u} and {v} is blocked.")
            
        elif scenario_type == 'passenger_surge':
            station = "JLN Stadium"
            if station in self.rail_network.nodes:
                surge_amount = 300
                self.rail_network.nodes[station].setdefault('passenger_objects', [])
                for _ in range(surge_amount):
                    self.rail_network.nodes[station]['passenger_objects'].append(Passenger(self.current_time_minutes))
                print(f"SCENARIO: Passenger surge of {surge_amount} at {station}.")
    
    def _initialize_fleet(self):
        for i in range(self.TOTAL_FLEET_SIZE):
            depot_id = "DEPOT_1" if i < self.TOTAL_FLEET_SIZE / 2 else "DEPOT_2"
            new_id = f"KM{i+1:02d}"
            train = Train(new_id, initial_depot=depot_id)
            self.trains[new_id] = train
            self.next_train_id += 1
            
    def _check_hard_constraints(self, train):
        if not train.rs_clearance_valid: return "RS_CERT_EXPIRED"
        if not train.snt_clearance_valid: return "SNT_CERT_EXPIRED"
        if train.open_job_cards > 0: return "OPEN_JOB_CARD"
        if train.days_since_deep_clean > 7: return "CLEANING_REQUIRED"
        if train.next_maintenance_due_min < 0: return "MAJOR_MAINT_DUE"
        return "READY"
        
    def _calculate_predictive_risk(self, train):
        health_risk = (100 - train.component_health_score) * 0.2
        mileage_risk = max(0, train.total_mileage_km - 8000) * 0.001
        train.predicted_failure_risk = health_risk + mileage_risk
        return train.predicted_failure_risk * 5
        
    def _calculate_branding_cost(self, train):
        required = train.branding_target_hours
        served = train.branding_exposure_hours
        if served < required:
            return (required - served) * 0.5
        elif served > required + 50:
            return (served - required) * 0.1
        return 0

    def _calculate_shunting_cost(self, induction_list, train_bay_map):
        total_shunts = 0
        blocker_bay_occupancy = collections.defaultdict(list)
        for tid in self.trains:
            if tid not in induction_list:
                blocker_bay_occupancy[train_bay_map.get(tid, "UNKNOWN")].append(tid)

        for tid in induction_list:
            bay = train_bay_map.get(tid, "UNKNOWN")
            for blocker_bay in self.DEPOT_TOPOLOGY.get(bay, []):
                if blocker_bay_occupancy.get(blocker_bay):
                    total_shunts += len(blocker_bay_occupancy[blocker_bay])
        return total_shunts * 5

    def _calculate_mileage_cost(self, induction_list, train_mileage_map):
        active_mileages = [train_mileage_map[tid] for tid in induction_list if tid in train_mileage_map]
        if not active_mileages: return 100
        cost = np.var(active_mileages) * 0.000001
        return cost

    def _calculate_total_cost(self, induction_list, train_data_map):
        total_cost = 0
        if not induction_list: return 10000

        for tid in induction_list:
            train = train_data_map[tid]
            total_cost += self._calculate_branding_cost(train)
            total_cost += self._calculate_predictive_risk(train)

        mileage_map = {tid: train_data_map[tid].total_mileage_km for tid in train_data_map}
        total_cost += self._calculate_mileage_cost(induction_list, mileage_map)
        
        bay_map = {tid: train_data_map[tid].bay_id for tid in train_data_map}
        total_cost += self._calculate_shunting_cost(induction_list, bay_map)
        
        return total_cost

    def _get_initial_train_data_map(self):
        return {t.id: copy.copy(t) for t in self.trains.values()}

    def _run_nightly_planning(self):
        train_data_map = self._get_initial_train_data_map()
        all_train_ids = list(train_data_map.keys())
        ineligible_trains = {t.id: self._check_hard_constraints(t) for t in self.trains.values() if self._check_hard_constraints(t) != "READY"}
        available_ids = [tid for tid in all_train_ids if tid not in ineligible_trains]
        fleet_mileages = [t.total_mileage_km for t in self.trains.values()]
        avg_mileage = np.mean(fleet_mileages) if fleet_mileages else 1
        
        plan_size = min(self.TARGET_FLEET_SIZE, len(available_ids))
        if plan_size == 0: current_plan = []
        else: current_plan = random.sample(available_ids, plan_size)
            
        current_cost = self._calculate_total_cost(current_plan, train_data_map)
        best_plan = current_plan
        best_cost = current_cost
        T = 1.0; T_min = 0.0001; alpha = 0.99
        
        if len(available_ids) > 0 and len(current_plan) < len(available_ids):
            while T > T_min:
                for _ in range(100):
                    trains_to_swap_out = current_plan
                    trains_to_swap_in = [tid for tid in available_ids if tid not in current_plan]
                    if not trains_to_swap_out or not trains_to_swap_in: break
                    
                    swap_out = random.choice(trains_to_swap_out)
                    swap_in = random.choice(trains_to_swap_in)
                    
                    neighbor_plan = [tid for tid in current_plan if tid != swap_out] + [swap_in]
                    neighbor_cost = self._calculate_total_cost(neighbor_plan, train_data_map)

                    delta_cost = neighbor_cost - current_cost
                    if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                        current_plan = neighbor_plan
                        current_cost = neighbor_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_plan = current_plan
                T = T * alpha

        self.induction_plan = best_plan
        self.induction_reasoning = ineligible_trains.copy()
        for tid in all_train_ids:
            if tid in ineligible_trains:
                self.induction_reasoning[tid] = {'decision': f"EXCLUDED: {ineligible_trains[tid]}"}
            else:
                train = train_data_map[tid]
                cost = self._calculate_total_cost([tid], train_data_map)
                self.induction_reasoning[tid] = {
                    "penalty": cost,
                    "branding_cost": self._calculate_branding_cost(train),
                    "shunting_cost": self._calculate_shunting_cost([tid], {t.id: t.bay_id for t in train_data_map.values()}),
                    "mileage_deviation": (train.total_mileage_km - avg_mileage) / 1000 if avg_mileage != 0 else 0,
                    "predicted_risk": train.predicted_failure_risk,
                    "decision": "INCLUDED_OPTIMAL" if tid in best_plan else "STANDBY_LOW_PRIORITY"
                }
        
        forward_starters, backward_starters = 0, 0
        for train in self.trains.values():
            if train.id in self.induction_plan:
                if train.assigned_depot == "DEPOT_1" and forward_starters < self.TARGET_FLEET_SIZE / 2:
                    source_node, initial_direction = "Aluva", "forward"
                    forward_starters += 1
                elif train.assigned_depot == "DEPOT_2" and backward_starters < self.TARGET_FLEET_SIZE / 2:
                    source_node, initial_direction = "SN Junction", "backward"
                    backward_starters += 1
                else:
                    source_node = "Aluva" if train.assigned_depot == "DEPOT_1" else "SN Junction"
                    initial_direction = "forward" if train.assigned_depot == "DEPOT_1" else "backward"
                path = nx.shortest_path(self.rail_network, source=train.assigned_depot, target=source_node)
                train.assign_path(path, initial_direction)
                train.state = "moving"
            elif train.id in ineligible_trains:
                train.state = "IBL"
            else:
                train.state = "standby"
            if train.state != "moving":
                train.path = []
                train.current_node = train.assigned_depot
                
        self.ai_plan_performance = self._simulate_day_performance(self.induction_plan)
        self._log_audit(self.induction_plan, best_cost)

    def _simulate_day_performance(self, induction_list):
        train_data_map = self._get_initial_train_data_map()
        predicted_cost_penalty = self._calculate_total_cost(induction_list, train_data_map)
        if not induction_list:
            return {"predicted_maintenance_cost": predicted_cost_penalty, "predicted_service_readiness": 0.0}
        predicted_service_readiness = sum(1 for tid in induction_list if self._check_hard_constraints(train_data_map[tid]) == "READY")
        return {"predicted_maintenance_cost": predicted_cost_penalty, "predicted_service_readiness": predicted_service_readiness / len(induction_list)}
        
    def _log_audit(self, induction_list, cost):
        new_row = {'time': self.current_time_minutes, 'waiting_psg': self._get_info()['total_passengers_waiting'],
                   'active_trains': len(induction_list), 'induction_plan': ", ".join(induction_list), 'cost_penalty': cost}
        self.history_df.loc[len(self.history_df)] = new_row
        self.history_df = self.history_df[self.history_df['time'] >= self.current_time_minutes - (7 * 1440)]
        
    def _update_forecast(self):
        last_day_data = self.history_df[self.history_df['time'] >= self.current_time_minutes - 1440]
        if len(last_day_data) > 0:
            avg_penalty = last_day_data['cost_penalty'].mean()
            self.demand_forecast_multiplier = min(1.5, max(0.8, 1.0 + (avg_penalty / 1000)))

    def get_current_demand_rate(self):
        rate = 0
        for time_thresh, demand_rate in self.passenger_demand_profile.items():
            if self.current_time_minutes >= time_thresh: rate = demand_rate
        return rate * self.demand_forecast_multiplier

    def _update_passengers(self):
        base_demand_rate_per_hour = self.get_current_demand_rate()
        for station_name in self.stations_list:
            multiplier = self.station_demand_multipliers.get(station_name, 1.0)
            station_demand = base_demand_rate_per_hour * multiplier
            passengers_per_step = station_demand / (3600 / self.time_step_seconds)
            new_passengers_count = int(np.random.poisson(passengers_per_step))
            self.rail_network.nodes[station_name].setdefault('passenger_objects', [])
            for _ in range(new_passengers_count): self.rail_network.nodes[station_name]['passenger_objects'].append(Passenger(self.current_time_minutes))
            for p in self.rail_network.nodes[station_name]['passenger_objects']: p.wait_time += self.time_step_seconds

    def _get_obs(self):
        active_trains_count = sum(1 for t in self.trains.values() if t.state not in ["in_depot", "standby", "IBL", "failed"])
        norm_active_trains = active_trains_count / self.TOTAL_FLEET_SIZE
        total_waiting = sum(len(d.get('passenger_objects', [])) for n, d in self.rail_network.nodes(data=True) if d.get('type') == 'station')
        norm_waiting = min(total_waiting / 2000.0, 1.0)
        norm_time = self.current_time_minutes / (24 * 60)
        standby_trains_count = sum(1 for t in self.trains.values() if t.state == "standby")
        norm_standby_trains = standby_trains_count / self.TOTAL_FLEET_SIZE
        ibl_trains_count = sum(1 for t in self.trains.values() if t.state == "IBL")
        norm_ibl_trains = ibl_trains_count / self.TOTAL_FLEET_SIZE
        norm_forecast = (self.demand_forecast_multiplier - 0.8) / 0.7
        return np.array([norm_time, norm_waiting, norm_active_trains, norm_standby_trains, norm_ibl_trains, norm_forecast], dtype=np.float32)

    def _get_info(self):
        station_data = {}
        total_passengers_waiting, current_headway_sum, current_headway_count = 0, 0, 0
        for station_name in self.stations_list:
            passengers = self.rail_network.nodes[station_name].get('passenger_objects', [])
            total_passengers_waiting += len(passengers)
            avg_wait_s = sum(p.wait_time for p in passengers) / len(passengers) if passengers else 0
            avg_headway = np.mean(list(self.station_headways[station_name])) if self.station_headways[station_name] else 0
            if avg_headway > 0:
                current_headway_sum += avg_headway
                current_headway_count += 1
            station_data[station_name] = {"passengers_waiting": len(passengers), "avg_wait_time_s": avg_wait_s, "avg_headway_min": avg_headway / 60}
        
        active_trains_count = sum(1 for t in self.trains.values() if t.state not in ["in_depot", "standby", "IBL", "failed"])
        standby_trains_count = sum(1 for t in self.trains.values() if t.state == "standby")
        ibl_trains_count = sum(1 for t in self.trains.values() if t.state == "IBL")
        depot_trains_count = sum(1 for t in self.trains.values() if t.current_node in ["DEPOT_1", "DEPOT_2"])
        avg_system_headway = (current_headway_sum / current_headway_count / 60) if current_headway_count > 0 else 0
        status = "CRITICAL" if total_passengers_waiting > 2000 or (avg_system_headway > 15 and avg_system_headway > 0) else "ALERT" if total_passengers_waiting > 1000 or (avg_system_headway > 10 and avg_system_headway > 0) else "NORMAL"
        
        train_data = {}
        for t in self.trains.values():
            data = t.__dict__.copy()
            data['is_induction_plan'] = t.id in self.induction_plan
            train_data[t.id] = data

        info_dict = {"current_time": self.current_time_minutes, "total_passengers_waiting": total_passengers_waiting, "active_trains_count": active_trains_count,
                     "standby_trains_count": standby_trains_count, "ibl_trains_count": ibl_trains_count, "depot_trains_count": depot_trains_count,
                     "current_headway_min": avg_system_headway, "system_status": status, "trains": train_data, "stations": station_data,
                     "induction_reasoning": getattr(self, 'induction_reasoning', {}), "ai_plan_performance": getattr(self, 'ai_plan_performance', {}),
                     "audit_history": self.history_df.to_dict('records'), "demand_forecast": self.demand_forecast_multiplier,
                     "disrupted_edges": list(self.disrupted_edges), "failed_trains": list(self.failed_trains),
                     "rail_network": self.rail_network}
        return info_dict

    def reset(self, seed=None, options=None):
        self.rail_network = self.original_rail_network.copy()
        self.disrupted_edges.clear()
        self.failed_trains.clear()
        self.current_time_minutes = 0
        self.trains = {}
        self.next_train_id = 0
        self._initialize_fleet()
        self._run_nightly_planning()
        for station in self.stations_list:
            self.rail_network.nodes[station]['passenger_objects'] = []
            self.station_last_train_arrival[station] = -np.inf
            self.station_headways[station] = collections.deque(maxlen=5)
        return self._get_obs(), self._get_info()

    def step(self, action):
        reward = 0
        self._update_passengers()
        if action == 1:
            train = next((t for t in self.trains.values() if t.state == "standby" and self._check_hard_constraints(t) == "READY"), None)
            if train:
                try:
                    source_node = "Aluva" if train.assigned_depot == "DEPOT_1" else "SN Junction"
                    path = nx.shortest_path(self.rail_network, source=train.assigned_depot, target=source_node)
                    train.assign_path(path, "forward" if source_node == "Aluva" else "backward")
                except nx.NetworkXNoPath: pass
        elif action == 2:
            train = next((t for t in self.trains.values() if t.state == "at_station" and t.current_node in ["Aluva", "SN Junction"]), None)
            if train:
                try:
                    target_state = "IBL" if self._check_hard_constraints(train) != "READY" else "standby"
                    train.state = target_state
                    path = nx.shortest_path(self.rail_network, source=train.current_node, target=train.assigned_depot)
                    train.assign_path(path, None)
                except nx.NetworkXNoPath: pass

        for train in self.trains.values():
            if train.state in ["in_depot", "standby", "IBL"]:
                if train.next_maintenance_due_min < 0: train.next_maintenance_due_min = random.randint(3000, 7200)
                if train.days_since_deep_clean > 7: train.days_since_deep_clean = 0
                if train.state == "IBL" and train.open_job_cards > 0: train.open_job_cards = 0

            if train.state == "moving" and train.path and len(train.path) > 1:
                u, v = train.path[0], train.path[1]
                if not self.rail_network.has_edge(u, v):
                    train.state = "at_station"
                    train.path = [train.current_node]

            train.update(self.time_step_seconds, self.rail_network, self.current_time_minutes)
            
            if train.state == "at_station" and train.current_node in self.stations_list:
                time_since_last = self.current_time_minutes * 60 - self.station_last_train_arrival[train.current_node]
                if self.station_last_train_arrival[train.current_node] > -np.inf: self.station_headways[train.current_node].append(time_since_last)
                self.station_last_train_arrival[train.current_node] = self.current_time_minutes * 60
                station_passengers = self.rail_network.nodes[train.current_node]['passenger_objects']
                alighting = int(len(train.passengers) * 0.2)
                train.passengers = train.passengers[alighting:]
                space = train.capacity - len(train.passengers)
                boarding = station_passengers[:space]
                train.passengers.extend(boarding)
                self.rail_network.nodes[train.current_node]['passenger_objects'] = station_passengers[space:]
                try:
                    current_idx = self.stations_list.index(train.current_node)
                    if train.direction == "forward":
                        target = self.stations_list[current_idx - 1] if current_idx == len(self.stations_list) - 1 else self.stations_list[current_idx + 1]
                        new_direction = "backward" if current_idx == len(self.stations_list) - 1 else "forward"
                    else: # backward
                        target = self.stations_list[current_idx + 1] if current_idx == 0 else self.stations_list[current_idx - 1]
                        new_direction = "forward" if current_idx == 0 else "backward"
                    path = nx.shortest_path(self.rail_network, source=train.current_node, target=target)
                    train.assign_path(path, new_direction)
                except (nx.NetworkXNoPath, ValueError): train.state = 'at_station'
        
        total_waiting = sum(len(d.get('passenger_objects', [])) for n, d in self.rail_network.nodes(data=True) if d.get('type') == 'station')
        reward = -total_waiting * 0.01
        self.current_time_minutes += self.time_step_seconds / 60
        terminated = self.current_time_minutes >= (24 * 60)
        
        if self.current_time_minutes >= 1260 and self.current_time_minutes - (self.time_step_seconds / 60) < 1260:
            self._update_forecast()
            self._run_nightly_planning()
            
        return self._get_obs(), reward, terminated, False, self._get_info()