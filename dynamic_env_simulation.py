import simpy # simulation library
import uuid # for generating unique IDs
import random # generate requests

class PassengerRequest:
# will define 1 passenger request
    def __init__(self, request_id, request_time, pickup_loc, dropoff_loc):
        self.request_id = request_id        # unique identifier for the request
        self.request_time = request_time    # time when the request was made
        self.pickup_loc = pickup_loc        # location where the passenger is picked up
        self.dropoff_loc = dropoff_loc      # location where the passenger is dropped off

        # to be updated by sim and algos
        self.status = "UNASSIGNED"          # status of request (UNASSIGNED, ASSIGNED, IN_PROGRESS, COMPLETED)
        self.assigned_bus_id = None         # bus assigned to this request
        self.pickup_time = -1             # time when passenger is picked up
        self.dropoff_time = -1            # time when passenger is dropped off

    def __repr__(self):
        return (f"Request({self.request_id[:6]}, T={self.request_time:.2f} | {self.pickup_loc} -> {self.dropoff_loc} | {self.status})")

class Vehicle:
# will define 1 vehicle (bus)
    def __init__(self, env, bus_id, depot_loc, capacity = 15):
        self.env = env                      # simpy environment
        self.bus_id = bus_id                # unique identifier for the bus
        self.capacity = capacity            # max passenger capacity of the bus

        self.current_load = 0               # current number of passengers on the bus
        self.passengers = []                # list of passenger requests currently on the bus

        self.current_location = depot_loc   # current location of the bus (x, y)
        self.route = []                     # list of passenger requests currently on the bus
        self.status = "IDLE"                # status of the bus (IDLE, EN_ROUTE, AT_STOP, RE-OPTIMIZING)

        print(f"Bus {bus_id} initialized at Depot {depot_loc} at t={env.now:.2f}")
    
    def assign_request(self, request):
        # PLACEHOLDER: logic to assign request to bus (ALNS, GA, TS)
        request.status = "ASSIGNED"
        request.assigned_bus_id = self.bus_id
        self.route.append(request)

        print(f"Bus {self.bus_id} assigned Request {request.request_id} at t={self.env.now:.2f}")

class SimulationEnvironment:
# Main Simulation Environment  
    def __init__(self, sim_duration=100, num_buses=3):
        print(f"Initializing simulation for {sim_duration} minutes")
        self.env = simpy.Environment()      # simpy environment
        self.sim_duration = sim_duration    # total duration of the simulation

        self.requests_received = []          # total requests received
        self.requests_completed = []         # total requests completed
        self.unassigned_requests = []         # total unassigned requests

        self.fleet = []                     # list of vehicles in the simulation
        # self.fleet = [Vehicle(self.env, f"Bus_{i}") for i in range(num_buses)]
        self.requests = []                  # list of all passenger requests

        self.algorithm = None               # placeholder for optimization algorithm instance

    def add_vehicle(self, depot_loc, capacity=15):
        bus_id = str(uuid.uuid4())         # generate unique bus ID
        vehicle = Vehicle(self.env, bus_id, depot_loc, capacity)
        self.fleet.append(vehicle)

    def batched_optimizer(self):
        # ON_TIMER: placeholder for batched optimization logic
        while True:
            yield self.env.timeout(20)  # run optimization every 20 time units

            if not self.unassigned_requests:
                continue  # skip if no unassigned requests
            
            print(f"Running batched optimizer at t={self.env.now:.2f} for {len(self.unassigned_requests)} unassigned requests")

            self.run_SA()
    
    def run_SA(self):
        pass  # placeholder for Simulated Annealing algorithm

        


