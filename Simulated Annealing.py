import math
import random
import time

# Depot + Stops
# (Future add-on: Solomon)

stops = {
    0: (10, 70),    # Stop 0 (Base/Depot)
    1: (100, 25),   # Stop 1
    2: (50, 50),    # Stop 2
    3: (75, 20),    # Stop 3
    4: (60, 20),    # Stop 4
    5: (45, 70),    # Stop 5
    6: (95, 85),    # Stop 6 
    7: (15, 10),    # Stop 7 
    8: (80, 50),    # Stop 8 
    9: (30, 90),    # Stop 9 
}

# Distance Function
# Euclidean distance between 2 stops (a, b)
# Input a,b: tuple e.g. (8,10)

def distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


# Route Length
# Calculates cost of entire route. Route is a list of stop keys (not incl. 0)
# Input route: list of stop keys

def route_length(route):
    total = 0
    # 1. Start at depot
    prev_coord = stops[0]
    # 2. Iterate route
    for stop_key in route:
        current_coord = stops[stop_key]
        total += distance(prev_coord, current_coord)
        prev_coord = current_coord
    # 3. Return to depot
    total += distance(prev_coord, stops[0]) 
    return total

# Initial Route
# Random initial route. 

def init_route(stops):
    # including only non-depot keys
    route = list(stops.keys())[1:]
    random.shuffle(route)
    return route
    
# Random Neighbour (Swap)
# Generates a neighbouring solutions (swaps 2 stops)
# Input route: list of stop keys

def neighbour(route):
    new_route = route.copy() # .copy() used to create seperate copy
    # pick 2 random indices, swap
    i, j = random.sample(range(len(new_route)), 2)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route
    
# Simulated Annealing
# Optimal bus route

def simulated_annealing(stops_info, init_temp, cool_rate, iter):
    current_route = init_route(stops_info)
    best_route = current_route
    current_cost = route_length(current_route)
    best_cost = current_cost
    temp = init_temp
    
    start_time = time.time()
    for i in range(iter):
        # 1. Get neighbour route + cost
        neighbour_route = neighbour(current_route)
        neighbour_cost = route_length(neighbour_route)
        
        # 2. Cost Difference between routes
        difference_cost = neighbour_cost - current_cost
        
        # 3. Accept / Decline neighbour
        if difference_cost < 0:
            # Accept
            current_route = neighbour_route
            current_cost = neighbour_cost
            # Update best soln
            if current_cost < best_cost:
                best_cost = current_cost
                best_route = current_route
            
        else:
            # Formula 
            probability = math.exp((-difference_cost)/temp)   
            # Accept worse solution (escape local optima)
            if random.random() < probability:
                current_route = neighbour_route
                current_cost = neighbour_cost
        
        # 4. Cooling
        temp *= cool_rate
        
        # 5. Exit
        if temp < 0.001:
            print(f"Temperature too low: {temp:.3f}")
            break
        
        # 6. Print progress
        if i % 100 == 0:
            print(f"Iter {i:4d}, Temp={temp:.3f}, Best={best_cost:.2f}")
    
    end_time = time.time()
    
    return best_route, best_cost, end_time - start_time

# Run Static SA

TEMP = 10000.0  
COOLING_RATE = 0.9999
ITERATIONS = 30000

# 1. Run SA 
static_route_keys, static_cost, static_time = simulated_annealing(stops, TEMP, COOLING_RATE, ITERATIONS)

# 2. Print
static_route_sequence = [0] + static_route_keys + [0]
print(f"Stops: {stops}")
print(f"Optimal Route Sequence (keys): {static_route_sequence}")
print(f"Optimal Route Cost: {static_cost:.2f} units")
print(f"Simulated Annealing Algorithm took {static_time:.4f} seconds")

