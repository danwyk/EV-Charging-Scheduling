# Winter 2025 IOE 543 Final Project
# Minimizing Maximum Makespan in Heterogeneous EV Charging Station Scheduling
# Integral Program for Pm|rj|Cmax with Time Limit
# Aimeng Yang|aimeng@umich.edu, Kuo Wang|kuowang@umich.edu, Yikai Wang|danwyk@umich.edu
# Code Author: Kuo Wang kuowang@umich.edu
# Latest Update: Mar 29th, 2025 (Modified Apr 2025)

from gurobipy import Model, GRB, quicksum
import argparse

def read_scheduling_input(file_path):
    """
    Reads scheduling input from a .txt file.
    Expected format:
        Line 1: number of machines (m)
        Line 2: number of jobs (n)
        Line 3: release times r_j (space-separated)
        Line 4: processing times p_j (space-separated)
    
    Returns:
        m (int): number of machines
        n (int): number of jobs
        r (List[int]): release times for each job
        p (List[int]): processing times for each job
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
        m = int(lines[0])
        n = int(lines[1])
        r = list(map(int, lines[2].strip().split()))
        p = list(map(int, lines[3].strip().split()))
    return m, n, r, p

def solve_pm_rj_cmax(file_path):
    # Read data from file
    m, n, r, p = read_scheduling_input(file_path)
    M = sum(p) + max(r)
    
    # Initialize Gurobi model
    model = Model("pm_rj_cmax")
    
    # Decision variables
    x = {(i, j): model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}") for i in range(m) for j in range(n)}
    s = {j: model.addVar(vtype=GRB.CONTINUOUS, name=f"s_{j}") for j in range(n)}
    Cmax = model.addVar(vtype=GRB.CONTINUOUS, name="Cmax")
    
    y = {}
    for i in range(m):
        for j in range(n):
            for k in range(n):
                if j != k:
                    y[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{k}")
                    
    for j in range(n):
        model.addConstr(quicksum(x[i, j] for i in range(m)) == 1, name=f"job_assignment_{j}")
        model.addConstr(s[j] >= r[j], name=f"release_time_{j}")
        model.addConstr(s[j] + p[j] <= Cmax, name=f"cmax_constraint_{j}")
        
    for i in range(m):
        for j in range(n):
            for k in range(n):
                if j != k:
                    model.addConstr(s[j] + p[j] <= s[k] + M * (1 - y[i, j, k]) + M * (1 - x[i, j]) + M * (1 - x[i, k]),
                                        name=f"seq1_{i}_{j}_{k}")
                    model.addConstr(s[k] + p[k] <= s[j] + M * y[i, j, k] + M * (1 - x[i, j]) + M * (1 - x[i, k]),
                                        name=f"seq2_{i}_{j}_{k}")
                    
    # Set objective to minimize Cmax
    model.setObjective(Cmax, GRB.MINIMIZE)
    
    # TODO: Set time limit
    # If the time limit is reached, Gurobi will return the best solution found so far.
    model.setParam("TimeLimit", 60)
    model.setParam("OutputFlag", 0)  # Disable solver output
    
    model.optimize()
    
    # Retrieve the best solution found (even if time limit was reached)
    result = {}
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        result["Cmax"] = Cmax.X
        result["Schedule"] = []
        for j in range(n):
            assigned_machine = [i for i in range(m) if x[i, j].X > 0.5]
            machine = assigned_machine[0] if assigned_machine else None
            result["Schedule"].append({
                "Job": j,
                "StartTime": round(s[j].X, 2),
                "Machine": machine
            })
    else:
        result["Cmax"] = None
        result["Schedule"] = []
    
    return result

if __name__ == "__main__":
    # Command-line argument parsing for input file path
    parser = argparse.ArgumentParser(description="Solve the Pm|rj|Cmax using IP with a 180-second time limit")
    parser.add_argument("input_file")
    args = parser.parse_args()

    # Solve the scheduling problem using the specified input file
    result = solve_pm_rj_cmax(args.input_file)

    print("Optimal Cmax:", result["Cmax"])
    for entry in result["Schedule"]:
        print(f"Job {entry['Job']}: starts at {entry['StartTime']} on machine {entry['Machine']}")