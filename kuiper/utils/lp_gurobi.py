import time, sys
import pickle, math
import numpy as np
from gurobipy import *


def lp_gurobi(data, systems, budget, preference, data_trans_size, init_values = None, time_limit = None, read_flag = False, write_flag = False, request_budget=True, gap = None):
    """
        See appendix G for detailed MILP formula
    """

    num_of_clients = len(data)
    num_of_class = len(data[0])

    # Create a new model
    m = Model("client_selection")

    qlist = []
    for i in range(num_of_clients):
        for j in range(num_of_class):
            qlist.append((i, j))

    slowest = m.addVar(vtype=GRB.CONTINUOUS, name="slowest", lb = 0.0) # Slowest client
    quantity = m.addVars(qlist, vtype=GRB.INTEGER, name="quantity", lb = 0) # number of samples for each category of each client

    # System[i][0] is the inference latency, so we should multiply, ms -> sec for inference latency
    time_list = [(quicksum([quantity[(i, j)] for j in range(num_of_class)])*systems[i][0])/1000. for i in range(num_of_clients)]
    comm_time_list = [data_trans_size/float(systems[i][1]) for i in range(num_of_clients)]

    # Preference Constraint
    for i in range(num_of_class):
        m.addConstr(quicksum([quantity[(client, i)] for client in range(num_of_clients)]) >= preference[i], name='preference_' + str(i))

    # Capacity Constraint
    m.addConstrs((quantity[i] <= data[i[0]][i[1]] for i in quantity), name='capacity_'+str(i))

    # Binary var indicates the selection status
    status = m.addVars([i for i in range(num_of_clients)], vtype = GRB.BINARY, name = 'status')
    for i in range(num_of_clients):
        m.addGenConstrIndicator(status[i], False, quantity.sum(i, '*') <= 0)

    # Budget Constraint
    if request_budget:
        m.addConstr(quicksum([status[i] for i in range(num_of_clients)]) <= budget, name = 'budget')

    # Minimize the slowest client
    for idx, t in enumerate(time_list):
        m.addConstr(slowest >= t + status[idx] * comm_time_list[idx], name=f'slow_{idx}')

    # Initialize variables if init_values is provided
    if init_values:
        for k, v in init_values.items():
            quantity[k].Start = v

    # Set a 'time_limit' second time limit
    if time_limit:
        m.Params.timeLimit = time_limit

    # set the optimality gap
    if gap is not None:
        m.Params.MIPgap = gap

    # The objective is to minimize the slowest
    m.setObjective(slowest, GRB.MINIMIZE)
    m.update()
    if read_flag:
        if os.path.exists('temp.mst'):
            m.read('temp.mst')

    m.optimize()
    lpruntime = m.Runtime
    # print(f'Optimization took {lpruntime}')
    # print(f'Gap between current and optimal is {m.MIPGap}')

    # Process the solution
    result = np.zeros([num_of_clients, num_of_class])
    if m.status == GRB.OPTIMAL:
        # print(f'Found optimal solution and slowest is {slowest.x}')
        pointx = m.getAttr('x', quantity)
        for i in qlist:
            if quantity[i].x > 0.0001:
                result[i[0]][i[1]] = pointx[i]
    else:
        print('No optimal solution')

    if write_flag:
        m.write('temp.mst')
    m.write('model.lp')

    return result, m.objVal, lpruntime
