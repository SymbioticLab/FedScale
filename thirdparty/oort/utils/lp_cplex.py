import cplex


def lp_cplex(datas, systems, budget, preference, data_size, init_values = None, time_limit = None, read_flag = False, write_flag = False, request_budget=True, gap = None):

    num_of_clients = len(datas)
    num_of_class = len(datas[0])

    # Create the modeler/solver
    prob = cplex.Cplex()
    prob.objective.set_sense(prob.objective.sense.minimize)

    # Time to transmit the data
    trans_time = [round(data_size/systems[i][1], 2) for i in range(num_of_clients)]

    # System speeds
    speed = [systems[i][0]/1000. for i in range(num_of_clients)]


    slowest = list(prob.variables.add(obj = [1.0], lb = [0.0], types = ['C'], names = ['slowest']))

    quantity = [None] * num_of_clients
    for i in range(num_of_clients):
        quantity[i] = list(prob.variables.add(obj = [0.0] * num_of_class,
                                              lb = [0.0] * num_of_class,
                                              ub = [q for q in datas[i]],
                                              types = ['I'] * num_of_class,
                                              names = [f'Client {i} Class{j}' for j in range(num_of_class)]))


    # Minimize the slowest
    for i in range(num_of_clients):
        ind = slowest + [quantity[i][j] for j in range(num_of_class)]
        val = [1.0] + [-speed[i]] * num_of_class
        prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=ind, val=val)],
                                    senses = ['G'],
                                    rhs = [trans_time[i]],
                                    names = [f'slow_{i}'])

    # Preference Constraint
    for j in range(num_of_class):
        ind = [quantity[i][j] for i in range(num_of_clients)]
        val = [1.0] * num_of_clients
        prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=ind, val=val)],
                                    senses = ['G'],
                                    rhs = [preference[j]],
                                    names = [f'preference_{i}'])

    # Budget Constraint
    if request_budget:
        status = list(prob.variables.add(obj = [0.0] * num_of_clients,
                                         types = ['B'] * num_of_clients,
                                         names = [f'status {i}' for i in range(num_of_clients)]))
        for i in range(num_of_clients):
            ind = [quantity[i][j] for j in range(num_of_class)]
            val = [1.0] * num_of_class
            prob.indicator_constraints.add(indvar=status[i],
                                           complemented=1,
                                           rhs=0.0,
                                           sense="L",
                                           lin_expr=cplex.SparsePair(ind=ind, val=val),
                                           name=f"ind{i}")
        prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=status, val=[1.0] * num_of_clients)],
                                    senses = ['L'],
                                    rhs = [budget],
                                    names = ['budget'])

    # Solve the problem
    prob.solve()

    # And print the solutions
    print("Solution status =", prob.solution.get_status_string())
    print("Optimal value:", prob.solution.get_objective_value())
    tol = prob.parameters.mip.tolerances.integrality.get()
    values = prob.solution.get_values()

    result = [[0] * num_of_class for _ in range(num_of_clients)]
    for i in range(num_of_clients):
        for j in range(num_of_class):
            if values[quantity[i][j]] > tol:
                result[i][j] = values[quantity[i][j]]

    return result