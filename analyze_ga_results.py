
def draw_hv(bug_res_path):

    res = np.load(bug_res_path)['res']

    # create the performance indicator object with reference point
    metric = Hypervolume(ref_point=np.array([1.0, 1.0, 1.0, 1.0]))

    # collect the population in each generation
    pop_each_gen = [a.pop for a in res.history]

    # receive the population in each generation
    obj_and_feasible_each_gen = [pop[pop.get("feasible")[:,0]].get("F") for pop in pop_each_gen]

    # calculate for each generation the HV metric
    hv = [metric.calc(f) for f in obj_and_feasible_each_gen]

    # function evaluations at each snapshot
    n_evals = np.array([a.evaluator.n_eval for a in res.history])

    # visualze the convergence curve
    plt.plot(n_evals, hv, '-o')
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.show()


def draw_performance(bug_res_path):

    time_bug_num_list = np.load(bug_res_path)['time_bug_num_list']

    for t, n in time_bug_num_list:
        t_list.append(t)
        n_list.append(n)
    plt.plot(t_list, n_list, '-o')
    plt.title("Time VS Number of Bugs")
    plt.xlabel("Time")
    plt.ylabel("Number of Bugs")
    plt.show()


def show_stats(bug_res_path):

if __name__ == '__main__':
    bug_res_path = ""
    draw_hv(bug_res_paths)
    draw_performance(bug_res_paths)
