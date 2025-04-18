import numpy as np
import random

def generate_chromosome():
    ''' 
    Generate a list of chromosomes to gene assignment
    -------------------------------------------------
    Chromosome: job
    Gene: Machine
    
    给每个 job 分配对应的 machine，1 个 job 只能被分配给
    1 个 machine；1 个 machine 可以处理多个 job
    '''
    return [random.randint(0, n_machines - 1) for _ in range(n_jobs)]


def generate_initial_population():
    '''
    Generate an initial population
    ------------------------------
    Population: a group of feasible schedules

    GA一开始生成一系列的schedule，用于后面一点点perturb，
    直到找到 optimum solution
    '''
    return [generate_chromosome() for _ in range(population_size)]


def calculate_schedule(chromosome):
    '''
    Calculate makespan for a given schedule
    ---------------------------------------
    计算每个 job 的开始、结束时间，以及其所对应
    machine 开始结束时间，最后得到整个 schedule
    的 Makespan

    * 需要讨论的地方 *
    Paper 里的方法没有考虑进去 release time，所以
    这里的 implementation 是默认按照 job list 的
    顺序，去依次分配的，有可能 release time 很靠后
    的 job x 在 job list 里排序在 release time 
    比较靠前的 job y 前面，所以最终导致 job x 会在
    job y 之前处理

    * 可以考虑的解决方案 *
    我们可以提前按照 release time 从小到大，对
    job list 排序，再带入到 GA 的算法里，这样可能会
    规避一些不合理的地方
    '''
    # Initialize machine times, job start & end time
    start_times = [0] * n_jobs
    completion_times = [0] * n_jobs

    # machine_times: List of (start, end) tuples for each job per machine
    machine_times = [[] for _ in range(n_machines)]  
    machine_available_time = [0] * n_machines

    for j in range(n_jobs):
        m = chromosome[j]
        start = max(machine_available_time[m], release_times[j])
        end = start + processing_times[j]
        
        # Update machien times
        start_times[j] = start
        completion_times[j] = end
        machine_times[m].append((start, end))
        machine_available_time[m] = end

    makespan = max(completion_times)
    
    return makespan, start_times, machine_available_time, completion_times


def calculate_fitness(chromosome):
    '''
    Compute the fitness for a given schedule
    ----------------------------------------
    Fitness 是用来衡量 schedule 的指标
    '''
    makespan, _, _, _ = calculate_schedule(chromosome)
    
    return alpha * np.exp(-beta * makespan), makespan


def roulette_wheel_selection(population, fitnesses):
    '''
    Roulette wheel selection method
    '''
    total_fitness = sum(fitnesses)
    probs = [f / total_fitness for f in fitnesses]
    
    # Calculate cumulative probablity for each chromosome
    cum_probs = np.cumsum(probs)
    selected = []
    
    for _ in range(len(population)):
        # Generates a float in [0.0, 1.0)
        r = random.random() 
        
        for i, cp in enumerate(cum_probs):
            if r <= cp:
                selected.append(population[i][:])
                break
    
    return selected


def crossover(population):
    '''
    Create a new chromosoe with an exiting one
    ---------------------------------------------
    调换 bottle neck machine 上面结束时间最早的 job，
    到最清闲的 machine 上面
    '''
    new_population = []

    for chrom in population:
        makespan, starts, _, _ = calculate_schedule(chrom)

        # Find f(gi') = max f(gi)
        machine_times = [0] * n_machines
        for j, m in enumerate(chrom):
            machine_times[m] += processing_times[j]

        busy_machine = np.argmax(machine_times)

        # Find j' = argmin [ max(release time, start time) + processing time]
        candidates = [j for j in range(n_jobs) if chrom[j] == busy_machine]
        if not candidates:
            new_population.append(chrom)
            continue

        job_to_move = min(candidates, key=lambda j: max(release_times[j], starts[j]) + processing_times[j])

        # Find least busy machine to move to (ensuring release time)
        best_machine = busy_machine
        min_makespan = makespan

        for m in range(n_machines):
            if m == busy_machine:
                continue

            chrom_tmp = chrom[:]
            chrom_tmp[job_to_move] = m
            new_makespan, new_starts, machine_available_time, _ = calculate_schedule(chrom_tmp)
            new_start_time = max(machine_available_time[m], release_times[job_to_move])

            if new_start_time >= release_times[job_to_move] and new_makespan < min_makespan:
                min_makespan = new_makespan
                best_machine = m

        if best_machine != busy_machine:
            chrom[job_to_move] = best_machine
        new_population.append(chrom)

    return new_population


def mutate(chromosome):
    '''
    Create a new chromosoe with small paturbations
    ---------------------------------------------
    '''
    j = random.randint(0, n_jobs - 1)
    current_machine = chromosome[j]
    best_m = current_machine
    best_makespan = calculate_schedule(chromosome)[0]

    for m in range(n_machines):
        if m == current_machine:
            continue
        new_chrom = chromosome[:]
        new_chrom[j] = m
        new_makespan, new_starts, _, _ = calculate_schedule(new_chrom)
        
        if new_starts[j] >= release_times[j] and new_makespan < best_makespan:
            best_makespan = new_makespan
            best_m = m

    chromosome[j] = best_m
    return chromosome


def run_genetic_algorithm(m_val=3, n_val=10,
                          release_times_val=np.random.randint(0, 10, 10),     
                          processing_times_val=np.random.randint(1, 10, 10),  
                          population_size_val=20,
                          max_generations_val=100,
                          alpha_val=1.0,
                          beta_val=0.1):

    global n_jobs, n_machines, processing_times, release_times
    global population_size, max_generations, alpha, beta

    n_jobs = n_val
    n_machines = m_val
    processing_times = processing_times_val
    release_times = release_times_val
    population_size = population_size_val
    max_generations = max_generations_val
    alpha = alpha_val
    beta = beta_val

    population = generate_initial_population()
    best_solution = None
    best_makespan = float('inf')

    for _ in range(max_generations):
        fitnesses = [calculate_fitness(ch)[0] for ch in population]
        makespans = [calculate_fitness(ch)[1] for ch in population]

        min_idx = np.argmin(makespans)
        if makespans[min_idx] < best_makespan:
            best_makespan = makespans[min_idx]
            best_solution = population[min_idx][:]

        selected = roulette_wheel_selection(population, fitnesses)
        crossed = crossover(selected)
        mutated = [mutate(ch[:]) for ch in crossed]
        population = mutated

    return release_times, processing_times, best_solution, best_makespan