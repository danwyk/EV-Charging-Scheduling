import os
import time
import csv

# Import functions from ip.py and genetic_algorithm.py
from ip import solve_pm_rj_cmax, read_scheduling_input
from genetic_algorithm import run_genetic_algorithm

def run_ip_solver(file_path):
    start_time = time.time()
    result = solve_pm_rj_cmax(file_path)
    runtime = time.time() - start_time
    ip_cmax = result["Cmax"]
    return ip_cmax, runtime

def run_ga_solver(file_path):
    m, n, release_times, processing_times = read_scheduling_input(file_path)
    start_time = time.time()
    _, _, _, best_makespan = run_genetic_algorithm(
        m_val=m,
        n_val=n,
        release_times_val=release_times,
        processing_times_val=processing_times,
        population_size_val=20,
        max_generations_val=100,
        alpha_val=1.0,
        beta_val=0.1
    )
    runtime = time.time() - start_time
    return best_makespan, runtime

def main():
    # TODO: Change the test folder path here
    base_dir = "test/RealisticCases/m9"
     # TODO: Change the result folder path here
    result_dir = "result/100_real"
    os.makedirs(result_dir, exist_ok=True) 

    for root, dirs, files in os.walk(base_dir):
        results = []
        subdir_has_test = False
        m = n = None 

        for file in files:
            if file.endswith(".txt"):
                subdir_has_test = True
                file_path = os.path.join(root, file)
                print(f"Processing {file_path} ...")
                
                ip_cmax, ip_runtime = run_ip_solver(file_path)
                ga_cmax, ga_runtime = run_ga_solver(file_path)
                m, n, _, _ = read_scheduling_input(file_path)  

                results.append({
                    "test_file": file,
                    "m": m,
                    "n": n,
                    "algorithm": "IP",
                    "Cmax": ip_cmax,
                    "runtime_sec": round(ip_runtime, 2)
                })
                results.append({
                    "test_file": file,
                    "m": m,
                    "n": n,
                    "algorithm": "GA",
                    "Cmax": ga_cmax,
                    "runtime_sec": round(ga_runtime, 2)
                })

                print(f"  IP: Cmax = {ip_cmax:.0f}, runtime = {ip_runtime:.2f} sec")
                print(f"  GA: Cmax = {ga_cmax:.0f}, runtime = {ga_runtime:.2f} sec")

        if subdir_has_test and m is not None and n is not None:
            output_filename = f"results_m{m}.csv"
            output_path = os.path.join(result_dir, output_filename)
            with open(output_path, "w", newline="") as csvfile:
                fieldnames = ["test_file", "m", "n", "algorithm", "Cmax", "runtime_sec"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    writer.writerow(row)
            print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
