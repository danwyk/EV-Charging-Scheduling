# Winter 2025 IOE 543 Final Project
# Minimizing Maximum Makespan in Heterogeneous EV Charging Station Scheduling
# Case Generator for simulation
# Aimeng Yang|aimeng@umich.edu, Kuo Wang|kuowang@umich.edu, Yikai Wang|danwyk@umich.edu
# Code Author: Kuo Wang kuowang@umich.edu
# Latest Update: Apr 2nd, 2025

import numpy as np
import os
import argparse


def generate_case(m, n, lambda_param, mu, sigma, max_release_time, output_folder, file_prefix):
    """
    Generate a case file for the Pm | r_j | Cmax problem.

    Parameters:
        m: int - Number of machines.
        n: int - Number of jobs.
        lambda_param: float - Parameter for the exponential distribution (used for job release time gaps).
        mu: float - Mean of the Gamma distribution (used for processing times).
        sigma: float - Standard deviation of the Gamma distribution (used for processing times).
        max_release_time: float - Maximum release time limit (e.g. 24*60 minutes for one day).
        output_folder: str - Directory to save the output file.

    File Format:
        Line 1: number of machines (m)
        Line 2: number of jobs (n)
        Line 3: release times r_j (space-separated)
        Line 4: processing times p_j (space-separated)
    """
    # Generate release times by accumulating exponential gaps,
    # but ensure they do not exceed max_release_time.
    release_times = [0.0]
    for i in range(1, n):
        gap = np.random.exponential(scale=1/lambda_param)
        candidate = release_times[-1] + gap
        if candidate > max_release_time:
            candidate = max_release_time
            release_times.append(candidate)
            # For remaining jobs, assign max_release_time.
            while len(release_times) < n:
                release_times.append(max_release_time)
            break
        else:
            release_times.append(candidate)
    
    # using k-theta for gamma dist
    shape = (mu / sigma) ** 2  # k  
    scale_param = sigma**2 / mu # theta
    
    # Generate processing times using the Gamma dist
    processing_times = np.random.gamma(shape, scale_param, size=n)
    # Ensure that processing times are at least 1
    processing_times = np.maximum(1, processing_times)
    
    # Format the release times and processing times as strings with int
    # TODO should we using sec instead of min??
    release_str = " ".join(f"{rt:.0f}" for rt in release_times)
    processing_str = " ".join(f"{pt:.0f}" for pt in processing_times)
    
    # Construct the output file name with parameters embedded in the name.
    file_name = f"case_m{m}_n{n}_lambda{lambda_param}_mu{mu}_sigma{sigma}_{file_prefix}.txt"
    
    file_path = os.path.join(output_folder, file_name)
    
    # Ensure the output directory exists.
    os.makedirs(output_folder, exist_ok=True)
    
    # Write the case data to the file.
    with open(file_path, 'w') as f:
        f.write(f"{m}\n")
        f.write(f"{n}\n")
        f.write(f"{release_str}\n")
        f.write(f"{processing_str}\n")
    
    print(f"Case file generated: {file_path}")

if __name__ == "__main__":
    # Set up command-line argument parsing.
    parser = argparse.ArgumentParser(description="Generate a case file for the Pm|r_j|Cmax problem.")
    parser.add_argument("--m", type=int, default=4, help="Number of machines (default: 3)")
    parser.add_argument("--n", type=int, default=100, help="Number of jobs")
    parser.add_argument("--lambda_param", type=float, default=0.2, help="Exponential distribution parameter for release time gaps")
    parser.add_argument("--mu", type=float, default=40, help="Mean for the Gamma distribution for processing times")
    parser.add_argument("--sigma", type=float, default=10, help="Standard deviation for the Gamma distribution for processing times")
    parser.add_argument("--max_release_time", type=float, default=24*60, help="Maximum release time in minutes")
    parser.add_argument("--output_folder", type=str, default="test", help="Directory to save the generated case file")
    parser.add_argument('--file_prefix', type=str, required=False, help='Prefix for output filename')
    
    args = parser.parse_args()
    
    # Generate the case file using the provided parameters.
    generate_case(args.m, args.n, args.lambda_param, args.mu, args.sigma, args.max_release_time, args.output_folder, args.file_prefix)
