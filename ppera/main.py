import CF
import CBF
import CF_BPR
import RL
import time
import logging # Import the logging module
import os
import traceback # To log stack traces for errors

# --- Setup Logging ---
def setup_logger(log_file='experiment_runner.log', level=logging.INFO):
    """Configures and returns a logger instance."""
    # Ensure log directory exists (assuming logs should go in the script's dir or a subdir)
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True) # Use exist_ok=True

    logger = logging.getLogger('ExperimentRunner')
    logger.setLevel(level)

    # Prevent adding multiple handlers if called multiple times (though not expected here)
    if not logger.handlers:
        # Create a file handler
        fh = logging.FileHandler(log_file, mode='a') # Append mode
        fh.setLevel(level)

        # Create a console handler (optional, keeps console output as well)
        # ch = logging.StreamHandler()
        # ch.setLevel(level)

        # Create a formatter and set it for the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        # ch.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(fh)
        # logger.addHandler(ch) # Uncomment to also log INFO/ERROR to console via logger

    return logger

# Configure logger (call this early)
logger = setup_logger()

# --- MLflow/Directory Comments ---
# mlflow server --host 127.0.0.1 --port 8080
# remember to be in the right directory to run this script from the command line (cd ppera)

def print_experiment_header(algorithm_name, dataset_name, num_rows):
    """Prints a formatted header for the experiment to console."""
    header = "\n" + "="*80 + "\n"
    header += f" Running Experiment: Algorithm = {algorithm_name}, Dataset = {dataset_name} "
    if num_rows:
        header += f"(Rows = {num_rows})"
    header += "\n" + "="*80 + "\n"
    print(header)

# --- Experiment Definitions ---
experiments = [
    {'algo': 'CBF', 'module': CBF, 'func': 'cbf_experiment_loop', 'dataset': 'movielens', 'rows': 10000},
    {'algo': 'CF', 'module': CF, 'func': 'cf_experiment_loop', 'dataset': 'movielens', 'rows': 10000},
    {'algo': 'CF_BPR', 'module': CF_BPR, 'func': 'cf_bpr_experiment_loop', 'dataset': 'movielens', 'rows': 10000},
    {'algo': 'RL', 'module': RL, 'func': 'rl_experiment_loop', 'dataset': 'movielens', 'rows': 10000},

    {'algo': 'CBF', 'module': CBF, 'func': 'cbf_experiment_loop', 'dataset': 'amazonsales', 'rows': 1000},
    {'algo': 'CF', 'module': CF, 'func': 'cf_experiment_loop', 'dataset': 'amazonsales', 'rows': 1000},
    {'algo': 'CF_BPR', 'module': CF_BPR, 'func': 'cf_bpr_experiment_loop', 'dataset': 'amazonsales', 'rows': 1000},
    {'algo': 'RL', 'module': RL, 'func': 'rl_experiment_loop', 'dataset': 'amazonsales', 'rows': 10000},

    {'algo': 'CBF', 'module': CBF, 'func': 'cbf_experiment_loop', 'dataset': 'postrecommendations', 'rows': 10000},
    {'algo': 'CF', 'module': CF, 'func': 'cf_experiment_loop', 'dataset': 'postrecommendations', 'rows': 10000},
    {'algo': 'CF_BPR', 'module': CF_BPR, 'func': 'cf_bpr_experiment_loop', 'dataset': 'postrecommendations', 'rows': 10000},
    {'algo': 'RL', 'module': RL, 'func': 'rl_experiment_loop', 'dataset': 'postrecommendations', 'rows': 10000},
]

# --- Common Parameters ---
common_params = {
    "TOP_K": 10,
    "want_col": ["userID", "itemID", "rating", "timestamp", 'title', 'genres'],
    "ratio": 0.75,
    "seed": 42,
}

# --- Run Experiments ---
logger.info("==================== Starting Experiment Batch ====================")
total_start_time = time.time()
completed_successfully = 0
failed_experiments = []

for i, exp in enumerate(experiments):
    algo = exp['algo']
    dataset = exp['dataset']
    rows = exp['rows']
    module = exp['module']
    func_name = exp['func']
    exp_label = f"{i+1}/{len(experiments)} ({algo}/{dataset})" # Label for logging

    print_experiment_header(algo, dataset, rows)
    logger.info(f"Starting Experiment {exp_label}...")
    start_time = time.time()

    # Get the actual function from the module
    try:
        experiment_function = getattr(module, func_name)
    except AttributeError:
        error_msg = f"Function '{func_name}' not found in module for algorithm '{algo}'. Skipping."
        print(f"!!! Error: {error_msg}")
        logger.error(f"Experiment {exp_label} SKIPPED: {error_msg}")
        failed_experiments.append(f"{exp_label} - Function not found")
        continue

    # Prepare arguments for the specific function call
    func_args = common_params.copy()
    func_args['dataset'] = dataset
    func_args['num_rows'] = rows

    # Log the arguments being used (optional but helpful)
    logger.info(f"  Function: {module.__name__}.{func_name}")
    logger.info(f"  Arguments: {func_args}")

    # Call the function and log result/errors
    try:
        print(f"Calling: {module.__name__}.{func_name}(**...)") # Keep console call concise
        experiment_function(**func_args)
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n--- Finished Experiment {exp_label} in {duration:.2f} seconds ---")
        logger.info(f"Finished Experiment {exp_label} successfully in {duration:.2f} seconds.")
        completed_successfully += 1
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        error_traceback = traceback.format_exc() # Get full traceback string
        print(f"\n!!! Experiment {exp_label} FAILED after {duration:.2f} seconds.")
        print(f"!!! Error: {e}")
        logger.error(f"Experiment {exp_label} FAILED after {duration:.2f} seconds.")
        # Log the exception type and message, plus the traceback
        logger.error(f"Error details: {type(e).__name__}: {e}", exc_info=False) # Don't need exc_info if logging traceback separately
        logger.error(f"Traceback:\n{error_traceback}")
        failed_experiments.append(f"{exp_label} - Error: {type(e).__name__}")
        # Decide whether to continue or break on error
        # break # Uncomment to stop after the first failure

total_end_time = time.time()
total_duration = total_end_time - total_start_time

summary_header = "\n" + "="*80 + "\n"
summary_header += " All Experiments Finished. "
summary_header += f"Total Time: {total_duration:.2f} seconds "
summary_header += "\n" + "="*80 + "\n"
print(summary_header)
logger.info("==================== Finished Experiment Batch ====================")
logger.info(f"Total Duration: {total_duration:.2f} seconds.")
logger.info(f"Experiments Completed Successfully: {completed_successfully}/{len(experiments)}")
if failed_experiments:
    logger.warning("Failed Experiments:")
    for failure in failed_experiments:
        logger.warning(f"  - {failure}")
logger.info("===================================================================")
