import logging
import os
import time
import traceback

from . import CBF
from . import CF
from . import RL

# --- MLflow/Directory Comments ---
# mlflow server --host 127.0.0.1 --port 8080
# remember to be in the right directory to run this script from the command line (cd ppera)


# --- Setup Logging ---
def setup_logger(log_file="experiment_runner.log", level=logging.INFO):
    """Configures and returns a logger instance."""
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger_instance = logging.getLogger("ExperimentRunner")  # Use a specific name
    logger_instance.setLevel(level)

    if not logger_instance.handlers:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(formatter)
        logger_instance.addHandler(fh)

    return logger_instance


logger = setup_logger()


def print_experiment_header(run_config_label, algorithm_name, dataset_name, num_rows):
    """Prints a formatted header for the experiment to console."""
    header = "\n" + "=" * 80 + "\n"
    header += f" Config: {run_config_label} | Running Experiment: Algorithm = {algorithm_name}, Dataset = {dataset_name} "
    if num_rows:
        header += f"(Rows = {num_rows})"
    header += "\n" + "=" * 80 + "\n"
    print(header)


# --- Experiment Definitions (Algo/Dataset combinations) ---
experiments_to_run = [
    {
        "algo": "CBF",
        "module": CBF,
        "func": "cbf_experiment_loop",
        "dataset": "movielens",
        "rows": 14000,
    },
    {
        "algo": "CF",
        "module": CF,
        "func": "cf_experiment_loop",
        "dataset": "movielens",
        "rows": 14000,
    },
    {
        "algo": "RL",
        "module": RL,
        "func": "rl_experiment_loop",
        "dataset": "movielens",
        "rows": 14000,
    },
    {"algo": "CBF", "module": CBF, "func": "cbf_experiment_loop", "dataset": "amazonsales"},
    {"algo": "CF", "module": CF, "func": "cf_experiment_loop", "dataset": "amazonsales"},
    {"algo": "RL", "module": RL, "func": "rl_experiment_loop", "dataset": "amazonsales"},
    {
        "algo": "CBF",
        "module": CBF,
        "func": "cbf_experiment_loop",
        "dataset": "postrecommendations",
        "rows": 14000,
    },
    {
        "algo": "CF",
        "module": CF,
        "func": "cf_experiment_loop",
        "dataset": "postrecommendations",
        "rows": 14000,
    },
    {
        "algo": "RL",
        "module": RL,
        "func": "rl_experiment_loop",
        "dataset": "postrecommendations",
        "rows": 14000,
    },
]

# --- Parameter Configurations ---
base_params = {
    "TOP_K": 10,
    "want_col": ["userID", "itemID", "rating", "timestamp", "title", "genres"],
    "ratio": 0.75,
    "seed": 42,
}

default_conditional_params = {
    "privacy": False,
    "hide_type": None,
    "columns_to_hide": None,
    "fraction_to_hide": None,
    "personalization": False,
    "fraction_to_change": None,
    "change_rating": False,
}

all_param_configurations = []

# 1. Clear configuration
clear_config = {**base_params, **default_conditional_params}
clear_config.update({"run_label": "Clear", "privacy": False, "personalization": False})
all_param_configurations.append(clear_config)

# 2. Privacy configurations
privacy_fractions = [0.1, 0.25, 0.5, 0.8]
for p_frac in privacy_fractions:
    privacy_config = {**base_params, **default_conditional_params}
    privacy_config.update(
        {
            "run_label": f"Privacy_{p_frac:.2f}",
            "privacy": True,
            "hide_type": "values_in_column",
            "columns_to_hide": ["title", "genres"],
            "fraction_to_hide": p_frac,
            "personalization": False,
        }
    )
    all_param_configurations.append(privacy_config)

# 3. Personalization configurations
personalization_fractions = [0.1, 0.25, 0.5, 0.8]
for pers_frac in personalization_fractions:
    personalization_config = {**base_params, **default_conditional_params}
    personalization_config.update(
        {
            "run_label": f"Personalization_{pers_frac:.2f}",
            "privacy": False,
            "personalization": True,
            "fraction_to_change": pers_frac,
            "change_rating": True,
        }
    )
    all_param_configurations.append(personalization_config)


# --- Run Experiments ---
logger.info("==================== Starting Experiment Batch ====================")
overall_start_time = time.time()
overall_completed_successfully = 0
overall_failed_experiments_details = []  # Stores detailed failure info

total_individual_experiments_planned = len(all_param_configurations) * len(experiments_to_run)
current_individual_experiment_count = 0

for config_idx, current_run_params in enumerate(all_param_configurations):
    run_label = current_run_params["run_label"]
    config_params_for_function = current_run_params.copy()
    if "run_label" in config_params_for_function:  # Remove label, not for experiment func
        del config_params_for_function["run_label"]

    logger.info(f"\n{'-' * 20} Starting Configuration Run {config_idx + 1}/{len(all_param_configurations)}: {run_label} {'-' * 20}")
    config_start_time = time.time()
    config_completed_count = 0
    config_failed_list = []

    for exp_idx, exp_definition in enumerate(experiments_to_run):
        current_individual_experiment_count += 1
        algo = exp_definition["algo"]
        dataset = exp_definition["dataset"]
        rows = exp_definition.get("rows", None)  # Use .get for optional 'rows'
        module = exp_definition["module"]
        func_name = exp_definition["func"]

        # More detailed label for logging and tracking
        exp_progress_label = (
            f"Config {config_idx + 1}/{len(all_param_configurations)} [{run_label}] - Exp {exp_idx + 1}/{len(experiments_to_run)} ({algo}/{dataset})"
        )
        # Label for console header, simpler
        header_label = f"[{run_label}] {algo}/{dataset}"

        print_experiment_header(run_label, algo, dataset, rows)
        logger.info(f"Starting Individual Experiment ({current_individual_experiment_count}/{total_individual_experiments_planned}): {exp_progress_label}")
        exp_start_time = time.time()

        try:
            experiment_function = getattr(module, func_name)
        except AttributeError:
            error_msg = f"Function '{func_name}' not found in module for algorithm '{algo}'. Skipping."
            print(f"!!! Error: {error_msg}")
            logger.error(f"Individual Experiment {exp_progress_label} SKIPPED: {error_msg}")
            overall_failed_experiments_details.append(f"{exp_progress_label} - Function not found")
            config_failed_list.append(f"{algo}/{dataset} - Function not found")
            continue

        func_args = config_params_for_function.copy()
        func_args["dataset"] = dataset
        func_args["num_rows"] = rows  # rows can be None, functions should handle

        logger.info(f"  Function: {module.__name__}.{func_name}")
        logger.info(f"  Arguments: {func_args}")  # Log the actual args being passed

        try:
            print(f"Calling: {module.__name__}.{func_name} for {header_label} (Rows: {rows if rows else 'All'})")
            experiment_function(**func_args)
            exp_end_time = time.time()
            duration = exp_end_time - exp_start_time
            print(f"\n--- Finished Individual Experiment {exp_progress_label} in {duration:.2f} seconds ---")
            logger.info(f"Finished Individual Experiment {exp_progress_label} successfully in {duration:.2f} seconds.")
            overall_completed_successfully += 1
            config_completed_count += 1
        except Exception as e:
            exp_end_time = time.time()
            duration = exp_end_time - exp_start_time
            error_traceback = traceback.format_exc()
            print(f"\n!!! Individual Experiment {exp_progress_label} FAILED after {duration:.2f} seconds.")
            print(f"!!! Error: {e}")
            logger.error(f"Individual Experiment {exp_progress_label} FAILED after {duration:.2f} seconds.")
            logger.error(f"Error details: {type(e).__name__}: {e}", exc_info=False)
            logger.error(f"Traceback:\n{error_traceback}")

            failure_detail = f"{exp_progress_label} - Error: {type(e).__name__}: {str(e)}"
            overall_failed_experiments_details.append(failure_detail)
            config_failed_list.append(f"{algo}/{dataset} - Error: {type(e).__name__}")
            # Decide whether to continue or break on error for the whole batch
            # break # Uncomment to stop all configurations after the first failure in an experiment

    # Summary for the current configuration run
    config_duration = time.time() - config_start_time
    logger.info(f"\n{'-' * 20} Finished Configuration Run: {run_label} {'-' * 20}")
    logger.info(f"Duration for configuration '{run_label}': {config_duration:.2f} seconds.")
    logger.info(f"Experiments completed successfully in this configuration: {config_completed_count}/{len(experiments_to_run)}")
    if config_failed_list:
        logger.warning(f"Failed experiments within configuration '{run_label}':")
        for failure in config_failed_list:
            logger.warning(f"  - {failure}")
    logger.info("-" * (40 + len(run_label) + 2))  # Match the start separator length


# --- Overall Summary ---
overall_end_time = time.time()
overall_duration = overall_end_time - overall_start_time

summary_header = "\n" + "=" * 80 + "\n"
summary_header += " All Experiment Configurations Finished. "
summary_header += f"Total Time: {overall_duration:.2f} seconds "
summary_header += "\n" + "=" * 80 + "\n"
print(summary_header)

logger.info("==================== Finished Experiment Batch ====================")
logger.info(f"Total Batch Duration: {overall_duration:.2f} seconds.")
logger.info(f"Total Individual Experiments Planned: {total_individual_experiments_planned}")
logger.info(f"Overall Individual Experiments Completed Successfully: {overall_completed_successfully}/{total_individual_experiments_planned}")

if overall_failed_experiments_details:
    logger.warning("Overall Failed Experiments (details):")
    for failure_detail_msg in overall_failed_experiments_details:
        logger.warning(f"  - {failure_detail_msg}")
else:
    logger.info("All planned individual experiments completed successfully across all configurations.")
logger.info("===================================================================")
