import logging
import os
import time
import traceback
from typing import Dict, List, Any, Optional

from . import CBF, CF, RL


def setup_logger(log_file: str = "experiment_runner.log", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger instance for experiment tracking."""
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger_instance = logging.getLogger("ExperimentRunner")
    logger_instance.setLevel(level)

    if not logger_instance.handlers:
        handler = logging.FileHandler(log_file, mode="a")
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger_instance.addHandler(handler)

    return logger_instance


def print_experiment_header(run_config_label: str, algorithm_name: str, dataset_name: str, num_rows: Optional[int]) -> None:
    """Print a formatted header for the experiment to console."""
    header = "\n" + "=" * 80 + "\n"
    header += f" Config: {run_config_label} | Algorithm: {algorithm_name} | Dataset: {dataset_name}"
    if num_rows:
        header += f" | Rows: {num_rows:,}"
    header += "\n" + "=" * 80 + "\n"
    print(header)


class ExperimentRunner:
    """Manages and executes recommendation system experiments with various configurations."""
    
    def __init__(self):
        """Initialize the experiment runner with logger and configurations."""
        self.logger = setup_logger()
        self.experiments = self._define_experiments()
        self.configurations = self._build_configurations()
        
    def _define_experiments(self) -> List[Dict[str, Any]]:
        """Define all algorithm-dataset combinations to run."""
        return [
            {"algo": "CBF", "module": CBF, "func": "cbf_experiment_loop", "dataset": "movielens", "rows": 14000},
            {"algo": "CF", "module": CF, "func": "cf_experiment_loop", "dataset": "movielens", "rows": 14000},
            {"algo": "RL", "module": RL, "func": "rl_experiment_loop", "dataset": "movielens", "rows": 14000},
            {"algo": "CBF", "module": CBF, "func": "cbf_experiment_loop", "dataset": "amazonsales"},
            {"algo": "CF", "module": CF, "func": "cf_experiment_loop", "dataset": "amazonsales"},
            {"algo": "RL", "module": RL, "func": "rl_experiment_loop", "dataset": "amazonsales"},
            {"algo": "CBF", "module": CBF, "func": "cbf_experiment_loop", "dataset": "postrecommendations", "rows": 14000},
            {"algo": "CF", "module": CF, "func": "cf_experiment_loop", "dataset": "postrecommendations", "rows": 14000},
            {"algo": "RL", "module": RL, "func": "rl_experiment_loop", "dataset": "postrecommendations", "rows": 14000},
        ]
    
    def _get_base_params(self) -> Dict[str, Any]:
        """Get base parameters common to all experiments."""
        return {
            "TOP_K": 10,
            "want_col": ["userID", "itemID", "rating", "timestamp", "title", "genres"],
            "ratio": 0.75,
            "seed": 42,
        }
    
    def _get_default_conditional_params(self) -> Dict[str, Any]:
        """Get default values for conditional experiment parameters."""
        return {
            "privacy": False,
            "hide_type": None,
            "columns_to_hide": None,
            "fraction_to_hide": None,
            "personalization": False,
            "fraction_to_change": None,
            "change_rating": False,
        }
    
    def _build_configurations(self) -> List[Dict[str, Any]]:
        """Build all parameter configurations for experiments."""
        base_params = self._get_base_params()
        default_conditional_params = self._get_default_conditional_params()
        configurations = []
        
        # 1. Clean/baseline configuration
        clear_config = {**base_params, **default_conditional_params}
        clear_config.update({
            "run_label": "Clean",
            "privacy": False,
            "personalization": False
        })
        configurations.append(clear_config)
        
        # 2. Privacy configurations
        privacy_fractions = [0.1, 0.25, 0.5, 0.8]
        for fraction in privacy_fractions:
            privacy_config = {**base_params, **default_conditional_params}
            privacy_config.update({
                "run_label": f"Privacy_{fraction:.2f}",
                "privacy": True,
                "hide_type": "values_in_column",
                "columns_to_hide": ["title", "genres"],
                "fraction_to_hide": fraction,
                "personalization": False,
            })
            configurations.append(privacy_config)
        
        # 3. Personalization configurations
        personalization_fractions = [0.1, 0.25, 0.5, 0.8]
        for fraction in personalization_fractions:
            personalization_config = {**base_params, **default_conditional_params}
            personalization_config.update({
                "run_label": f"Personalization_{fraction:.2f}",
                "privacy": False,
                "personalization": True,
                "fraction_to_change": fraction,
                "change_rating": True,
            })
            configurations.append(personalization_config)
        
        return configurations
    
    def _run_single_experiment(
        self, 
        experiment: Dict[str, Any], 
        config: Dict[str, Any],
        progress_info: Dict[str, int]
    ) -> bool:
        """
        Run a single experiment with the given configuration.
        
        Returns:
            True if successful, False if failed
        """
        algo = experiment["algo"]
        dataset = experiment["dataset"]
        rows = experiment.get("rows")
        module = experiment["module"]
        func_name = experiment["func"]
        
        # Create experiment labels
        run_label = config["run_label"]
        exp_progress_label = (
            f"Config {progress_info['config_idx']}/{progress_info['total_configs']} "
            f"[{run_label}] - Exp {progress_info['exp_idx']}/{progress_info['total_exps']} "
            f"({algo}/{dataset})"
        )
        
        print_experiment_header(run_label, algo, dataset, rows)
        self.logger.info(
            f"Starting Individual Experiment "
            f"({progress_info['current_exp']}/{progress_info['total_individual']}): "
            f"{exp_progress_label}"
        )
        
        start_time = time.time()
        
        try:
            # Get experiment function
            experiment_function = getattr(module, func_name)
        except AttributeError:
            error_msg = f"Function '{func_name}' not found in module for algorithm '{algo}'"
            self.logger.error(f"Individual Experiment {exp_progress_label} SKIPPED: {error_msg}")
            return False
        
        # Prepare function arguments
        func_args = {k: v for k, v in config.items() if k != "run_label"}
        func_args.update({
            "dataset": dataset,
            "num_rows": rows
        })
        
        self.logger.info(f"  Function: {module.__name__}.{func_name}")
        self.logger.info(f"  Arguments: {func_args}")
        
        try:
            print(f"Executing: {module.__name__}.{func_name} for {algo}/{dataset}")
            experiment_function(**func_args)
            
            duration = time.time() - start_time
            print(f"\n--- Completed {exp_progress_label} in {duration:.2f} seconds ---")
            self.logger.info(f"Completed {exp_progress_label} successfully in {duration:.2f} seconds")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            error_traceback = traceback.format_exc()
            
            print(f"\n!!! FAILED {exp_progress_label} after {duration:.2f} seconds")
            print(f"!!! Error: {e}")
            
            self.logger.error(f"FAILED {exp_progress_label} after {duration:.2f} seconds")
            self.logger.error(f"Error: {type(e).__name__}: {e}")
            self.logger.error(f"Traceback:\n{error_traceback}")
            return False
    
    def _run_configuration(
        self, 
        config: Dict[str, Any], 
        config_idx: int, 
        total_configs: int,
        current_exp_count: int
    ) -> tuple[int, int, List[str]]:
        """
        Run all experiments for a single configuration.
        
        Returns:
            Tuple of (successful_count, updated_current_exp_count, failed_experiments)
        """
        run_label = config["run_label"]
        self.logger.info(
            f"\n{'-' * 20} Starting Configuration {config_idx + 1}/{total_configs}: "
            f"{run_label} {'-' * 20}"
        )
        
        config_start_time = time.time()
        successful_count = 0
        failed_experiments = []
        total_individual = len(self.configurations) * len(self.experiments)
        
        for exp_idx, experiment in enumerate(self.experiments):
            current_exp_count += 1
            
            progress_info = {
                "config_idx": config_idx + 1,
                "total_configs": total_configs,
                "exp_idx": exp_idx + 1,
                "total_exps": len(self.experiments),
                "current_exp": current_exp_count,
                "total_individual": total_individual
            }
            
            success = self._run_single_experiment(experiment, config, progress_info)
            
            if success:
                successful_count += 1
            else:
                algo = experiment["algo"]
                dataset = experiment["dataset"]
                failed_experiments.append(f"{algo}/{dataset}")
        
        # Log configuration summary
        config_duration = time.time() - config_start_time
        self.logger.info(f"\n{'-' * 20} Finished Configuration: {run_label} {'-' * 20}")
        self.logger.info(f"Duration: {config_duration:.2f} seconds")
        self.logger.info(f"Success rate: {successful_count}/{len(self.experiments)}")
        
        if failed_experiments:
            self.logger.warning(f"Failed experiments in '{run_label}':")
            for failure in failed_experiments:
                self.logger.warning(f"  - {failure}")
        
        return successful_count, current_exp_count, failed_experiments
    
    def run_all_experiments(self) -> None:
        """Execute all experiments across all configurations."""
        self.logger.info("==================== Starting Experiment Batch ====================")
        
        overall_start_time = time.time()
        total_successful = 0
        all_failed_experiments = []
        current_exp_count = 0
        
        total_configs = len(self.configurations)
        total_individual = total_configs * len(self.experiments)
        
        print(f"\nStarting experiment batch:")
        print(f"  - Configurations: {total_configs}")
        print(f"  - Experiments per config: {len(self.experiments)}")
        print(f"  - Total individual experiments: {total_individual}")
        print(f"  - MLflow server should be running at: http://127.0.0.1:8080")
        
        for config_idx, config in enumerate(self.configurations):
            config_successful, current_exp_count, config_failed = self._run_configuration(
                config, config_idx, total_configs, current_exp_count
            )
            
            total_successful += config_successful
            all_failed_experiments.extend([
                f"{config['run_label']}: {failure}" for failure in config_failed
            ])
        
        # Print final summary
        self._print_final_summary(
            overall_start_time, total_successful, total_individual, all_failed_experiments
        )
    
    def _print_final_summary(
        self, 
        start_time: float, 
        total_successful: int, 
        total_individual: int, 
        failed_experiments: List[str]
    ) -> None:
        """Print and log final experiment summary."""
        total_duration = time.time() - start_time
        
        summary_header = "\n" + "=" * 80 + "\n"
        summary_header += f" Experiment Batch Complete - Duration: {total_duration:.2f}s "
        summary_header += f"| Success: {total_successful}/{total_individual} "
        summary_header += "\n" + "=" * 80 + "\n"
        print(summary_header)
        
        self.logger.info("==================== Experiment Batch Complete ====================")
        self.logger.info(f"Total Duration: {total_duration:.2f} seconds")
        self.logger.info(f"Total Success Rate: {total_successful}/{total_individual}")
        
        if failed_experiments:
            self.logger.warning("Failed Experiments Summary:")
            for failure in failed_experiments:
                self.logger.warning(f"  - {failure}")
            print(f"\nFailed experiments: {len(failed_experiments)}")
            print("Check experiment_runner.log for detailed error information")
        else:
            self.logger.info("All experiments completed successfully!")
            print("All experiments completed successfully!")


def main():
    """Main entry point for running experiments."""
    print("Personalization, Privacy, and Explainability of Recommendation Algorithms")
    print("=" * 80)
    print("\nNote: Ensure MLflow server is running:")
    print("  mlflow server --host 127.0.0.1 --port 8080")
    print("  Run this script from the project root directory")
    
    runner = ExperimentRunner()
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
