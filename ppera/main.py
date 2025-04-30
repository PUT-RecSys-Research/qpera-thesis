import CF
import CBF
import RL

# mlflow server --host 127.0.0.1 --port 8080
# remember to be in the right directory to run this script from the command line (cd ppera)

CF.cf_experiment_loop(TOP_K=10, dataset='movielens',
                        want_col=["userID", "itemID", "rating", "timestamp", 'title', 'genres'],
                        num_rows=10000,
                        ratio=0.75,
                        seed=42)

CBF.cbf_experiment_loop(TOP_K=10, dataset='movielens',
                        want_col=["userID", "itemID", "rating", "timestamp", 'title', 'genres'],
                        num_rows=10000,
                        ratio=0.75,
                        seed=42)

CF.cf_experiment_loop(TOP_K=10, dataset='amazonsales',
                        want_col=["userID", "itemID", "rating", "timestamp", 'title', 'genres'],
                        num_rows=1000,
                        ratio=0.75,
                        seed=42)

CBF.cbf_experiment_loop(TOP_K=10, dataset='amazonsales',
                        want_col=["userID", "itemID", "rating", "timestamp", 'title', 'genres'],
                        num_rows=1000,
                        ratio=0.75,
                        seed=42)

RL.rl_experiment_loop(dataset='movielens',
    want_col=["userID", "itemID", "rating", "timestamp", 'title', 'genres'],
    num_rows=10000,
    ratio=0.75,
    seed=42,
)