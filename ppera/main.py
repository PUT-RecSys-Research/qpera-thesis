import CF
import CBF


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


# remember to be in the right directory to run this script from the command line (cd ppera)
