from rl_preprocess import preprocess_rl
from rl_train_transe_model import train_transe_model_rl
from rl_train_agent import train_agent_rl
from rl_test_agent import test_agent_rl

def rl_experiment_loop(TOP_K, dataset, want_col, num_rows, ratio, seed):

    print("\n===== Stage 1: Preprocessing =====")
    data_df, train_df, test_df = preprocess_rl(
        dataset=dataset,
        want_col=want_col,
        num_rows=num_rows,
        ratio=ratio,
        seed=seed,
    )

    print("\n===== Stage 2: KGE Training =====")
    train_transe_model_rl(dataset=dataset,seed=seed)

    print("\n===== Stage 3: RL Agent Training =====")
    train_agent_rl(dataset=dataset,seed=seed)

    print("\n===== Stage 4: Testing & Evaluation =====")
    test_agent_rl(dataset=dataset, TOP_K=TOP_K, want_col=want_col, num_rows=num_rows, ratio=ratio, seed=seed, data_df=data_df, train_df=train_df, test_df=test_df)

    print("\n===== Pipeline Finished =====")