from rl_preprocess import preprocess_rl
from train_transe_model import train_transe_model_rl
from train_agent import train_agent_rl
from test_agent import test_agent_rl

def rl_experiment_loop(dataset, want_col, num_rows, ratio, seed):

    print("\n===== Stage 1: Preprocessing =====")
    preprocess_rl(
        dataset=dataset,
        want_col=want_col,
        num_rows=num_rows,
        ratio=ratio,
        seed=seed,
    )

    print("\n===== Stage 2: KGE Training =====")
    train_transe_model_rl(dataset=dataset)

    print("\n===== Stage 3: RL Agent Training =====")
    train_agent_rl(dataset=dataset)

    print("\n===== Stage 4: Testing & Evaluation =====")
    test_agent_rl(dataset=dataset)

    print("\n===== Pipeline Finished =====")