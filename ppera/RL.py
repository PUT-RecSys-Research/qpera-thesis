from rl_preprocess import preprocess_rl
from rl_train_transe_model import train_transe_model_rl
from rl_train_agent import train_agent_rl
from rl_test_agent import test_agent_rl

def rl_experiment_loop(TOP_K, dataset, want_col, num_rows, ratio, seed, 
                       personalization=False, fraction_to_change = 0, change_rating = False, privacy=False, hide_type="values_in_column", columns_to_hide=None, fraction_to_hide = 0, records_to_hide=None):

    print("\n===== Stage 1: Preprocessing =====")
    data_df, train_df, test_df = preprocess_rl(
        dataset=dataset,
        want_col=want_col,
        num_rows=num_rows,
        ratio=ratio,
        seed=seed,
        personalization=personalization,
        fraction_to_change=fraction_to_change,
        change_rating=change_rating,
        privacy = privacy,
        hide_type=hide_type,
        columns_to_hide=columns_to_hide,
        fraction_to_hide=fraction_to_hide,
        records_to_hide=records_to_hide,
    )

    print("\n===== Stage 2: KGE Training =====")
    train_transe_model_rl(dataset=dataset,seed=seed)

    print("\n===== Stage 3: RL Agent Training =====")
    train_agent_rl(dataset=dataset,seed=seed)

    print("\n===== Stage 4: Testing & Evaluation =====")
    test_agent_rl(dataset=dataset, TOP_K=TOP_K, want_col=want_col, num_rows=num_rows, ratio=ratio, seed=seed, data_df=data_df, train_df=train_df, test_df=test_df, privacy=privacy, fraction_to_hide=fraction_to_hide, personalization=personalization, fraction_to_change=fraction_to_change)

    print("\n===== Pipeline Finished =====")