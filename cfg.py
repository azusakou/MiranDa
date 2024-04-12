class CFG:
    track_logging = True

    dataset_list = ['mimic3']#, 'eicu', 'mimic4']
    model_list = ['transformer']
    batch_size = 512
    epochs = 50
    val_start_epoch = 5
    trials_for_each_model = 30

    optuna = False
    optuna_trials=1000

    DEMO_mode = False

    train_all_models = False

    save_memory = True
    save_memory_del_features = ["texts"]

    train_the_last = False
    train_the_last_last_features = ["conditions", "procedures"]

    read_data = True
