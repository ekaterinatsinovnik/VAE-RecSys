def set_template(config):
    if config.template is None:
        return

    elif config.template.startswith('als'):
        config.train = True
        config.test = True

        config.dataset = 'ml_1m'
        config.min_rating = 3.5
        config.min_uc = 5
        config.min_sc = 0
        config.test_size = 0.1

        config.factors = 100
        config.regularization = 0.01
        config.alpha = 1
        config.iterations = 20
        config.model_path = None
    
    elif config.template.startswith('ease'):
        config.train = True
        config.test = True

        config.dataset = 'ml_1m'
        config.min_rating = 3.5
        config.min_uc = 5
        config.min_sc = 0
        config.test_size = 0.1

        config.regularization = 500

    elif config.template.startswith('train_vae_search_beta'):
        config.train = True
        config.test = True

        config.dataset = 'ml_1m'
        config.min_rating = 3.5
        config.min_uc = 5
        config.min_sc = 0
        config.val_size = 0.1
        config.test_size = 0.1

        config.batch_size = 128

        config.latent_dim = 200,
        config.num_hidden = 1,
        config.hidden_dim = 600,
        config.dropout = 0.5,
        config.epochs_num = 50,
        config.learning_rate = 1e-3,
        config.weight_decay = 0.01,

        config.beta = None,
        config.anneal_cap = 1.0,
        config.total_anneal_steps = 2000,
        config.set_lr_scheduler=False
    
    elif config.template.startswith('train_vae_give_beta'):
        config.train = True
        config.test = True

        config.dataset = 'ml_1m'
        config.min_rating = 3.5
        config.min_uc = 5
        config.min_sc = 0
        config.val_size = 0.1
        config.test_size = 0.1

        config.batch_size = 128

        config.latent_dim = 200,
        config.num_hidden = 1,
        config.hidden_dim = 600,
        config.dropout = 0.5,
        config.epochs_num = 50,
        config.learning_rate = 1e-3,
        config.weight_decay = 0.01,

        config.beta = None,
        config.anneal_cap = 0.4,
        config.total_anneal_steps = 2000,
        config.set_lr_scheduler=False
