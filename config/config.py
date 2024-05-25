from .parser import config

from src.dataloaders import ALSDataLoader, EASEDataLoader, VAEDataLoader
from src.recommenders import (ALSRecommender, EASERecommender,
                              MultiVAERecommender)


if config.template == "als":
    Recommender = ALSRecommender
    Loader = ALSDataLoader

    dataset_config = {
            "dataset": config.dataset,
            "min_rating": config.min_rating,
            "min_user_count": config.min_user_count,
            "min_item_count": config.min_item_count,
            "test_size": config.test_size,
        }

    recommender_config = {
            "dataset": config.dataset,
            "factors": config.factors,
            "regularization": config.regularization,
            "alpha": config.alpha,
            "iterations": config.iterations,
        }

elif config.template == "ease":
    Recommender = EASERecommender
    Loader = EASEDataLoader

    dataset_config = {
            "dataset": config.dataset,
            "min_rating": config.min_rating,
            "min_user_count": config.min_user_count,
            "min_item_count": config.min_item_count,
            "test_size": config.test_size
        }

    recommender_config = vars({"regularization": config.regularization})

else:
    Recommender = MultiVAERecommender
    Loader = VAEDataLoader

    dataset_config ={
            "dataset": config.dataset,
            "min_rating": config.min_rating,
            "min_user_count": config.min_user_count,
            "min_item_count": config.min_item_count,
            "batch_size": config.batch_size,
            "val_size": config.val_size,
            "test_size": config.test_size
        }

    recommender_config = {
            "dataset": config.dataset,
            "latent_dim": config.latent_dim,
            "num_hidden": config.num_hidden,
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
            "epochs_num": config.epochs_num,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "beta": config.beta,
            "anneal_cap": config.anneal_cap,
            "total_anneal_steps": config.total_anneal_steps
        }
