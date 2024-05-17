import torch
from clearml import Task

from conf import config
from src.dataloaders import VAEDataLoader
from src.recommenders import MultiVAERecommender


def main():
    torch.manual_seed(0)

    task = Task.init(
        project_name="VAE-RecSys",
        task_name=f"{config.model}_{config.dataset}",
        task_type="training" if config.train else "testing",
        auto_connect_frameworks={"pytorch": ['*.pth'], "detect_repository": False},
        output_uri="models",
        tags=[config.dataset],
        auto_connect_streams={"stdout": False, "stderr": False, "logging": True},
        auto_resource_monitoring=False
    )
    
    dataset_args = {
        "dataset" : config.dataset,
        "min_rating" : config.min_rating,
        "min_user_count" : config.min_user_count,
        "min_item_count" : config.min_item_count,
        "val_size" : config.val_size,
        "test_size" : config.test_size,
        "batch_size" : config.batch_size
        }
    dataset = VAEDataLoader(**dataset_args)
    dataloader, interactions = dataset.get_dataloaders()

    item_num = interactions['train'].shape[1]
    recommender_args = {
        "dataset" : config.dataset,
        "epochs_num" : config.epochs_num,
        "learning_rate" : config.learning_rate,
        "weight_decay" : config.weight_decay,
        "dropout" : config.dropout,
        "latent_dim" : config.latent_dim,
        "hidden_dim" : config.hidden_dim,
        "num_hidden" : config.num_hidden,
        "beta" : config.beta,
        "anneal_cap" : config.anneal_cap,
        "total_anneal_steps" : config.total_anneal_steps,
        }

    recommender = MultiVAERecommender(item_num, **recommender_args)

    if config.train:
        recommender.train(dataloader, interactions['train'], interactions['val'])
    if config.test:
        recommender.test(dataloader, interactions['test'], ks=config.ks, path_to_load=config.model_path)

    task.close()


if __name__ == "__main__":
    main()
