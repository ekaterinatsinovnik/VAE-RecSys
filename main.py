import torch
from clearml import Task

from config.config import (config, 
            dataset_config, Loader, 
            recommender_config, Recommender)


def main():
    torch.manual_seed(0)

    if 'vae' in config.template:

        task = Task.init(
            project_name="VAE-RecSys",
            task_name=f"{config.template}_{config.dataset}",
            task_type="training" if config.train else "testing",
            auto_connect_frameworks={"pytorch": ["*.pth"], "detect_repository": False},
            output_uri="models",
            tags=[config.dataset],
            auto_connect_streams={"stdout": False, "stderr": False, "logging": True},
            auto_resource_monitoring=False
        )

    loader = Loader(**dataset_config)
    dataloader, interactions, item_num = loader.get_dataloaders()

    if 'vae' in config.template:
        recommender_config['item_num'] = item_num
        
    recommender = Recommender(**recommender_config)

    if config.train:
        recommender.train(dataloader, interactions)
    if config.test:
        recommender.test(dataloader, interactions["test"], ks=config.ks)

    if 'vae' in config.template:
        task.close()


if __name__ == "__main__":
    main()
