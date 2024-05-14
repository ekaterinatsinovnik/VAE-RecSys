import torch
from clearml import Task

from src.dataloaders import VAEDataLoader
from src.trainers import MultiVAETrainer


def main():
    torch.manual_seed(0)
    # Task.set_random_seed(0)
    task = Task.init(
        project_name="VAE-RecSys",
        task_name="vae_test",
        task_type="training",
        auto_connect_frameworks={"pytorch": ['*.pth']},
        output_uri="models",  # сделать проверку на существование директории и создатть, если нет
        # /home/kate/drive/files/diploma/VAE-RecSys/logs
        tags=["ml-1m"],
        auto_connect_streams={"stdout": True, "stderr": True, "logging": True},
        # auto_resource_monitoring=False
    )
    task.name += " {}".format(task.id)

    args = {
        "dataset_shortname": "ml_1m",
        "min_rating": 3.5,
        "min_user_count": 5,  # default 5
        "min_item_count": 0,
    }

    loader = VAEDataLoader(batch_size=512)
    task.connect(loader)
    item_num, dataloader, interactions = loader.get_dataloaders(**args)

    trainer = MultiVAETrainer(item_num, beta=None, epochs_num=5)
    task.connect(trainer)
    trainer.train(dataloader, interactions['train'], interactions['val'])
    trainer.test(dataloader, interactions['test'], ks=[1, 5, 10, 20])

    task.close()


# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
# подумать по поводу логгера: захватывает ли клирмл логи из обычного логгера и нужен ли тензорборд (сранвить)?

if __name__ == "__main__":
    main()
