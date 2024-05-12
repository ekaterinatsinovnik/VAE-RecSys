from clearml import Task

from src.dataloaders import VAEDataLoader
from src.trainers import MultiVAETrainer


def main():
    task = Task.init(
        project_name="VAE-RecSys",
        task_name="vae_test",
        task_type="training",
        auto_connect_frameworks={"pytorch": True},
        output_uri="models",  # сделать проверку на существование директории и создатть, если нет
        # /home/kate/drive/files/diploma/VAE-RecSys/logs
        tags=["ml-1m"],
        auto_connect_streams={"stdout": False, "stderr": True, "logging": True},
    )
    task.name += " {}".format(task.id)

    args = {
        "dataset_shortname": "ml_1m",
        "min_rating": 3.5,
        "min_user_count": 5,  # default 5
        "min_item_count": 0,
    }
    train, val, test = VAEDataLoader(train_batch_size=512).get_dataloaders(**args)
    print(train.dataset.shape[0])
    # trainer = MultiVAETrainer(train.dataset.shape[0])
    trainer = MultiVAETrainer(train.dataset.shape[1], beta=None, epochs_num=5)
    print(trainer.model)
    trainer.train(train, val)

    task.close()


# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
# подумать по поводу логгера: захватывает ли клирмл логи из обычного логгера и нужен ли тензорборд (сранвить)?
# посмотреть логгер в корейском коде
# подумать, как сохранять модель (по какому пути)

if __name__ == "__main__":
    main()
