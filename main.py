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
        auto_connect_frameworks={"pytorch": ['*.pth'], "detect_repository": False},
        output_uri="models",  # сделать проверку на существование директории и создатть, если нет
        # /home/kate/drive/files/diploma/VAE-RecSys/logs
        tags=["ml-1m"],
        auto_connect_streams={"stdout": False, "stderr": False, "logging": True},
        auto_resource_monitoring=False
    )
    task.name += " {}".format(task.id)
    

    args = {
        "dataset_shortname": "ml_1m",
        "min_rating": 3.5,
        "min_user_count": 5,  # default 5
        "min_item_count": 0,
    }

    # loader = VAEDataLoader(**args, val_size=0.1, test_size=0.1, batch_size=128)
    loader = VAEDataLoader(**args, val_size=1, test_size=1, batch_size=128)
    task.connect(loader)
    dataloader, interactions = loader.get_dataloaders()

    item_num = interactions['train'].shape[1]
    trainer = MultiVAETrainer(item_num, dropout=0.22, learning_rate=0.01, weight_decay= 0.01, beta=0.342, epochs_num=50)
    # trainer = MultiVAETrainer(item_num, dropout=0.5, beta=0.342, learning_rate=0.001, 
    #     weight_decay= 0.01, epochs_num=200, total_anneal_steps = 3000, num_hidden = 2)
    task.connect(trainer)
#     print(task.get_parameters())
# 'General/batch_size'
# 'General/latent_dim': '200', 'General/hidden_dim'
# 'General/num_hidden': '1', 'General/dropout': '0.1', 'General/learning_rate': '0.002', 'General/epochs_num':
    trainer.train(dataloader, interactions['train'], interactions['val'])
    trainer.test(dataloader, interactions['test'], ks=[1, 5, 10, 20])
    # trainer.test(dataloader, interactions['test'], ks=[1, 5, 10, 20], model_path="models/ml-1m/val3/multivae_final.pth")

    task.close()


# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
# подумать по поводу логгера: захватывает ли клирмл логи из обычного логгера и нужен ли тензорборд (сранвить)?

if __name__ == "__main__":
    main()
