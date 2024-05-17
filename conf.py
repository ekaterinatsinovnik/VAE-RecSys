import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--train", type=bool, default=True)
parser.add_argument("--test", type=bool, default=True)

parser.add_argument("--dataset", type=str, default="ml_1m")
parser.add_argument("--min_rating", type=float, default=3.5)
parser.add_argument("--min_user_count",type=int, default=5)
parser.add_argument("--min_item_count", type=int, default=0)
parser.add_argument("--val_size", type=float, default=0.1)
parser.add_argument("--test_size", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=512)

parser.add_argument("--model", type=str, default="vae")

parser.add_argument("--epochs_num", type=int, default=50)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--weight_decay", type=float, default=0.01)

parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--latent_dim", type=int, default=200)
parser.add_argument("--hidden_dim", type=int, default=600)
parser.add_argument("--num_hidden", type=int, default=1)

parser.add_argument("--beta", type=None, default=None)
parser.add_argument("--anneal_cap", type=float, default=1.0)
parser.add_argument("--total_anneal_steps", type=int, default=2000)

parser.add_argument("--model_path", type=str, default=None)

parser.add_argument("--ks", type=list, default=[1, 5, 10, 20])

config = parser.parse_args()
