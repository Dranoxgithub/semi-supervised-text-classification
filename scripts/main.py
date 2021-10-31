import dataloader
import torch
import argparse as ap
from train import BiLSTM

def load_data(dataset_file, num_token_per_batch=3000):
    return dataloader.load(dataset_file + ".train.pkl", dataset_file + ".valid.pkl", dataset_file + ".test.pkl", dataset_file + ".unlabel.pkl", num_token_per_batch)

if __name__ == '__main__':
    # parse arguments
    parser = ap.ArgumentParser(description="Use this file with the required command line arguments to train a model and log metrics to wandb.ai")

    parser.add_argument("--data_folder", dest="path", help="path to train dataset")
    parser.add_argument("--dataset", dest="dataset_name", help="path to train dataset")

    parser.add_argument("--random_seed", dest="random_seed", type=int, required=True, help="random seed for reproducibility")

    # # hyperparameters
    parser.add_argument("--batch_size", dest="batch_size", type=int, required=True, help="batch size")
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, required=True, help="number of training epochs")
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, default=0.3, help="weight decay")
    parser.add_argument("--num_labels", dest="num_labels", type=int, default=5, help="number of labels in multilabel task")

    args = parser.parse_args()

    dataset = load_data(args.path + "/" + args.dataset_name, 3000)
    hidden_dim = 512
    class_size = 2
    embeddings = torch.rand(114426, 300)
    model = BiLSTM(hidden_dim, class_size, embeddings, dropout=0.5)
    model.train(dataset['train'])
    # for i, batch_list in enumerate(dataset['train']):
    #     print(batch_list)
    #     print(i)
    #
    #     break
