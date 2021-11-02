import dataloader
import torch
import argparse as ap
from train import Trainer


def load_data(dataset_file, num_token_per_batch=3000):
    return dataloader.load(dataset_file + ".train.pkl", dataset_file + ".valid.pkl", dataset_file + ".test.pkl",
                           dataset_file + ".unlabel.pkl", num_token_per_batch)


if __name__ == '__main__':
    # parse arguments
    parser = ap.ArgumentParser(
        description="Use this file with the required command line arguments to train a model and log metrics to wandb.ai")

    parser.add_argument("--data_folder", dest="data_folder", help="path to train dataset")
    parser.add_argument("--dataset_name", dest="dataset_name", help="path to train dataset")

    parser.add_argument("--random_seed", dest="random_seed", type=int, required=True,
                        help="random seed for reproducibility")

    # # training parameters
    parser.add_argument("--words_per_batch", dest="words_per_batch", type=int, required=True, help="words_per_batch")
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, required=True, help="number of training epochs")
    parser.add_argument("--lr", dest="lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--num_labels", dest="num_labels", type=int, default=2,
                        help="number of labels in multilabel task")
    parser.add_argument("--gradient_clip_value", dest="gradient_clip_value", type=float, default=5,
                        help="gradient_clip_value")

    # embedding settings
    parser.add_argument("--embedding_dropout", dest="embedding_dropout", type=float, default=0.5,
                        help="embedding dropout chance")

    # LSTM settings
    parser.add_argument("--LSTM_input_dropout", dest="LSTM_input_dropout", type=float, default=0.5,
                        help="LSTM_input")
    parser.add_argument("--LSTM_input_dim", dest="LSTM_input_dim", type=int, default=300,
                        help="LSTM_input_dim")
    parser.add_argument("--LSTM_hidden_dim", dest="LSTM_hidden_dim", type=int, default=512,
                        help="LSTM_hidden_dim")
    parser.add_argument("--LSTM_num_layers", dest="LSTM_num_layers", type=int, default=1,
                        help="LSTM_num_layers")
    parser.add_argument("--LSTM_internal_dropout", dest="LSTM_internal_dropout", type=float, default=0.0,
                        help="LSTM_internal_dropout")
    parser.add_argument("--LSTM_output_dropout", dest="LSTM_output_dropout", type=float, default=0.5,
                        help="LSTM_output_dropout")

    args = parser.parse_args()

    dataloaders, dataset_lens = load_data(args.data_folder + "/" + args.dataset_name, args.words_per_batch)
    cuda = torch.device('cuda')
    # hidden_dim = 512
    # class_size = 2
    # embeddings = torch.rand(114426, 300)
    model = Trainer(dataloaders['train'], dataset_lens, cuda, args)
    model.train()

    # for i, batch_list in enumerate(dataset['train']):
    #     print(batch_list)
    #     print(i)
    #
    #     break
