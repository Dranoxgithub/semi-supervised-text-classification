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
    parser.add_argument("--eval_freq", dest="eval_freq", type=int, default=100,
                        help="How frequent do we print the evaluation frequency for a batch currently in training")

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

    # adversarial training settings 
    parser.add_argument("--at_epsilon", dest="at_epsilon", type=float, default=5.0,
                        help="Pertubation hyperparameter for adversarial training")

    # virtual adversarial training settings 
    parser.add_argument("--vat_epsilon", dest="vat_epsilon", type=float, default=5.0,
                        help="Pertubation hyperparameter for virtual adversarial training")
    parser.add_argument("--hyperpara_for_vat", dest="hyperpara_for_vat", type=float, default=1e-1,
                        help="Pertubation hyperparameter for virtual adversarial training")
    # loss weights 
    parser.add_argument("--ml_loss_weight", dest="ml_loss_weight", type=float, default=1.0,
                        help="Weight for ml loss")
    parser.add_argument("--at_loss_weight", dest="at_loss_weight", type=float, default=1.0,
                        help="Weight for at loss")
    parser.add_argument("--vat_loss_weight", dest="vat_loss_weight", type=float, default=1.0,
                        help="Weight for vat loss")
    parser.add_argument("--EM_loss_weight", dest="EM_loss_weight", type=float, default=1.0,
                        help="Weight for EM loss")

    # whether to use weights
    parser.add_argument('--use_CE', dest='use_CE', action='store_true')
    parser.set_defaults(use_CE=False)

    parser.add_argument('--use_AT', dest='use_AT', action='store_true')
    parser.set_defaults(use_AT=False)

    parser.add_argument('--use_VAT', dest='use_VAT', action='store_true')
    parser.set_defaults(use_VAT=False)

    parser.add_argument('--use_EM', dest='use_EM', action='store_true')
    parser.set_defaults(use_EM=False)

    # logging
    parser.add_argument('--enable_logging', dest='enable_logging', action='store_true')
    parser.set_defaults(enable_logging=False)

    parser.add_argument("--logging_freq", dest="logging_freq", type=int, default=10,
                        help="logging_freq")

    args = parser.parse_args()

    dataloaders, dataset_lens = load_data(args.data_folder + "/" + args.dataset_name, args.words_per_batch)

    print(args)
    train_loader = dataloaders['unlabel']
    iter_un = iter(train_loader)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Trainer(dataloaders, dataset_lens, device, args)
    model.train()
