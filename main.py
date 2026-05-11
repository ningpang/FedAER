import argparse
import logging
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class logging_Config:
    def __init__(self, name, log_dir):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()
        os.makedirs(log_dir, exist_ok=True)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s -> %(message)s')
        handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        self.logger.addHandler(handler)

    def get_config(self):
        return self.logger


def format_alpha(alpha):
    alpha = str(alpha).strip()
    try:
        value = float(alpha)
    except ValueError:
        return alpha
    return str(int(value)) if value.is_integer() else format(value, 'g')


def configure_partition_args(args):
    supported = {'fewrel': {'niid_label', 'dir_label', 'niid_quantity'}, 'tacred': {'niid_label', 'dir_label', 'niid_quantity'}, 'nyt': {'niid_label', 'dir_label', 'niid_quantity'}}
    defaults = {'niid_label': '2', 'dir_label': '0.1', 'niid_quantity': '1'}
    if args.niid not in supported[args.dataset_name]:
        raise ValueError(f"Dataset '{args.dataset_name}' does not support partition mode '{args.niid}'.")
    if args.alpha is None:
        args.alpha = defaults[args.niid]
    args.alpha = format_alpha(args.alpha)
    if args.niid == 'niid_label' and args.alpha not in {'2', '4', '8'}:
        raise ValueError('niid_label only supports alpha in {2,4,8}.')
    if args.niid in {'dir_label', 'niid_quantity'} and float(args.alpha) <= 0:
        raise ValueError(f'{args.niid} requires positive alpha, got {args.alpha}.')
    args.partition_name = f'{args.niid}_{args.alpha}'
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser('arguments for model')
    parser.add_argument('--dataset_name', default='fewrel', choices=['fewrel', 'tacred', 'nyt'])
    parser.add_argument('--pad_to_max_length', default=True)
    parser.add_argument('--max_seq_length', default=200)
    parser.add_argument('--overwrite_cache', default=False)
    parser.add_argument('--niid', default='niid_label', choices=['niid_label', 'dir_label', 'niid_quantity'])
    parser.add_argument('--alpha', type=str, default=None)
    parser.add_argument('--algorithm', type=str, default='FedAER', choices=['FedAER', 'FedAvg', 'FedProx', 'MOON', 'Per_FedAvg', 'Ditto', 'pFedMe', 'FedFomo', 'FedPAC'])
    parser.add_argument('--mu', type=int, default=1)
    parser.add_argument('--tau', type=int, default=0.5)
    parser.add_argument('--beta', type=int, default=0.001)
    parser.add_argument('--personalized_epochs', type=int, default=5)
    parser.add_argument('--ditto_lambda', type=int, default=0.1)
    parser.add_argument('--beta_pFedMe', type=int, default=2)
    parser.add_argument('--pFedMe_lambda', type=int, default=30)
    parser.add_argument('--FedFomo_M', type=int, default=5)
    parser.add_argument('--FedFomo_epsilon', type=float, default=0.3)
    parser.add_argument('--use_marker', action='store_true', default=True)
    parser.add_argument('--use_template', action='store_true', default=False)
    parser.add_argument('--communication_rounds', type=int, default=50)
    parser.add_argument('--train_epochs', type=int, default=5)
    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--do_test', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='lr for bitfit')
    parser.add_argument('--bert_learning_rate', type=float, default=1e-5)
    parser.add_argument('--sample_rate', type=float, default=1)
    parser.add_argument('--model_name_or_path', default='/home/bd/data/zs/data/bert-base-uncased')
    parser.add_argument('--use_fast_tokenizer', default=True)
    parser.add_argument('--model_revision', default='main')
    parser.add_argument('--prefix', action='store_true', default=False)
    parser.add_argument('--prompt', action='store_true', default=False)
    parser.add_argument('--hardprompt', action='store_true', default=False)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--bitfit', action='store_true', default=False)
    parser.add_argument('--bitfitmlm', action='store_true', default=False)
    parser.add_argument('--lora', action='store_true', default=False)
    parser.add_argument('--loramlm', action='store_true', default=False)
    parser.add_argument('--adapter', action='store_true', default=False)
    parser.add_argument('--adaptermlm', action='store_true', default=False)
    parser.add_argument('--prefixmlm', action='store_true', default=False)
    parser.add_argument('--promptmlm', action='store_true', default=False)
    parser.add_argument('--entity_loss', action='store_true', default=False)
    parser.add_argument('--proto_loss', action='store_true', default=False)
    parser.add_argument('--hidden_dropout_prob', default=0.1, type=float)
    parser.add_argument('--pre_seq_len', default=20, type=int)
    parser.add_argument('--prefix_projection', default=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_clients', type=int, default=10, help='<10')
    parser.add_argument('--federated_mode', default='part', type=str, choices=['part', 'partv1', 'partv2', 'partv3'])
    parser.add_argument('--output_root', default='output')

    args = configure_partition_args(parser.parse_args())
    args.peft_name = 'unknown'
    if args.prefix:
        args.peft_name = 'prefix'
    if args.bitfit:
        args.peft_name = 'bitfit'
    if args.lora:
        args.peft_name = 'lora'
    if args.adapter:
        args.peft_name = 'adapter'

    plm_name = 'base' if 'base' in args.model_name_or_path else 'large'
    args.log_dir = os.path.join(args.output_root, 'logs')
    args.result_dir = os.path.join(args.output_root, 'results')
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    log_file = f'{args.algorithm}-{args.dataset_name}-{args.niid}[{args.alpha}]-{args.peft_name}[{plm_name}]-{args.federated_mode}-epoch[{args.train_epochs}]-sample[{args.sample_rate}]'
    if args.entity_loss:
        log_file += '-entity'
    if args.proto_loss:
        log_file += '-relation'
    logger = logging_Config(log_file, args.log_dir).get_config()

    if args.algorithm == 'FedAER':
        from get_trainer import get_FL_trainer
    elif args.algorithm == 'FedAvg':
        from baselines.FedAvg import get_FL_trainer
    elif args.algorithm == 'FedProx':
        from baselines.FedProx import get_FL_trainer
    elif args.algorithm == 'MOON':
        from baselines.MOON import get_FL_trainer
    elif args.algorithm == 'Per_FedAvg':
        from baselines.Per_FedAvg import get_FL_trainer
    elif args.algorithm == 'Ditto':
        from baselines.Ditto import get_FL_trainer
    elif args.algorithm == 'pFedMe':
        from baselines.pFedMe import get_FL_trainer
    elif args.algorithm == 'FedFomo':
        from baselines.FedFomo import get_FL_trainer
    elif args.algorithm == 'FedPAC':
        from baselines.FedPAC import get_FL_trainer

    get_FL_trainer(args, logger).train(log_file)
