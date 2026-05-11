import os.path
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from model.utils import get_model, TaskType
from tasks.dataset import ReDataset
from training.trainer_base import BaseTrainer
from utils import graph_aggregate
from copy import deepcopy
import numpy as np

class client(object):
    def __init__(self, args, tokenizer, dataset, model, cid):
        # print("")
        # print(f'Initializing the {cid}-th client ...')
        # print("")
        self.cid = cid
        self.args = args
        self.dataset = dataset
        self.args.mask_id = tokenizer.mask_token_id
        self.model = model

        self.args.hidden_size = self.args.prefix_hidden_size


        self.trainer = BaseTrainer(
            args=self.args,
            device=self.model.device,
            dataset=dataset,
            tokenizer=tokenizer
        )

        # if self.model_args.prefixmlm:
        #     self.trainer = OurTrainer(
        #         args=self.training_args,
        #         device=self.model.device,
        #         dataset=dataset,
        #         tokenizer=tokenizer
        #     )
        # else:
        #     self.trainer = BaseTrainer(
        #         args=self.training_args,
        #         device=self.model.device,
        #         dataset=dataset,
        #         tokenizer=tokenizer
        #     )

    def train(self, global_model, personalized_model):
        print("")
        print(f"Training the {self.cid}-th client")
        results = self.trainer.train(self.model, global_model, personalized_model)
        return results

    def evaluate(self):
        results = self.trainer.evaluate(self.model)
        return results

    def get_state_dict(self):
        return deepcopy(self.model.state_dict())



class get_FL_trainer(object):

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.device = self.args.device
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path,
            use_fast=self.args.use_fast_tokenizer,
            revision=self.args.model_revision,
            never_split=['[E11]', '[E12]', '[E21]', '[E22]']
        )

        special_tokens_dict = {'additional_special_tokens': ['[E11]', '[E12]', '[E21]', '[E22]']}  # add special token
        tokenizer.add_special_tokens(special_tokens_dict)

        datasets = [ReDataset(tokenizer, self.args, cid) for cid in range(self.args.num_clients)]

        config = AutoConfig.from_pretrained(
            self.args.model_name_or_path,
            num_labels=datasets[0].num_labels,
            label2id=datasets[0].label2id,
            id2label=datasets[0].id2label,
            finetuning_task=self.args.dataset_name,
            revision=self.args.model_revision,
        )
        config.num_entities = len(datasets[0].entity2id)

        self.args.mask_id = tokenizer.mask_token_id
        self.model = get_model(self.args, self.args, TaskType.SEQUENCE_CLASSIFICATION, config)

        self.target_parameter_names = []

        for name, params in self.model.named_parameters():
            if params.requires_grad == True:
                self.target_parameter_names.append(name)


        self.client_entity_id_sets = self._collect_client_entity_id_sets(datasets)
        self.shared_entity_ids, self.client_shared_entity_ids = self._build_shared_entity_cache(self.client_entity_id_sets)

        model_list = [deepcopy(self.model) for cid in range(self.args.num_clients)]

        self.A = np.ones([self.args.num_clients, self.args.num_clients])
        self.clients = [client(args, tokenizer, datasets[cid], model_list[cid], cid) for cid in range(self.args.num_clients)]
        self.cos = nn.CosineSimilarity(dim=-1)

    def _collect_client_entity_id_sets(self, datasets):
        client_entity_id_sets = []
        for dataset in datasets:
            local_ids = set()
            for field in ('hid', 'tid'):
                if field not in dataset.train_dataset.column_names:
                    continue
                for entity_id in dataset.train_dataset[field]:
                    entity_id = int(entity_id)
                    if 0 <= entity_id < dataset.num_entities:
                        local_ids.add(entity_id)
            client_entity_id_sets.append(local_ids)
        return client_entity_id_sets

    def _build_shared_entity_cache(self, client_entity_id_sets):
        entity_frequency = {}
        for local_ids in client_entity_id_sets:
            for entity_id in local_ids:
                entity_frequency[entity_id] = entity_frequency.get(entity_id, 0) + 1

        shared_entity_ids = sorted(entity_id for entity_id, freq in entity_frequency.items() if freq > 1)
        shared_entity_id_set = set(shared_entity_ids)
        client_shared_entity_ids = [
            sorted(entity_id for entity_id in local_ids if entity_id in shared_entity_id_set)
            for local_ids in client_entity_id_sets
        ]
        return shared_entity_ids, client_shared_entity_ids


    def train(self, path):
        best_round = 0
        best_weighted_f1 = 0
        best_accuracy = 0
        best_performance = None
        current_performance = {idx: {'Accuracy': 0, 'weighted-F1': 0, 'macro-F1': 0, 'mac-F1': 0} for idx in range(self.args.num_clients)}
        save_results = []
        save_macro_results = []
        print('Starting federated learning ...')
        # 初始化一些需要用到的权重
        global_model = deepcopy(self.model)
        personalized_model = []
        for cid in range(self.args.num_clients):
            personalized_model.append(deepcopy(self.clients[cid].model))

        for epoch in range(self.args.communication_rounds):
            print("")
            self.logger.info('======== Round {:} / {:} ========'.format(epoch + 1, self.args.communication_rounds))
            print('======== Round {:} / {:} ========'.format(epoch + 1, self.args.communication_rounds))
            print('Training ...')

            active_clients = np.random.choice(range(self.args.num_clients), int(self.args.num_clients*self.args.sample_rate), replace=False)
            active_clients.sort()
            if epoch == 0:
                active_clients = range(self.args.num_clients)

            for cid in active_clients:
                results = self.clients[cid].train(global_model, personalized_model[cid])
                current_performance[cid] = results

            current_f1s = []
            current_macro_f1s = []
            for cid in current_performance:
                current_f1s.append(current_performance[cid].get('weighted-F1', current_performance[cid]['mac-F1']))
                current_macro_f1s.append(current_performance[cid].get('macro-F1', current_performance[cid]['mac-F1']))
            save_results.append(current_f1s)
            save_macro_results.append(current_macro_f1s)
            df = pd.DataFrame(save_results)
            macro_df = pd.DataFrame(save_macro_results)
            basepath = os.path.join(self.args.result_dir, f'{self.args.peft_name}_{self.args.dataset_name}_{self.args.algorithm}')
            os.makedirs(basepath, exist_ok=True)
            df.to_csv(os.path.join(basepath, f'{path}.csv'), index=False, header=False)
            macro_df.to_csv(os.path.join(basepath, f'{path}_macro.csv'), index=False, header=False)


            total_weighted_f1 = 0
            total_accuracy = 0
            for cid in range(len(current_performance)):
                total_weighted_f1 += current_performance[cid].get('weighted-F1', current_performance[cid]['mac-F1'])
                total_accuracy += current_performance[cid]['Accuracy']

            average_weighted_f1 = round(total_weighted_f1 / len(current_performance), 4)
            average_accuracy = round(total_accuracy / len(current_performance), 4)
            if average_weighted_f1 > best_weighted_f1:
                best_weighted_f1 = average_weighted_f1
                best_accuracy = average_accuracy
                best_round = epoch
                best_performance = current_performance
            self.logger.info(current_performance)
            self.logger.info("  Best results: Round = {0}, Acc = {1:.4f}, Weighted-F1 = {2:.4f}".format(best_round+1, best_accuracy, best_weighted_f1))
            print("")
            print("  Current results: ")
            print(current_performance)
            print("  Best results: Round = {0}, Acc = {1:.4f}, Weighted-F1 = {2:.4f}".format(best_round+1, best_accuracy, best_weighted_f1))
            print(best_performance)

            if self.args.federated_mode == 'part':
                self.update_target_parameters()
                self.update_entity_embeddings()
                self.update_relation_embeddings()
            elif self.args.federated_mode == 'partv1':
                self.update_target_parameters()
                # self.update_entity_embeddings()
                self.update_relation_embeddings()
            elif self.args.federated_mode == 'partv2':
                self.update_target_parameters()
                self.update_entity_embeddings()
                # self.update_relation_embeddings()
            elif self.args.federated_mode == 'partv3':
                self.update_target_parameters()
    
            else:
                print("There is no federation ...")
        print('Federated learning stopped.')

  
    def update_target_parameters(self):
        target_parameter_names = self.target_parameter_names
        weight_global = {name: None for name in target_parameter_names}

        for cid in range(self.args.num_clients):
            named_parameters = deepcopy(self.clients[cid].model.state_dict())
            if list(weight_global.values())[0] is None:
                for name in target_parameter_names:
                    weight_global[name] = named_parameters[name]
            else:
                for name in target_parameter_names:
                    weight_global[name] += named_parameters[name]

        for name in weight_global:
            weight_global[name] = torch.div(weight_global[name], self.args.num_clients)

        for cid in range(self.args.num_clients):
            named_parameters = deepcopy(self.clients[cid].model.state_dict())
            for name in target_parameter_names:
                named_parameters[name] = weight_global[name]
            self.clients[cid].model.load_state_dict(deepcopy(named_parameters))

    def update_entity_embeddings(self):
        if not self.shared_entity_ids:
            return

        hidden_size = self.clients[0].model.entity_embedding.size(-1)
        shared_position = {entity_id: pos for pos, entity_id in enumerate(self.shared_entity_ids)}
        global_entity_embeddings = torch.zeros(
            (len(self.shared_entity_ids), hidden_size),
            dtype=self.clients[0].model.entity_embedding.dtype,
            device=self.device,
        )
        global_entity_counts = torch.zeros(
            (len(self.shared_entity_ids), 1),
            dtype=self.clients[0].model.entity_embedding.dtype,
            device=self.device,
        )

        for cid in range(self.args.num_clients):
            local_shared_ids = self.client_shared_entity_ids[cid]
            if not local_shared_ids:
                continue
            local_entity_index = torch.tensor(local_shared_ids, dtype=torch.long, device=self.device)
            local_shared_position = torch.tensor(
                [shared_position[entity_id] for entity_id in local_shared_ids],
                dtype=torch.long,
                device=self.device,
            )
            current_entity_embedding = self.clients[cid].model.entity_embedding.index_select(0, local_entity_index)
            global_entity_embeddings.index_add_(0, local_shared_position, current_entity_embedding)
            global_entity_counts.index_add_(
                0,
                local_shared_position,
                torch.ones((len(local_shared_ids), 1), dtype=current_entity_embedding.dtype, device=self.device),
            )

        global_entity_embeddings = global_entity_embeddings / global_entity_counts.clamp_min(1.0)

        for cid in range(self.args.num_clients):
            local_shared_ids = self.client_shared_entity_ids[cid]
            if not local_shared_ids:
                continue
            local_entity_index = torch.tensor(local_shared_ids, dtype=torch.long, device=self.device)
            local_shared_position = torch.tensor(
                [shared_position[entity_id] for entity_id in local_shared_ids],
                dtype=torch.long,
                device=self.device,
            )
            self.clients[cid].model.entity_embedding[local_entity_index, :] = global_entity_embeddings.index_select(0, local_shared_position)

    def update_relation_embeddings(self):
        global_relation_embeddings = None
        global_relation_ids = None
        for cid in range(self.args.num_clients):
            if global_relation_embeddings is None:
                current_relation_embedding = self.clients[cid].model.relation_embedding
                counter = torch.zeros((current_relation_embedding.size(0)), dtype=torch.long, device=self.device)
                counter[self.clients[cid].model.relation_ids] = 1
                counter = counter.view(-1, 1).expand(-1, current_relation_embedding.size(-1)).clone()
                global_relation_ids = counter
                global_relation_embeddings = counter * current_relation_embedding
            else:
                current_relation_embedding = self.clients[cid].model.relation_embedding
                counter = torch.zeros((current_relation_embedding.size(0)), dtype=torch.long, device=self.device)
                counter[self.clients[cid].model.relation_ids] = 1
                counter = counter.view(-1, 1).expand(-1, current_relation_embedding.size(-1)).clone()

                global_relation_ids += counter
                global_relation_embeddings += counter * current_relation_embedding

        global_relation_embeddings /= global_relation_ids
        for cid in range(self.args.num_clients):
            self.clients[cid].model.relation_embedding = global_relation_embeddings

    
    def get_old_local_parameters(self):
        weight_local_list = []
        for cid in range(self.args.num_clients):
            weight_local = {}
            named_parameters = deepcopy(self.clients[cid].model.state_dict())
            for name in self.target_parameter_names:
                weight_local[name] = named_parameters[name]
            weight_local_list.append(weight_local)
        return weight_local_list

   

    def update_all_parameters(self):
        weight_global = {}
        for cid in range(self.args.num_clients):
            named_parameters = deepcopy(self.clients[cid].model.state_dict())
            for name in self.target_parameter_names:
                if name not in weight_global:
                    weight_global[name] = deepcopy(named_parameters[name])
                else:
                    weight_global[name] += named_parameters[name]

        for name in self.target_parameter_names:
            weight_global[name] = torch.div(weight_global[name], self.args.num_clients)

        for cid in range(self.args.num_clients):
            self.clients[cid].model.load_state_dict(weight_global)
