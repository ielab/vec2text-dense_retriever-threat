import math
from typing import Dict

import torch
import torch.nn as nn
import transformers
import random
from copy import deepcopy
import faiss
from vec2text.trainers.base import BaseTrainer
import os
import datasets
import numpy as np
from datasets.fingerprint import is_caching_enabled, set_caching_enabled

class InversionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ######################################################
        self.tokenizer = self.model.tokenizer
        self.embedder_tokenizer = self.model.embedder_tokenizer
        self.call_embedding_model = self.model.call_embedding_model

        # if self.args.pq_quantization_m > 0:
        #     all_reps = self.train_dataset['frozen_embeddings'].numpy()
        #     if os.path.exists(self.model.config.embedder_model_name):  # if it is local model
        #         embedder_dir = self.model.config.embedder_model_name
        #     else:  # huggingface model
        #         resolved_config_file = transformers.utils.cached_file(
        #             self.model.config.embedder_model_name,
        #             transformers.utils.CONFIG_NAME,
        #             _raise_exceptions_for_missing_entries=False,
        #             _raise_exceptions_for_connection_errors=False,
        #         )
        #         embedder_dir = resolved_config_file
        #
        #     if os.path.exists(os.path.join(embedder_dir, 'pq_index.faiss')):
        #         print(f'Loading PQ index from {os.path.join(embedder_dir, "pq_index.faiss")}')
        #         pq_index = faiss.read_index(os.path.join(embedder_dir, 'pq_index.faiss'))
        #     else:
        #         pq_index = faiss.IndexPQ(all_reps.shape[1], self.args.pq_quantization_m, 8,
        #                                       faiss.METRIC_INNER_PRODUCT)  # nbits=8 always, 256 centroids per sub-vector
        #         # self.pq_index.sa_encode(all_reps[:3])
        #         print('Training PQ index...')
        #         pq_index.train(all_reps)
        #
        #         faiss.write_index(pq_index, os.path.join(embedder_dir, 'pq_index.faiss'))
        #
        #     def quantize(example):
        #         code = pq_index.sa_encode(np.expand_dims(example["frozen_embeddings"].numpy(), axis=0))
        #         example["quantized_frozen_embeddings"] = torch.from_numpy(code).to(torch.float32)[0]
        #         return example
        #
        #     print('Quantize embeddings...')
        #     if not is_caching_enabled():
        #         datasets.enable_caching()
        #
        #     self.eval_dataset = self.eval_dataset.map(quantize)
        #     self.eval_dataset = self.eval_dataset.select_columns(
        #         ['input_ids', 'attention_mask', 'labels', 'length', 'embedder_input_ids', 'embedder_token_type_ids',
        #          'embedder_attention_mask', 'idx', 'quantized_frozen_embeddings'])
        #     self.eval_dataset = self.eval_dataset.rename_column("quantized_frozen_embeddings", "frozen_embeddings")
        #
        #     self.train_dataset = self.train_dataset.map(quantize)
        #     self.train_dataset = self.train_dataset.select_columns(
        #         ['input_ids', 'attention_mask', 'labels', 'length', 'embedder_input_ids', 'embedder_token_type_ids',
        #          'embedder_attention_mask', 'idx', 'quantized_frozen_embeddings'])
        #     self.train_dataset = self.train_dataset.rename_column("quantized_frozen_embeddings", "frozen_embeddings")


            ###########################################################################

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        return self.model.generate(inputs=inputs, generation_kwargs=generation_kwargs)

    def mix_string_batch(self, input_ids, label_length):
        texts = self.embedder_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        words = []
        for text in texts:
            words.extend(text.split(' '))
        random.shuffle(words)

        num_chunks = input_ids.shape[0]
        chunk_size = len(words) // num_chunks
        remainder = len(words) % num_chunks

        start = 0
        batch_mix_string = []
        for i in range(num_chunks):
            end = start + chunk_size + (1 if i < remainder else 0)
            batch_mix_string.append(' '.join(words[start:end]))
            start = end

        new_inputs = self.embedder_tokenizer(batch_mix_string,
                                             return_tensors='pt',
                                             padding='max_length',
                                             truncation=True,
                                             max_length=input_ids.shape[1],).to(input_ids.device)
        labels = self.tokenizer(batch_mix_string,
                                return_tensors='pt',
                                padding='max_length',
                                truncation=True,
                                max_length=label_length).to(input_ids.device)['input_ids']
        # set -100 for 0
        labels[labels == 0] = -100
        labels = labels[:, :label_length]
        return new_inputs['input_ids'], new_inputs['attention_mask'], labels

    def training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Performs a training step. we override to compute data-specific metrics.
        """
        # TODO: Log training metrics from below... (How to do with huggingface?)
        if self.args.str_mix_augmentation:
            new_inputs_ids, new_attention_mask, new_labels = self.mix_string_batch(inputs['embedder_input_ids'], inputs['labels'].shape[1])
            inputs['embedder_input_ids'] = torch.cat([inputs['embedder_input_ids'], new_inputs_ids], dim=0)
            inputs['embedder_attention_mask'] = torch.cat([inputs['embedder_attention_mask'], new_attention_mask], dim=0)
            inputs['labels'] = torch.cat([inputs['labels'], new_labels], dim=0)

        self._compute_data_metrics(inputs=inputs)
        # self.log({ f"train/{k}": v for k,v in metrics.items() })

        return super().training_step(model, inputs)

    def evaluation_loop(
        self, *args, **kwargs
    ) -> transformers.trainer_utils.EvalLoopOutput:
        """
        Run evaluation and returns metrics.

        Override to compute ppl from eval loss.
        """
        output = super().evaluation_loop(*args, **kwargs)

        metric_key_prefix = kwargs["metric_key_prefix"]
        try:
            perplexity = math.exp(output.metrics[f"{metric_key_prefix}_loss"])
        except KeyError:
            perplexity = -1
        except OverflowError:
            perplexity = float("inf")
        output.metrics[f"{metric_key_prefix}_perplexity"] = perplexity

        return output

    def _remap_state_dict(self, state_dict: Dict) -> Dict:
        """Edit keys posthumously on model load."""
        # Rename keys for backward compatibility w/ model trained before
        # we added extra dropout to the model
        if {
            "embedding_transform.2.weight",
            "embedding_transform.2.bias",
        } <= state_dict.keys():
            print(
                "Renaming keys",
                {"embedding_transform.2.weight", "embedding_transform.2.bias"},
                "for backward compatibility.",
            )
            state_dict["embedding_transform.3.weight"] = state_dict.pop(
                "embedding_transform.2.weight"
            )
            state_dict["embedding_transform.3.bias"] = state_dict.pop(
                "embedding_transform.2.bias"
            )
        return state_dict
