import torch
import torch.nn as nn
from torch import Tensor
import logging
from .encoder import EncoderPooler, EncoderModel

logger = logging.getLogger(__name__)


class DensePooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, output_dim: int = 768, tied=True, normalize=False, pooling_mode='cls'):
        super(DensePooler, self).__init__()
        self.normalize = normalize
        self.pooling_mode = pooling_mode
        self.input_dim = input_dim
        self.output_dim = output_dim
        if input_dim != 0:
            self.linear_q = nn.Linear(input_dim, output_dim)
        else:
            self.linear_q = None
        if tied:
            self.linear_p = self.linear_q
        else:
            if input_dim != 0:
                self.linear_p = nn.Linear(input_dim, output_dim)
            else:
                self.linear_p = None
        self._config = {'input_dim': input_dim,
                        'output_dim': output_dim,
                        'tied': tied,
                        'normalize': normalize,
                        'pooling_mode': pooling_mode}
        print("Pooler config:", self._config)

    def forward(self,
                q: Tensor = None,
                q_attention: Tensor = None,
                p: Tensor = None,
                p_attention: Tensor = None,
                **kwargs):
        if q is not None:
            rep = q
            attention = q_attention
            linear = self.linear_q
        elif p is not None:
            rep = p
            attention = p_attention
            linear = self.linear_p
        else:
            raise ValueError

        if self.pooling_mode == 'cls':
            rep = rep[:, 0]
        elif self.pooling_mode == 'mean':
            rep = mean_pool(rep, attention)
        else:
            raise NotImplementedError(f'Pooling mode {self.pooling_mode} not implemented.')

        if linear is not None:
            rep = linear(rep)

        if self.normalize:
            rep = nn.functional.normalize(rep, dim=-1)
        return rep


def mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


class DenseModel(EncoderModel):
    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        if self.pooler is not None:
            p_reps = self.pooler(p=p_hidden, p_attention=psg['attention_mask'])
        else:
            p_reps = p_hidden[:, 0]

        if self.pq_index is not None and not self.training:
            with torch.no_grad():
                code = self.pq_index.sa_encode(p_reps.cpu().numpy())
                p_reps = torch.from_numpy(code).to(p_reps.dtype).to(p_reps.device)

        if self.noise != 0.0:
            p_reps += self.noise * torch.randn(p_reps.shape, dtype=torch.float32).to(p_reps.device)

        if self.transform:
            p_reps = p_reps * -2.6

        return p_reps

    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out.last_hidden_state
        if self.pooler is not None:
            q_reps = self.pooler(q=q_hidden, q_attention=qry['attention_mask'])
        else:
            q_reps = q_hidden[:, 0]

        if self.pq_index is not None and not self.training:
            with torch.no_grad():
                code = self.pq_index.sa_encode(q_reps.cpu().numpy())
                q_reps = torch.from_numpy(code).to(q_reps.dtype).to(q_reps.device)

        if self.noise != 0.0:
            q_reps += self.noise * torch.randn(q_reps.shape, dtype=torch.float32).to(q_reps.device)

        if self.transform:
            q_reps = q_reps * -2.6

        return q_reps

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1)) / self.temperature

    @staticmethod
    def load_pooler(model_weights_file, **config):
        pooler = DensePooler(**config)
        pooler.load(model_weights_file)
        return pooler

    @staticmethod
    def build_pooler(model_args):
        pooler = DensePooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder,
            normalize=model_args.normalize,
            pooling_mode=model_args.pooling_mode
        )
        pooler.load(model_args.model_name_or_path)
        return pooler
