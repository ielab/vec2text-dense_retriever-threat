from vec2text import analyze_utils
import logging
from argparse import ArgumentParser
from vec2text.models import CorrectorEncoderModel
from vec2text.models.config import InversionConfig
from vec2text.data_helpers import dataset_from_args, load_nq_val_datasets, load_ag_news_test, load_nq_val
# set up logging to info level
logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument('--model_dir', required=True)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--steps', default=20, type=int)
parser.add_argument('--beam_width', default=1, type=int)
parser.add_argument('--transform', action='store_true')
parser.add_argument('--noise', default=0.0, type=float)

args = parser.parse_args()
# from IPython import embed; embed(), exit(1)

experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(args.model_dir)
# type(trainer.inversion_trainer.model.embedder)

# embedder config
# trainer.inversion_trainer.model.embedder.noise = args.noise
trainer.inversion_trainer.model.eval_embedding_noise_level = args.noise
trainer.inversion_trainer.model.embedder.transform = args.transform
trainer.inversion_trainer.call_embedding_model = trainer.inversion_trainer.model.call_embedding_model

# generation config
trainer.args.per_device_eval_batch_size = args.batch_size
trainer.sequence_beam_width = args.beam_width
trainer.num_gen_recursive_steps = args.steps
experiment.data_args.max_eval_samples = 1000

# model config, we close all the parameters during training. Eval everything on the fly
experiment.model_args.use_frozen_embeddings_as_input = False
experiment.training_args.str_mix_augmentation = False
experiment.training_args.training_embedding_noise_level = 0.0
trainer.inversion_trainer.model.training_embedding_noise_level = 0.0
trainer.model.training_embedding_noise_level = 0.0


val_datasets_dict = load_nq_val_datasets()
val_datasets = experiment._prepare_val_datasets_dict(
    model=trainer.model,
    tokenizer=trainer.tokenizer,
    embedder_tokenizer=trainer.embedder_tokenizer,
    val_datasets_dict=val_datasets_dict,
)


eval_results = trainer.evaluate(
    eval_dataset=val_datasets["nq"]
)

for key in ['eval_bleu_score', 'eval_token_set_f1', 'eval_exact_match', 'eval_emb_cos_sim']:
    print(key, eval_results[key])

'''
jxm/gtr__nq__32__correct
/scratch/project/neural_ir/arvin/vec2text-reproduce/saves/corrector/gtr_base-corrector
/scratch/project/neural_ir/arvin/vec2text-reproduce/saves/corrector/gtr_base_st-corrector
/scratch/project/neural_ir/arvin/vec2text-reproduce/saves/corrector/nq/dim_768
/scratch/project/neural_ir/arvin/vec2text-reproduce/saves/corrector/nq/dim_768_norm
/scratch/project/neural_ir/arvin/vec2text-reproduce/saves/corrector/nq/dim_768_norm_mean
/scratch/project/neural_ir/arvin/vec2text-reproduce/saves/corrector/nq/dim_768_pq_m768
/scratch/project/neural_ir/arvin/vec2text-reproduce/saves/corrector/nq/dim_768_pq_m256
/scratch/project/neural_ir/arvin/vec2text-reproduce/saves/corrector/nq/dim_256

python3 eval_v2t.py \
--model_dir /scratch/project/neural_ir/arvin/vec2text-reproduce/saves/corrector/nq/dim_768 \
--batch_size 16 \
--steps 50 \
--beam_width 4

'''
