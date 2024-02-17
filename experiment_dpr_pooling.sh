dim=768 # set other numbers for different embedding dimensions
norm=True # False is dot product, True is cosine similarity
pooling_mode=mean # mean or cls, mean is average of all tokens, cls is the cls token
model_dir=tevatron_model/dpr_${dim}_norm-${norm}_${pooling_mode}
query_dir=embedings/query/nq/dpr_${dim}_norm-${norm}_${pooling_mode}/
corpus_dir=embedings/corpus/nq/dpr_${dim}_norm-${norm}_${pooling_mode}/
result_dir=dr_results/nq
mkdir -p ${query_dir}
mkdir -p ${corpus_dir}
mkdir -p ${result_dir}

################# Step1: train DPR embedding model ###############
python3 -m tevatron.driver.train \
  --output_dir ${model_dir} \
  --model_name_or_path bert-base-uncased \
  --save_steps 20000 \
  --dataset_name Tevatron/wikipedia-nq \
  --bf16 \
  --per_device_train_batch_size 128 \
  --positive_passage_no_shuffle \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 156 \
  --num_train_epochs 40 \
  --logging_steps 100 \
  --run_name dpr_${dim}_norm-${norm}_${pooling_mode} \
  --add_pooler True \
  --projection_in_dim 0 \
  --normalize True \
  --pooling_mode mean \
  --overwrite_output_dir
####################################################################

################# Step2: encode queries and corpus ##################
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path ${model_dir} \
  --bf16 \
  --per_device_eval_batch_size 1024 \
  --dataset_name Tevatron/wikipedia-nq/test \
  --encoded_save_path ${query_dir}/query_emb.pkl \
  --encode_is_qry

for s in $(seq -f "%02g" 0 19)
do
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path ${model_dir} \
  --bf16 \
  --per_device_eval_batch_size 1024 \
  --dataset_name Tevatron/wikipedia-nq-corpus \
  --encoded_save_path ${corpus_dir}/corpus_emb.$s.pkl \
  --encode_num_shard 20 \
  --encode_shard_index $s
done
####################################################################

################# Step3: evaluate retrieval ##########################
python -m tevatron.faiss_retriever \
--query_reps ${query_dir}/query_emb.pkl \
--passage_reps ${corpus_dir}/'corpus_emb.*.pkl' \
--depth 1000 \
--batch_size 128 \
--save_text \
--save_ranking_to ${result_dir}/run.nq.dpr_${dim}_norm-${norm}_${pooling_mode}.txt

python -m tevatron.utils.format.convert_result_to_trec \
              --input ${result_dir}/run.nq.dpr_${dim}_norm-${norm}_${pooling_mode}.txt \
              --output ${result_dir}/run.nq.dpr_${dim}_norm-${norm}_${pooling_mode}.trec

python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
              --topics dpr-nq-test \
              --index wikipedia-dpr \
              --input ${result_dir}/run.nq.dpr_${dim}_norm-${norm}_${pooling_mode}.trec \
              --output ${result_dir}/run.nq.dpr_${dim}_norm-${norm}_${pooling_mode}.json

python -m pyserini.eval.evaluate_dpr_retrieval \
                --retrieval ${result_dir}/run.nq.dpr_${dim}_norm-${norm}_${pooling_mode}}.json \
                --topk 10 20 100 1000
####################################################################

################# Step4: train vec2text ##########################
embedder=dpr_${dim}_norm-${norm}_${pooling_mode}
python3 vec2text/run.py \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 512 \
    --max_seq_length 32 \
    --model_name_or_path t5-base \
    --dataset_name nq \
    --embedder_model_name ${model_dir} \
    --num_repeat_tokens 16 \
    --embedder_no_grad True \
    --num_train_epochs 50 \
    --max_eval_samples 2000 \
    --eval_steps 2000 \
    --warmup_steps 10000 \
    --bf16=1 \
    --use_wandb=1 \
    --logging_steps 50 \
    --use_frozen_embeddings_as_input True \
    --experiment inversion \
    --lr_scheduler_type constant_with_warmup \
    --exp_group_name nq_${embedder} \
    --learning_rate 0.001 \
    --output_dir ./saves/inversion/${embedder} \
    --save_steps 2000 \
    --experiment_id inversion_${embedder}


python3 vec2text/run.py \
     --per_device_train_batch_size 512 \
     --per_device_eval_batch_size 512 \
     --max_seq_length 32 \
     --model_name_or_path t5-base \
     --dataset_name nq \
     --embedder_model_name ${model_dir} \
     --num_repeat_tokens 16 \
     --embedder_no_grad True \
     --num_train_epochs 50 \
     --max_eval_samples 2000 \
     --eval_steps 2000 \
     --warmup_steps 10000 \
     --bf16=1 \
     --use_wandb=1 \
     --logging_steps 50 \
     --use_frozen_embeddings_as_input True \
     --experiment corrector \
     --lr_scheduler_type constant_with_warmup \
     --exp_group_name ${embedder} \
     --learning_rate 0.001 \
     --output_dir ./saves/correcotr/${embedder} \
     --save_steps 2000 \
     --corrector_model_from_pretrained ./saves/inversion/${embedder} \
     --disable_tqdm=False \
     --experiment_id corrector_${embedder}
####################################################################

################# Step5: evaluate vec2text ##########################
python3 eval_v2t.py \
--model_dir ./saves/correcotr/${embedder} \
--batch_size 16 \
--steps 50 \
--beam_width 4 \

# Add the following configs for other experiments
# add noise to embeddings, default no noise: --noise 0.0
# set for linear transform to embeddings: --transform
####################################################################