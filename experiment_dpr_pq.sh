# step1, step2 and step3 are same as experiment_dpr.sh

# The default setting is using CLS token, dot product
dim=768 # set other numbers for different embedding dimensions
model_dir=tevatron_model/dpr_${dim}
query_dir=embedings/query/nq/dpr_${dim}/
corpus_dir=embedings/corpus/nq/dpr_${dim}/
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
  --run_name dpr_${dim} \
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
--save_ranking_to ${result_dir}/run.nq.dpr_${dim}.txt

python -m tevatron.utils.format.convert_result_to_trec \
              --input ${result_dir}/run.nq.dpr_${dim}.txt \
              --output ${result_dir}/run.nq.dpr_${dim}.trec

python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
              --topics dpr-nq-test \
              --index wikipedia-dpr \
              --input ${result_dir}/run.nq.dpr_${dim}.trec \
              --output ${result_dir}/run.nq.dpr_${dim}.json

python -m pyserini.eval.evaluate_dpr_retrieval \
                --retrieval ${result_dir}/run.nq.dpr_${dim}.json \
                --topk 10 20 100 1000
####################################################################

################# Step4: Product quantization and its evaluation ##########################
m=768 # num of subgroups, set other values for different dimensions
python3 embedding_transform.py \
--query_reps embedings/query/nq/dpr_${dim}/query_emb.pkl \
--passage_reps embedings/corpus/nq/dpr_${dim}/'corpus_emb.*.pkl' \
--query_save_to embedings/query/nq/dpr_${dim}_pq_m${m}/ \
--passage_save_to embedings/corpus/nq/dpr_${dim}_pq_m${m}/ \
--quantization \
--m ${m}

query_dir=embedings/query/nq/dpr_${dim}_pq_m${m}/
corpus_dir=embedings/corpus/nq/dpr_${dim}_pq_m${m}/
result_dir=dr_results/nq

python -m tevatron.faiss_retriever \
--query_reps ${query_dir}/query_emb.pkl \
--passage_reps ${corpus_dir}/'*.faiss' \
--depth 1000 \
--batch_size 128 \
--save_text \
--save_ranking_to ${result_dir}/run.nq.dpr_${dim}_pq_m${m}.txt

python -m tevatron.utils.format.convert_result_to_trec \
              --input ${result_dir}/run.nq.dpr_${dim}_pq_m${m}.txt \
              --output ${result_dir}/run.nq.dpr_${dim}_pq_m${m}.trec

python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
              --topics dpr-nq-test \
              --index wikipedia-dpr \
              --input ${result_dir}/run.nq.dpr_${dim}_pq_m${m}.trec \
              --output ${result_dir}/run.nq.dpr_${dim}_pq_m${m}.json

python -m pyserini.eval.evaluate_dpr_retrieval \
                --retrieval ${result_dir}/run.nq.dpr_${dim}_pq_m${m}.json \
                --topk 10 20 100 1000
####################################################################

################# Step5: train vec2text with PQ embeddings ##########################
cp embedings/corpus/nq/dpr_${dim}_pq_m${m}/index.faiss ${model_dir}/pq_index.faiss
# note, the above command is to copy the faiss pq index file to the dpr embedding model directory.
# Our code checks if the index file exists in the model directory, then it will do the PQ to embeddings.



embedder=dpr_768_pq
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