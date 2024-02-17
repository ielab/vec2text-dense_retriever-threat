# Is Vec2Text Really a Threat to Dense Retrieval Systems?

## Installation
```bash
# install vec2text
pip install --editable .

# install tevatron
cd tevatron
pip install --editable .
```

## Example: Train and eval Vec2Text with correct GTR-base embeddings

### Step 1: Train inversion model (base model)
```bash
python3 vec2text/run.py \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 512 \
    --max_seq_length 32 \
    --model_name_or_path t5-base \
    --dataset_name nq \
    --embedder_model_name gtr_base_st \
    --num_repeat_tokens 16 \
    --embedder_no_grad True \
    --num_train_epochs 50 \
    --max_eval_samples 2000 \
    --eval_steps 2000 \
    --warmup_steps 10000 \
    --bf16=1 \
    --use_wandb=1 \
    --use_frozen_embeddings_as_input True \
    --experiment inversion \
    --lr_scheduler_type constant_with_warmup \
    --exp_group_name gtr_base_st \
    --learning_rate 0.001 \
    --output_dir ./saves/inversion/gtr_base_st \
    --save_steps 2000
```
### Step 2: Train the corrector model
```bash
python3 vec2text/run.py \
     --per_device_train_batch_size 512 \
     --per_device_eval_batch_size 512 \
     --max_seq_length 32 \
     --model_name_or_path t5-base \
     --dataset_name nq \
     --embedder_model_name gtr_base_st \
     --num_repeat_tokens 16 \
     --embedder_no_grad True \
     --num_train_epochs 50 \
     --logging_steps 50 \
     --max_eval_samples 2000 \
     --eval_steps 2000 \
     --warmup_steps 10000 \
     --bf16=1 \
     --use_wandb=1 \
     --use_frozen_embeddings_as_input True \
     --experiment corrector \
     --lr_scheduler_type constant_with_warmup \
     --exp_group_name gtr_base_st_corrector \
     --learning_rate 0.001 \
     --output_dir ./saves/corrector/gtr_base_st-corrector \
     --save_steps 2000 \
     --corrector_model_from_pretrained ./saves/inversion/gtr_base_st
```
We made our trained Vec2Text GTR-base model available at [huggingface](https://huggingface.co/ielabgroup/vec2text_gtr-base-st_corrector).

## Evaluation
### Evaluate Vec2Text
```bash
python3 eval_v2t.py \
--model_dir ./saves/corrector/gtr_base_st-corrector \
--batch_size 16 \
--steps 50 \
--beam_width 4
```

### Evaluate Retrieval
```bash
query_dir=embedings/query/nq/gtr-base-st/
corpus_dir=embedings/corpus/nq/gtr-base-st/
result_dir=dr_results/nq
mkdir -p ${query_dir}
mkdir -p ${corpus_dir}
mkdir -p ${result_dir}

# encode queries
python encode_gtr-base-st.py \
  --output_dir=temp \
  --model_name_or_path sentence-transformers/gtr-t5-base \
  --bf16 \
  --per_device_eval_batch_size 1024 \
  --dataset_name Tevatron/wikipedia-nq/test \
  --encoded_save_path ${query_dir}/query_emb.pkl \
  --encode_is_qry

# encode corpus
for s in $(seq -f "%02g" 0 19)
do
python encode_gtr-base-st.py \
  --output_dir=temp \
  --model_name_or_path sentence-transformers/gtr-t5-base \
  --bf16 \
  --per_device_eval_batch_size 1024 \
  --dataset_name Tevatron/wikipedia-nq-corpus \
  --encoded_save_path ${corpus_dir}/corpus_emb.$s.pkl \
  --encode_num_shard 20 \
  --encode_shard_index $s
done

python -m tevatron.faiss_retriever \
--query_reps ${query_dir}/query_emb.pkl \
--passage_reps ${corpus_dir}/'corpus_emb.*.pkl' \
--depth 1000 \
--batch_size 128 \
--save_text \
--save_ranking_to ${result_dir}/run.nq.gtr-base-st.txt

python -m tevatron.utils.format.convert_result_to_trec \
              --input ${result_dir}/run.nq.gtr-base-st.txt \
              --output ${result_dir}/run.nq.gtr-base-st.trec

python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
              --topics dpr-nq-test \
              --index wikipedia-dpr \
              --input ${result_dir}/run.nq.gtr-base-st.trec \
              --output ${result_dir}/run.nq.gtr-base-st.json

python -m pyserini.eval.evaluate_dpr_retrieval \
                --retrieval ${result_dir}/run.nq.gtr-base-st.json \
                --topk 10 20 100 1000
```

## Other experiments:
- DPR experiments: [experiment_dpr.sh](experiment_dpr.sh)
- DPR with different pooling methods experiments: [experiment_dpr_pooling.sh](experiment_dpr_pooling.sh)
- DPR with product quantization experiments: [experiment_dpr_pq.sh](experiment_dpr_pq.sh)