run_name='wiki'
node='none'
hidden_size=400
num_attention_heads=8
num_hidden_layers=4
intermediate_size=400
layer_norm_eps=1e-12
hidden_dropout_prob=0.25
attention_probs_dropout_prob=0.0
chunk_size_feed_forward=0
cnet_hidden_size=400
cnet_num_attention_heads=8
cnet_num_hidden_layers=1
cnet_intermediate_size=400
embed_dim=50
embedding_type='TransE'
dropout=0.1
max_step=300000
max_adj=50
batch_size=1
num_positive_samples=128
dataset='../data/Wiki'
k=5
llambda=5
lr=5e-5
finetune=0
grad_clip=5
weight_decay=0
num_workers=3
report_step=200
eval_per_step=50000
warm_up_step=10000
early_stop_step=150000
ckpt_save_step=0
seed=42

for ((i=0;i<=3;++i));
do
    CUDA_VISIBLE_DEVICES=7 python train_test.py\
     --run_name ${run_name}\
     --hidden_size ${hidden_size}\
     --num_attention_heads ${num_attention_heads}\
     --num_hidden_layers ${num_hidden_layers}\
     --intermediate_size ${intermediate_size}\
     --layer_norm_eps ${layer_norm_eps}\
     --hidden_dropout_prob ${hidden_dropout_prob}\
     --attention_probs_dropout_prob ${attention_probs_dropout_prob}\
     --chunk_size_feed_forward ${chunk_size_feed_forward}\
     --cnet_hidden_size ${cnet_hidden_size}\
     --cnet_num_attention_heads ${cnet_num_attention_heads}\
     --cnet_num_hidden_layers ${cnet_num_hidden_layers}\
     --cnet_intermediate_size ${cnet_intermediate_size}\
     --embed_dim ${embed_dim}\
     --embedding_type ${embedding_type}\
     --dropout ${dropout}\
     --max_step ${max_step}\
     --max_adj ${max_adj}\
     --batch_size ${batch_size}\
     --num_positive_samples ${num_positive_samples}\
     --dataset ${dataset}\
     --k ${k}\
     --llambda ${llambda}\
     --fold ${i}\
     --lr ${lr}\
     --finetune ${finetune}\
     --grad_clip ${grad_clip}\
     --weight_decay ${weight_decay}\
     --num_workers ${num_workers}\
     --report_step ${report_step}\
     --eval_per_step ${eval_per_step}\
     --warm_up_step ${warm_up_step}\
     --early_stop_step ${early_stop_step}\
     --ckpt_save_step ${ckpt_save_step}\
     --seed ${seed} & wait
done
