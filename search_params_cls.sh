# cls lr
python finetune_glm.py --expname glm-roberta-large-qasc1-lr1e-4 --lr 1e-4 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 256 --train_bsz 64 --cls --dataset qasc --prompt "is_correct_1" --choices Yes No
python finetune_glm.py --expname glm-roberta-large-qasc1-lr1e-5 --lr 1e-5 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 256 --train_bsz 64 --cls --dataset qasc --prompt "is_correct_1" --choices Yes No
python finetune_glm.py --expname glm-roberta-large-qasc1-lr1e-6 --lr 1e-6 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 256 --train_bsz 64 --cls --dataset qasc --prompt "is_correct_1" --choices Yes No

# cls bsz
python finetune_glm.py --expname glm-roberta-large-qasc1-bsz32 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 512 --train_bsz 32 --cls --dataset qasc --prompt "is_correct_1" --choices Yes No
python finetune_glm.py --expname glm-roberta-large-qasc1-bsz64 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 256 --train_bsz 64 --cls --dataset qasc --prompt "is_correct_1" --choices Yes No
python finetune_glm.py --expname glm-roberta-large-qasc1-bsz128 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 128 --train_bsz 64 --gradient_accumulation_steps 2 --cls --dataset qasc --prompt "is_correct_1" --choices Yes No

# cls weight_decay
python finetune_glm.py --expname glm-roberta-large-qasc1-wd1e-1 --weight_decay 0.1 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 256 --train_bsz 64 --cls --dataset qasc --prompt "is_correct_1" --choices Yes No
python finetune_glm.py --expname glm-roberta-large-qasc1-wd1e-2 --weight_decay 0.01 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 256 --train_bsz 64 --cls --dataset qasc --prompt "is_correct_1" --choices Yes No
python finetune_glm.py --expname glm-roberta-large-qasc1-wd1e-3 --weight_decay 0.001 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 256 --train_bsz 64 --cls --dataset qasc --prompt "is_correct_1" --choices Yes No
