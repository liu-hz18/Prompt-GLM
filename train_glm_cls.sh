python finetune_glm.py --expname glm-roberta-large-qasc1-notune --test_only --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 256 --train_bsz 64 --cls --dataset qasc --prompt "is_correct_1" --choices Yes No
python finetune_glm.py --expname glm-roberta-large-qasc2-notune --test_only --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 256 --train_bsz 64 --cls --dataset qasc --prompt "is_correct_2" --choices Yes No
python finetune_glm.py --expname glm-roberta-large-sst-notune --test_only --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 2048 --train_bsz 64 --cls --dataset glue/sst2 --prompt "happy or mad" --choices good bad

python finetune_glm.py --expname glm-roberta-large-qasc1 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 256 --train_bsz 64 --cls --dataset qasc --prompt "is_correct_1" --choices Yes No
python finetune_glm.py --expname glm-roberta-large-qasc2 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 256 --train_bsz 64 --cls --dataset qasc --prompt "is_correct_2" --choices Yes No
python finetune_glm.py --expname glm-roberta-large-sst --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 2048 --train_bsz 64 --cls --dataset glue/sst2 --prompt "happy or mad" --choices good bad
