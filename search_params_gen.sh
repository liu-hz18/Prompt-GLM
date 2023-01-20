# gen lr
python finetune_glm.py --expname glm-roberta-large-wiqa1-lr1e-4 --lr 1e-4 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 500 --train_bsz 16 --gradient_accumulation_steps 4 --eval_bsz 32 --dataset wiqa --prompt "what_might_be_the_first_step_of_the_process"
python finetune_glm.py --expname glm-roberta-large-wiqa1-lr1e-5 --lr 1e-5 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 500 --train_bsz 16 --gradient_accumulation_steps 4 --eval_bsz 32 --dataset wiqa --prompt "what_might_be_the_first_step_of_the_process"
python finetune_glm.py --expname glm-roberta-large-wiqa1-lr1e-6 --lr 1e-6 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 500 --train_bsz 16 --gradient_accumulation_steps 4 --eval_bsz 32 --dataset wiqa --prompt "what_might_be_the_first_step_of_the_process"

# gen bsz
python finetune_glm.py --expname glm-roberta-large-wiqa1-bsz16 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 1000 --train_bsz 16 --gradient_accumulation_steps 1 --eval_bsz 32 --dataset wiqa --prompt "what_might_be_the_first_step_of_the_process"
python finetune_glm.py --expname glm-roberta-large-wiqa1-bsz64 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 500 --train_bsz 16 --gradient_accumulation_steps 4 --eval_bsz 32 --dataset wiqa --prompt "what_might_be_the_first_step_of_the_process"
python finetune_glm.py --expname glm-roberta-large-wiqa1-bsz128 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 250 --train_bsz 16 --gradient_accumulation_steps 8 --eval_bsz 32 --dataset wiqa --prompt "what_might_be_the_first_step_of_the_process"

# gen topk
python finetune_glm.py --expname glm-roberta-large-wiqa1-topk1 --topk 1 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 500 --train_bsz 16 --gradient_accumulation_steps 4 --eval_bsz 32 --dataset wiqa --prompt "what_might_be_the_first_step_of_the_process"
python finetune_glm.py --expname glm-roberta-large-wiqa1-topk10 --topk 10 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 500 --train_bsz 16 --gradient_accumulation_steps 4 --eval_bsz 32 --dataset wiqa --prompt "what_might_be_the_first_step_of_the_process"
python finetune_glm.py --expname glm-roberta-large-wiqa1-topk100 --topk 100 --backbone "BAAI/glm-roberta-large" --epoch 3 --warmup_steps 500 --train_bsz 16 --gradient_accumulation_steps 4 --eval_bsz 32 --dataset wiqa --prompt "what_might_be_the_first_step_of_the_process"
