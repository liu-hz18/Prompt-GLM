# gen
python finetune_t5.py --expname t5-wiqa1 --backbone "google/flan-t5-large" --epoch 3 --warmup_steps 2000 --train_bsz 16 --gradient_accumulation_steps 4 --eval_bsz 32 --dataset wiqa --prompt "what_might_be_the_first_step_of_the_process"
python finetune_t5.py --expname t5-wiqa2 --backbone "google/flan-t5-large" --epoch 3 --warmup_steps 2000 --train_bsz 16 --gradient_accumulation_steps 4 --eval_bsz 32 --dataset wiqa --prompt "what_might_be_the_last_step_of_the_process"
python finetune_t5.py --expname t5-wiqa3 --backbone "google/flan-t5-large" --epoch 3 --warmup_steps 2000 --train_bsz 16 --gradient_accumulation_steps 4 --eval_bsz 32 --dataset wiqa --prompt "what_is_the_missing_first_step"
# cls
python finetune_t5.py --cls --expname t5-qasc1 --backbone "google/flan-t5-large" --epoch 3 --warmup_steps 256 --train_bsz 32 --eval_bsz 64 --dataset qasc --prompt "is_correct_1" --choices Yes No
python finetune_t5.py --cls --expname t5-qasc2 --backbone "google/flan-t5-large" --epoch 3 --warmup_steps 256 --train_bsz 32 --eval_bsz 64 --dataset qasc --prompt "is_correct_2" --choices Yes No
python finetune_t5.py --cls --expname t5-sst --backbone "google/flan-t5-large" --epoch 3 --warmup_steps 2048 --train_bsz 32 --eval_bsz 64 --dataset glue/sst2 --prompt "happy or mad" --choices good bad
