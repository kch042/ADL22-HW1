# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
python3 test_slot.py --test_file "${1}" --pred_file "${2}" --ckpt_path model_slot.pt --use_crf True