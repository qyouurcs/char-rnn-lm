CUDA_VISIBLE_DEVICES=1 th main.lua -data_dir data/ptb -savefile ptb-word-small -EOS '+' -rnn_size 300 -use_chars 1  -gpuid 0
