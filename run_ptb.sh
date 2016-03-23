
CUDA_VISIBLE_DEVICES=1 th main_char.lua -data_dir data/ptb -savefile ptb-char-lstm -EOS '+' -rnn_size 100 -gpuid 0 -data_dir data/ptb 
