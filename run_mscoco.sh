
CUDA_VISIBLE_DEVICES=1 th main.lua -data_dir data/ptb -savefile ptb-char-large -EOS '+' -rnn_size 650 -use_chars 1 -use_words 0 -char_vec_size 15 -highway_layers 2 -kernels '{1,2,3,4,5,6,7}' -feature_maps '{50,100,150,200,200,200,200}' -gpuid 0 -data_dir data/mscoco  -hsm -1
