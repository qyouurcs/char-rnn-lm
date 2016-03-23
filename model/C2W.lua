-- We need to accept a batch module of many different words and encode them together.

require 'torch'
require 'nn'

local C2W,parent = torch.class('nn.C2W', 'nn.Module')

model_utils = require 'util.model_utils'
debugger = require ('fb.debugger')
require 'util.misc'

function C2W:__init(max_word_l, batch_size, char_vocab_size, char_rnn_size, gpuid)
    parent.__init(self)
    self.max_word_l = max_word_l 
    self.char_vocab_size = char_vocab_size
    self.char_rnn_size = char_rnn_size

    self.rnn = LSTM_c.lstm(char_rnn_size, char_rnn_size, 1)

    self.char_vec_layer = nn.LookupTable(self.char_vocab_size, char_rnn_size)
    self.c2w_module = {}
    self.init_state = {}
    self.zeros = torch.zeros(batch_size, self.char_rnn_size)
    self.gpuid = gpuid

    if self.gpuid >= 0 then
        self.zeros = self.zeros:cuda()
    end

    table.insert(self.init_state, self.zeros:clone()) -- prev_c
    table.insert(self.init_state, self.zeros:clone()) -- prev_h

    self.c2w_module = model_utils.clone_many_times(self.rnn, self.max_word_l)
end

function C2W:forward(word, input_word_len, is_train)
    local word_vec = self.char_vec_layer:forward(word) -- batch*word_len * emb_size
    local word_len = word_vec:size(2)
    assert(word_len <= self.max_word_l)

    self.rnn_state  = {[0] = self.init_state}
    for t = 1, word_len do
        if is_train then
            self.c2w_module[t]:training()
        else
            self.c2w_module[t]:evaluate()
        end
        local lst = self.c2w_module[t]:forward{word_vec[{{},t}], unpack(self.rnn_state[t-1])}
        --debugger.enter()
        self.rnn_state[t] = {}
        for i = 1, #self.init_state do
            table.insert(self.rnn_state[t], lst[i])
        end
    end
    local rnn_state = self.rnn_state[#self.rnn_state][2]:clone()
    for t = 1, word:size(1) do
        rnn_state[{t,{}}] = self.rnn_state[input_word_len[t][1]][2][{t,{}}]
    end
    return rnn_state
end

function C2W:backward(word, input_word_len, input_drnn_state)
    -- This is crap here, since we need to repeatedly use this C2W. Thus, the relation has been changed.
    -- Now, need to use the forward again to restore the rnn_state for current word.
    -- I know this is kind of crap, anyway it works.

    local rnn_state = self:forward(word, input_word_len, is_train)

    local word_vec = self.char_vec_layer:forward(word) -- batch*word_len * emb_size
    local grad_inputs = torch.Tensor(word_vec:size())

    if self.gpuid >= 0 then
        grad_inputs = grad_inputs:cuda()
    end


    local max_word_len = input_word_len:max()
    local drnn_state = {[max_word_len] = {[1] = self.zeros, [2] = self.zeros} }
    -- We only need to bp from the last char.
    for t = max_word_len, 1, -1 do
        for len = 1, input_word_len:size(1) do
            if t == input_word_len[len][1] then
                drnn_state[t][2][len] = input_drnn_state[len]
            end
        end

        local dlst = self.c2w_module[t]:backward({word_vec[{{},t}], unpack(self.rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}

        for k, v in pairs(dlst) do
            if k > 1 then
                drnn_state[t-1][k-1] = v
            else
                grad_inputs[{{},t,{}}] = v
            end
        end
    end
    self.char_vec_layer:backward(word, grad_inputs)
end

function C2W:parameters()
    local params, grad_params = {}, {}

    --local params_rnn, grad_params_rnn = model_utils.combine_all_parameters(self.rnn)
    local params_rnn, grad_params_rnn = self.rnn:parameters()
    tablex.insertvalues(params, params_rnn)
    tablex.insertvalues(grad_params, grad_params_rnn)

    local params_emb, grad_params_emb = self.char_vec_layer:parameters()
    tablex.insertvalues(params, params_emb)
    tablex.insertvalues(grad_params, grad_params_emb)

    return params, grad_params
end
