require 'torch'
require 'nn'

local C2W,parent = torch.class('nn.C2W', 'nn.Module')

model_utils = require 'util.model_utils'
require 'util.misc'

function C2W:__init(max_word_l, char_vocab_size, char_rnn_size, gpuid)
    parent.__init(self)
    self.max_word_l = max_word_l 
    self.char_vocab_size = char_vocab_size
    self.char_rnn_size = char_rnn_size

    self.rnn = LSTM_c.lstm(char_vec_size, char_rnn_size, 1)

    self.char_vec_layer = nn.LookupTable(self.char_vocab_size, char_vec_size)
    self.c2w_module = {}
    self.init_state = {}
    self.zeros = torch.zeros(self.char_rnn_size)
    self.gpuid = self.gpuid

    if self.gpuid >= 0 then
        self.zeros = self.zeros:cuda()
    end

    table.insert(self.init_state, self.zeros:clone()) -- prev_c
    table.insert(self.init_state, self.zeros:clone()) -- prev_h

    for name, proto in pairs(self.rnn) do
        print('cloning ' .. name)
        self.c2w_module[name] = model_utils.clone_many_times(proto, self.max_word_l)
    end

end

function C2W:forward(word)
    local word_vec = self.char_vec_layer:forward(word) -- batch*word_len * emb_size
    local word_len = word:size(1)[1]
    assert(word_len <= self.max_word_l)

    self.rnn_state  = {[0] = self.init_state}
    for t = 1, word_len do
        local lst = self.c2w_module[t]:forward{word[t], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i = 1, #self.init_state do
            table.insert(rnn_state[t], lst[i])
        end
    end
    return rnn_state[#rnn_state][2] -- return the last h as the encoding of the word.
end

function C2W:backward(word, drnn_state)
    -- This is crap here, since we need to repeatedly use this C2W. Thus, the relation has been changed.
    -- Now, need to use the forward again to restore the rnn_state for current word.
    -- I know this is kind of crap, anyway it works.

    local word_vec = self.char_vec_layer:forward(word) -- batch*word_len * emb_size
    self:forward(word)

    local grad_inputs = torch.Tensor(word_vec:size())

    local word_len = word:size(1)[1]
    assert(word_len <= self.max_word_l)

    local drnn_state = {[word_len] = {[1] = self.zeros, [2] = drnn_state} }

    for t = word_len, 1, -1 do
        local dlst = clones.rnn[t]:backward({word[t], unpack(self.rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k, v in pairs(dlst) do
            if k > 0 then
                drnn_state[t-1][k-1] = v
            else
                grad_inputs[t] = v
            end
        end
    end
    self.char_vec_layer:backward(word, grad_inputs)
end

function C2W:parameters()
    local params, grad_params = {}, {}

    local params_rnn, grad_params_rnn = model_utils.combine_all_parameters(self.rnn)
    tablex.insertvalues(params, params_rnn)
    tablex.insertvalues(grad_params, grad_params_rnn)

    local params_emb, grad_params_emb = self.char_vec_layer:getParameters()
    tablex.insertvalues(params, params_emb)
    tablex.insertvalues(grad_params, grad_params_emb)

    return params, grad_params
end
