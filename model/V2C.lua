-- This module is quit like the W2C module.
--
-- However, here, we need to calculate the prob of the current word.
--
--
require 'torch'
require 'nn'

local V2C,parent = torch.class('nn.V2C', 'nn.Module')

model_utils = require 'util.model_utils'
require 'util.misc'
debugger = require('fb.debugger')

function share_params(cell, src)

    if torch.type(cell) == 'nn.gModule' then
        for i = 1, #cell.forwardnodes do
            local node = cell.forwardnodes[i]
            if node.data.module then
                node.data.module:share(src.forwardnodes[i].data.module,
                  'weight', 'bias', 'gradWeight', 'gradBias')
            end
        end
    elseif torch.isTypeOf(cell, 'nn.Module') then
        cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
    else
        error('parameters cannot be shared for this input')
    end
end


function V2C:__init(word_max_l, input_size, char_vocab_size, char_rnn_size, batch_size, gpuid)
    parent.__init(self)
    self.word_max_l = word_max_l 
    self.input_size = input_size
    self.char_vocab_size = char_vocab_size
    self.char_rnn_size = char_rnn_size
    self.batch_size = batch_size

    self.com_module = self:new_composer()

    self.rnn = LSTM_v.lstm(input_size + char_rnn_size, char_vocab_size, char_rnn_size, 1)

    self.char_vec_layer = nn.LookupTable(self.char_vocab_size, char_rnn_size)
    self.v2c_module = model_utils.clone_many_times(self.rnn, self.word_max_l)

    self.init_state = {}
    self.zeros = torch.zeros(self.batch_size,self.char_rnn_size)
    self.gpuid = gpuid

    if self.gpuid >= 0 then
        self.zeros = self.zeros:cuda()
    end

    table.insert(self.init_state, self.zeros:clone()) -- prev_c
    table.insert(self.init_state, self.zeros:clone()) -- prev_h
    self.vec_module = model_utils.clone_many_times(self.rnn, self.word_max_l)
end

function V2C:new_composer()
    local x = nn.Identity()() -- This is the encoding from previous LSTM. batch_size * hid_size
    local c = nn.Identity()() -- This is the encoding of the chars in a word. batch_size * word_len * char_size
    -- Firstly, repeat x, to make it match with c.
    local x_r = nn.Replicate(self.word_max_l,2)(x) -- batch_size * word_len * hid_size 
    local input_n = nn.JoinTable(3)({x_r, c})

    local com_module = nn.gModule({x, c}, {input_n})
    if self.com_module ~= nil then
        share_params(com_module, self.com_module)
    end
    return com_module
end

function V2C:forward(word, x_hid)
    local word_vec = self.char_vec_layer:forward(word) -- batch*word_len * emb_size
    local word_len = word:size(2)
    assert(word_len <= self.word_max_l)

    local input_n = self.com_module:forward({x_hid, word_vec})
    local predictions = torch.zeros(self.batch_size, self.word_max_l, self.char_vocab_size)
    if self.gpuid >= 0 then 
        predictions = predictions:float():cuda()
    end
    self.rnn_state  = {[0] = self.init_state}

    for t = 1, self.word_max_l do
        self.v2c_module[t]:training() -- for dropout proper functioning
        local lst = self.v2c_module[t]:forward{input_n[{{},t,{}}], unpack(self.rnn_state[t-1])}
        self.rnn_state[t] = {}
        for i = 1, #self.init_state do
            table.insert(self.rnn_state[t], lst[i])
        end
        predictions[{{},t,{}}] = lst[#lst]
    end
    return predictions -- Return the predictions for the current word.
end

function V2C:forward_eval_beam(word, x_hid, beam_c, beam_w)
    -- This function intends to evaluate the module. We do not know the ground truth.
    -- We can only kind of use a beam search to find the next char as well as the next word.
    --
    -- We need to use two beam search, this function will only cover the char level beam-search.
    -- Each time, we will generate a total of beam_w words.
    -- However, to calculate the PPL, we need to generate exactly the next word. 
    -- For now, we implement this function, first. THis is enough for generation of the captions.
    
    local word_vec = self.char_vec_layer:forward(word) -- batch*word_len * emb_size
    local word_len = word:size(2)
    assert(word_len <= self.word_max_l)

    local input_n = self.com_module:forward({x_hid, word_vec})
    --local predictions = {}
    local predictions = torch.zeros(self.batch_size, self.word_max_l, self.char_vocab_size)
    if self.gpuid >= 0 then 
        predictions = predictions:float():cuda()
    end
 
    self.rnn_state  = {[0] = self.init_state}

    local w_cand = {}

    local batch_of_beams = {}
    for i = 1, self.batch_size do
        batch_of_beams[i] = {}
        batch_of_beams[i][0] = 0.0
        batch_of_beams[i][1] = {}
    end
    -- We need to generate a total of beam_w words.
    for t = 1, self.word_max_l do
        self.v2c_module[t]:evaluate() -- for dropout proper functioning
        local lst = self.v2c_module[t]:forward{input_n[{{},t,{}}], unpack(self.rnn_state[t-1])}
        self.rnn_state[t] = {}
        for i = 1, #self.init_state do
            table.insert(self.rnn_state[t], lst[i])
        end
        --predictions[t] = lst[#lst] -- The last one is the probability.
        predictions[{{},t,{}}] = lst[#lst]
    end
    return predictions -- Return the predictions for the current word.
end


function V2C:backward(word, x_hid, grad_output) 
    -- grad_ouput should also be a table.
    local word_vec = self.char_vec_layer:forward(word) -- batch*word_len * emb_size
    local input_n = self.com_module:forward({x_hid, word_vec})
    self:forward(word, x_hid) -- inorder to obtain rnn_state.

    local word_len = word:size(2)
    assert(word_len <= self.word_max_l)

    local drnn_state = {[word_len] = clone_list(self.init_state,true)}

    local grad_com = torch.zeros(input_n:size())
    if self.gpuid >= 0 then
        grad_com = grad_com:cuda()
    end

    for t = word_len, 1, -1 do
        table.insert(drnn_state[t], grad_output[{{},t,{}}])

        local dlst = self.v2c_module[t]:backward({input_n[{{},t,{}}], unpack(self.rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k, v in pairs(dlst) do
            if k > 1 then
                drnn_state[t-1][k-1] = v
            else
                grad_com[{{},t,{}}] = v
            end
        end
    end
    
    local hid_grad, word_vec = unpack(self.com_module:backward({x_hid, word_vec}, grad_com))
    return hid_grad
end

function V2C:parameters()
    local params, grad_params = {}, {}

    local params_rnn, grad_params_rnn = self.rnn:parameters()
    tablex.insertvalues(params, params_rnn)
    tablex.insertvalues(grad_params, grad_params_rnn)

    local params_emb, grad_params_emb = self.char_vec_layer:parameters()
    tablex.insertvalues(params, params_emb)
    tablex.insertvalues(grad_params, grad_params_emb)

    return params, grad_params
end
