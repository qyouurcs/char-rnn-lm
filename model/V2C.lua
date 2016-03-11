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


function V2C:__init(word_max_l, input_size, char_vocab_size, char_rnn_size)
    parent.__init(self)
    self.word_max_l = word_max_l 
    self.input_size = input_size
    self.char_vocab_size = char_vocab_size
    self.char_rnn_size = char_rnn_size

    self.com_module = self:new_composer()

    self.rnn = LSTM_v.lstm(input_size + char_rnn_size, char_vocab_size, char_rnn_size, 1)

    self.char_vec_layer = nn.LookupTable(self.char_vocab_size, char_vec_size)
    self.v2c_module = {}
    self.init_state = {}
    self.zeros = torch.zeros(input_size + self.char_rnn_size)
    self.gpuid = self.gpuid

    if self.gpuid >= 0 then
        self.zeros = self.zeros:cuda()
    end

    table.insert(self.init_state, self.zeros:clone()) -- prev_c
    table.insert(self.init_state, self.zeros:clone()) -- prev_h

    for name, proto in pairs(self.rnn) do
        print('cloning ' .. name)
        self.v2c_module[name] = model_utils.clone_many_times(proto, self.word_max_l)
    end

end

function V2C:new_composer()
    local x = nn.Identity()() -- This is the encoding from previous LSTM.
    local c = nn.Identity()() -- This is the encoding of the chars in a word.
    -- Firstly, repeat x, to make it match with c.
    local x_r = nn.Replicate(self.word_max_l)(x) -- length * input_size
    local input_n = nn.JoinTable(1)({x_r, c})

    local com_module = nn.gModule({x, c}, {input_n})
    if self.com_module ~= nil then
        share_params(com_module, self.com_module)
    end
end

function V2C:forward(word, x_hid)
    local word_vec = self.char_vec_layer:forward(word) -- batch*word_len * emb_size
    local word_len = word:size(1)[1]
    assert(word_len <= self.word_max_l)

    local input_n = self.com_module:forward({x_hid, word_vec})

    local predictions = {}

    self.rnn_state  = {[0] = self.init_state}
    for t = 1, word_len do
        local lst = self.v2c_module[t]:forward{word_vec[t], unpack(self.rnn_state[t-1])}
        self.rnn_state[t] = {}
        for i = 1, #self.init_state do
            table.insert(self.rnn_state[t], lst[i])
        end
        predictions[t] = lst[#lst] -- The last one is the probability.
    end
    return predictions -- Return the predictions for the current word.
end

function V2C:backward(word, x_hid, grad_output) 
    -- grad_ouput should also be a table.
    local word_vec = self.char_vec_layer:forward(word) -- batch*word_len * emb_size

    local input_n = self.com_module:forward({x_hid, word_vec})

    local predictions = self:forward(word, x_hid)

    local word_len = word:size(1)[1]
    assert(word_len <= self.word_max_l)

    local drnn_state = {[word_len] = clone_list(self.init_state.true)}

    local grad_com = torch.zeros(input_n:size())
    if self.gpuid >= 0 then
        grad_com = grad_com:cuda()
    end

    for t = word_len, 1, -1 do
        table_insert(drnn_state[t], grad_output[t])

        local dlst = clones.rnn[t]:backward({word_vec[t], unpack(self.rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k, v in pairs(dlst) do
            if k > 0 then
                drnn_state[t-1][k-1] = v
            else
                grad_com[t] = v
            end
        end
    end
    
    local hid_grad, word_vec = self.com_module:backward({hid, word_vec}, grad_com)
    return hid_grad
end

function V2C:parameters()
    local params, grad_params = {}, {}

    local params_rnn, grad_params_rnn = model_utils.combine_all_parameters(self.rnn)
    tablex.insertvalues(params, params_rnn)
    tablex.insertvalues(grad_params, grad_params_rnn)

    local params_emb, grad_params_emb = self.char_vec_layer:getParameters()
    tablex.insertvalues(params, params_emb)
    tablex.insertvalues(grad_params, grad_params_emb)

    return params, grad_params
end
