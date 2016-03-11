--[[
Trains a word-level or character-level (for inputs) lstm language model
Predictions are still made at the word-level.

Much of the code is borrowed from the following implementations
https://github.com/karpathy/char-rnn
https://github.com/wojzaremba/lstm
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'lfs'
require 'util.misc'

debugger = require('fb.debugger')

BatchLoader = require 'util.BatchLoaderUnk'
model_utils = require 'util.model_utils'

local stringx = require('pl.stringx')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a word+character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/ptb','data directory. Should contain train.txt/valid.txt/test.txt with input data')
-- model params
cmd:option('-rnn_size', 650, 'size of LSTM internal state')
cmd:option('-char_rnn_size', 100, 'size of char_LSTM internal state')
cmd:option('-char_emb_size', 50, 'size of char emb size')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
--cmd:option('-dropout',0.5,'dropout. 0 = no dropout')
-- optimization
cmd:option('-learning_rate',1,'starting learning rate')
cmd:option('-learning_rate_decay',0.5,'learning rate decay')
cmd:option('-decay_when',1,'decay if validation perplexity does not improve by more than this much')
cmd:option('-param_init', 0.05, 'initialize parameters at')
cmd:option('-seq_length',35,'number of timesteps to unroll for')
cmd:option('-batch_size',100,'number of sequences to train on in parallel')
cmd:option('-max_epochs',25,'number of full passes through the training data')
cmd:option('-max_grad_norm',5,'normalize gradients at')
cmd:option('-max_word_l',65,'maximum word length')
-- bookkeeping
cmd:option('-seed',3435,'torch manual random number generator seed')
cmd:option('-print_every',100,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every', 5, 'save every n epochs')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','char','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-time', 0, 'print batch times')
-- GPU/CPU
cmd:option('-gpuid', -1,'which gpu to use. -1 = use CPU')
cmd:option('-cudnn', 0,'use cudnn (1=yes). this should greatly speed up convolutions')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- global constants for certain tokens
opt.tokens = {}
opt.tokens.UNK = '|' -- unk word token
opt.tokens.START = '{' -- start-of-word token
opt.tokens.END = '}' -- end-of-word token
opt.tokens.ZEROPAD = ' ' -- zero-pad token 

-- load necessary packages depending on config options
if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1)
end

if opt.cudnn == 1 then
   assert(opt.gpuid >= 0, 'GPU must be used if using cudnn')
   print('using cudnn...')
   require 'cudnn'
end
-- create the data loader class
loader = BatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, opt.max_word_l)
print('Word vocab size: ' .. #loader.idx2word .. ', Char vocab size: ' .. #loader.idx2char
        .. ', Max word length (incl. padding): ', loader.max_word_l)

-- load model objects. we do this here because of cudnn and hsm options
C2W = require 'model.C2W'
V2C = require 'model.V2C'
LSTM = require 'model.LSTM_w'

char_vocab_size = #loader.idx2char
word_vocab_size = #loader.idx2word

-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- Now, firstly, define the normal LSTM model on top of word features.
protos = {}

print('creating an LSTM with ' .. opt.num_layers .. ' layers')

protos.rnn = LSTM_w.lstm(opt.char_rnn_size, opt.rnn_size, opt.num_layers, dropout)

protos.criterion = nn.ClassNLLCriterion()

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    --local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    local h_init = torch.zeros(opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end

params, grad_params = model_utils.combine_all_parameters(protos.rnn)

protos.c2w = C2W(opt.max_word_l, char_vocab_size, opt.char_rnn_size, opt.gpuid)
protos.v2c = V2C(opt.max_word_l, opt.rnn_size, char_vocab_size, opt.char_rnn_size)

if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

-- put the above things into one flattened parameters tensor

params_c2w, grad_params_c2w = protos.c2w:getParameters()
params_v2c, grad_params_v2c = protos.v2c:getParameters()

print('number of parameters in the rnn model: ' .. params:nElement())
print('number of parameters in the c2w model: ' .. params_c2w:nElement())
print('number of parameters in the v2c model: ' .. params_v2c:nElement())

-- initialization
params:uniform(-opt.param_init, opt.param_init) -- small numbers uniform

-- get layers which will be referenced layer (during SGD or introspection)
function get_layer(layer)
    local tn = torch.typename(layer)
    if layer.name ~= nil then
        if layer.name == 'word_vecs' then
            word_vecs = layer
        elseif layer.name == 'char_vecs' then
            char_vecs = layer
        elseif layer.name == 'cnn' then
            cnn = layer
        end
    end
end 
protos.rnn:apply(get_layer)

clones_rnn = model_utils.clone_many_times(proto.rnn, opt.seq_length)

-- for easy switch between using words/chars (or both)
function get_input(x, x_char, t, prev_states)
    local u = {}
    if opt.use_chars == 1 then table.insert(u, x_char[{{},t}]) end
    if opt.use_words == 1 then table.insert(u, x[{{},t}]) end
    for i = 1, #prev_states do table.insert(u, prev_states[i]) end
    return u
end


-- evaluate the loss over an entire split
function eval_split(split_idx, max_batches)
    print('evaluating loss over split index ' .. split_idx)
    local n = loader.split_sizes[split_idx]
    if opt.hsm > 0 then
        protos.criterion:change_bias()
    end

    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_idx) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state}    
    if split_idx<=2 then -- batch eval        
    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y, x_char = loader:next_batch(split_idx)
        if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
        x_char = x_char:float():cuda()
        end
        -- forward pass
        for t=1,opt.seq_length do
            clones.rnn[t]:evaluate() -- for dropout proper functioning
            local lst = clones.rnn[t]:forward(get_input(x, x_char, t, rnn_state[t-1]))
            rnn_state[t] = {}
            for i=1,#init_state do 
                table.insert(rnn_state[t], lst[i])
            end
            prediction = lst[#lst]
                loss = loss + clones.criterion[t]:forward(prediction, y[{{}, t}])
        end
        -- carry over lstm state
        rnn_state[0] = rnn_state[#rnn_state]
    end
    loss = loss / opt.seq_length / n
    else -- full eval on test set
        local token_perp = torch.zeros(#loader.idx2word, 2) 
        local x, y, x_char = loader:next_batch(split_idx)
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
        x_char = x_char:float():cuda()
    end
    protos.rnn:evaluate() -- just need one clone
    for t = 1, x:size(2) do
        local lst = protos.rnn:forward(get_input(x, x_char, t, rnn_state[0]))
        rnn_state[0] = {}
        for i=1,#init_state do table.insert(rnn_state[0], lst[i]) end
        prediction = lst[#lst] 
            local tok_perp
            tok_perp = protos.criterion:forward(prediction, y[{{},t}])
            loss = loss + tok_perp
            token_perp[y[1][t]][1] = token_perp[y[1][t]][1] + 1 --count
            token_perp[y[1][t]][2] = token_perp[y[1][t]][2] + tok_perp
    end
    loss = loss / x:size(2)
    end    
    local perp = torch.exp(loss)    
    return perp, token_perp
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()
    if opt.hsm > 0 then
        hsm_grad_params:zero()
    end
    ------------------ get minibatch -------------------
    local x, y, x_char = loader:next_batch(1) --from train
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
        x_char = x_char:float():cuda()
    end
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)        
        local lst = clones.rnn[t]:forward(get_input(x, x_char, t, rnn_state[t-1]))
        rnn_state[t] = {}
        for i=1,#init_state do 
            table.insert(rnn_state[t], lst[i]) 
        end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
    end
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        table.insert(drnn_state[t], doutput_t)
        table.insert(rnn_state[t-1], drnn_state[t])
        local dlst = clones.rnn[t]:backward(get_input(x, x_char, t, rnn_state[t-1]), drnn_state[t])
        drnn_state[t-1] = {}
        local tmp = opt.use_words + opt.use_chars -- not the safest way but quick
        for k,v in pairs(dlst) do
            if k > tmp then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-tmp] = v
            end
        end    
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    
    -- renormalize gradients
    local grad_norm, shrink_factor
    if opt.hsm==0 then
        grad_norm = grad_params:norm()
    else
        grad_norm = torch.sqrt(grad_params:norm()^2 + hsm_grad_params:norm()^2)
    end
    if grad_norm > opt.max_grad_norm then
        shrink_factor = opt.max_grad_norm / grad_norm
        grad_params:mul(shrink_factor)
        if opt.hsm > 0 then
            hsm_grad_params:mul(shrink_factor)
        end
    end    
    params:add(grad_params:mul(-lr)) -- update params
    if opt.hsm > 0 then
        hsm_params:add(hsm_grad_params:mul(-lr))
    end
    return torch.exp(loss)
end


-- start optimization here
train_losses = {}
val_losses = {}
lr = opt.learning_rate -- starting learning rate which will be decayed
local iterations = opt.max_epochs * loader.split_sizes[1]
if char_vecs ~= nil then char_vecs.weight[1]:zero() end -- zero-padding vector is always zero
for i = 1, iterations do
    local epoch = i / loader.split_sizes[1]

    local timer = torch.Timer()
    local time = timer:time().real
    
    train_loss = feval(params) -- fwd/backprop and update params
    if char_vecs ~= nil then -- zero-padding vector is always zero
        char_vecs.weight[1]:zero() 
        char_vecs.gradWeight[1]:zero()
    end 
    train_losses[i] = train_loss

    -- every now and then or on last iteration
    if i % loader.split_sizes[1] == 0 then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[#val_losses+1] = val_loss
        local savefile = string.format('%s/lm_%s_epoch%.2f_%.2f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = {loader.idx2word, loader.word2idx, loader.idx2char, loader.char2idx}
        checkpoint.lr = lr
        print('saving checkpoint to ' .. savefile)
        if epoch == opt.max_epochs or epoch % opt.save_every == 0 then
            torch.save(savefile, checkpoint)
        end
    end

    -- decay learning rate after epoch
    if i % loader.split_sizes[1] == 0 and #val_losses > 2 then
        if val_losses[#val_losses-1] - val_losses[#val_losses] < opt.decay_when then
            lr = lr * opt.learning_rate_decay
        end
    end    

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.2f), train_loss = %6.4f", i, iterations, epoch, train_loss))
    end   
    if i % 10 == 0 then collectgarbage() end
    if opt.time ~= 0 then
       print("Batch Time:", timer:time().real - time)
    end
end

--evaluate on full test set. this just uses the model from the last epoch
--rather than best-performing model. it is also incredibly inefficient
--because of batch size issues. for faster evaluation, use evaluate.lua, i.e.
--th evaluate.lua -model m
--where m is the path to the best-performing model

test_perp, token_perp = eval_split(3)
print('Perplexity on test set: ' .. test_perp)
torch.save('token_perp-ss.t7', {token_perp, loader.idx2word})

