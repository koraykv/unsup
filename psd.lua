local PSD, parent = torch.class('unsup.PSD','nn.Module')

-- inputSize   : size of input
-- outputSize  : size of code
-- lambda      : sparsity coefficient
-- beta        : prediction coefficient
-- params      : optim.FistaLS parameters
function PSD:__init(inputSize, outputSize, lambda, beta, params)
	
	parent.__init(self)

	-- prediction weight
	self.beta = beta

	-- decoder is L1 solution
	self.decoder = unsup.LinearFistaL1(inputSize, outputSize, lambda, params)

	-- prediction cost
	self.predcost = nn.MSECriterion()

	-- encoder
	params = params or {}
	self.params = params
	self.params.encoderType = params.encoderType or 'linear'

	if params.encoderType == 'linear' then
		self.encoder = nn.Linear(inputSize,outputSize)
	elseif params.encoderType == 'tanh' then
		self.encoder = nn.Sequential()
		self.encoder:add(nn.Linear(inputSize,outputSize))
		self.encoder:add(nn.Tanh())
		self.encoder:add(nn.Diag())
	elseif params.encoderType == 'tanh_shrink' then
		self.encoder = nn.Sequential()
		self.encoder:add(nn.Linear(inputSize,outputSize))
		self.encoder:add(nn.TanhShrink())
		self.encoder:add(nn.Diag())
	else
		error('params.encoderType unknown " ' .. params.encoderType)
	end

	self:reset()
end

function PSD:parameters()
	local dw,dgw = self.decoder:parameters()
	local ew,egw = self.encoder:parameters()
	for i=1,#dw do
		table.insert(ew,dw[i])
		table.insert(egw,dgw[i])
	end
end

function PSD:reset(stdv)
	self.decoder:reset(stdv)
	self.encoder:reset(stdv)
end

function PSD:updateOutput(input)
	-- pass through encoder
	local prediction = self.encoder:updateOutput(input)
	-- do FISTA
	local fval,h = self.decoder:updateOutput(input)
	-- calculate prediction error
	local perr = self.predcost:updateOutput(prediction, self.decoder.code)
	-- return total cost
	return fval + perr*self.beta, h
end

function PSD:updateGradInput(input, gradOutput)
	-- get gradient from decoder
	--local decgrad = decoder:updateGradInput(input, gradOutput)
	-- get grad from prediction cost
	local predgrad = self.predcost:updateGradInput(self.encoder.output, self.decoder.code)
	predgrad:mul(self.beta)
	self.encoder:updateGradInput(input, predgrad)
end

function PSD:accGradParameters(input, gradOutput)
	-- update decoder
	self.decoder:accGradParameters(input)
	-- update encoder
	self.encoder:accGradParameters(input,self.predcost.gradInput)
end

function PSD:zeroGradParameters()
	self.encoder:zeroGradParameters()
	self.decoder:zeroGradParameters()
end

function PSD:updateParameters(learningRate)
	self.decoder:updateParameters(learningRate)
	self.encoder:updateParameters(learningRate*100)
end

