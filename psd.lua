local PSD, parent = torch.class('unsup.PSD','unsup.UnsupModule')

-- encoder     : predictor module (something dervied from nn.Module generally)
-- decoder     : decoding module (generally a linear or convolutional fista L1)
-- beta	       : prediction coefficient
-- params      : optim.FistaLS parameters
function PSD:__init(encoder, decoder, beta, params)
   
   parent.__init(self)

   -- prediction weight
   self.beta = beta

   -- encoder
   self.encoder = encoder

   -- decoder is most of the time L1 solution
   self.decoder = decoder

   -- prediction cost
   self.predcost = nn.MSECriterion()
   self.predcost.sizeAverage = false

   -- parameters
   params = params or {}
   self.params = params

   self:reset()
end

function PSD:parameters()
   local seq = nn.Sequential()
   seq:add(self.encoder)
   seq:add(self.decoder)
   return seq:parameters()
end

function PSD:initDiagHessianParameters()
   self.encoder:initDiagHessianParameters()
   self.decoder:initDiagHessianParameters()
end

function PSD:reset(stdv)
   self.decoder:reset(stdv)
   self.encoder:reset(stdv)
end

function PSD:updateOutput(input)
   -- pass through encoder
   local prediction = self.encoder:updateOutput(input)
   -- do FISTA
   local fval,h = self.decoder:updateOutput(input,prediction)
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

function PSD:updateDiagHessianInput(input, diagHessianOutput)
   local predhess = self.predcost:updateDiagHessianInput(self.encoder.output, self.decoder.code)
   predhess:mul(self.beta)
   self.encoder:updateDiagHessianInput(input,predhess)
end

function PSD:accDiagHessianParameters(input, diagHessianOutput)
   self.decoder:accDiagHessianParameters(input)
   self.encoder:accDiagHessianParameters(input,self.predcost.diagHessianInput)
end

function PSD:zeroGradParameters()
   self.encoder:zeroGradParameters()
   self.decoder:zeroGradParameters()
end

function PSD:updateParameters(learningRate)
   local eta = {}
   if type(learningRate) ~= 'number' then
      eta = learningRate
   else
      eta[1] = learningRate
      eta[2] = learningRate
   end
   self.decoder:updateParameters(eta[2])
   self.encoder:updateParameters(eta[1])
end

function PSD:normalize()
   self.decoder:normalize()
end