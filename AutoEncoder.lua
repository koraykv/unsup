local AutoEncoder = torch.class('unsup.AutoEncoder','unsup.UnsupModule')

function AutoEncoder:__init(encoder, decoder, beta, loss)
   self.encoder = encoder
   self.decoder = decoder
   self.beta = beta
   if loss then
      self.loss = nn.MSECriterion()
   else
      self.loss = nn.MSECriterion()
      self.loss.sizeAverage = false
   end
end

function AutoEncoder:parameters()
   local seq = nn.Sequential()
   seq:add(self.encoder)
   seq:add(self.decoder)
   return seq:parameters()
end

function AutoEncoder:initDiagHessianParameters()
   self.encoder:initDiagHessianParameters()
   self.decoder:initDiagHessianParameters()
end

function AutoEncoder:reset(stdv)
   self.decoder:reset(stdv)
   self.encoder:reset(stdv)
end

function AutoEncoder:updateOutput(input,target)
   self.encoder:updateOutput(input)
   self.decoder:updateOutput(self.encoder.output)
   self.output = self.beta * self.loss:updateOutput(self.decoder.output, target)
   return self.output
end

function AutoEncoder:updateGradInput(input,target)
   self.loss:updateGradInput(self.decoder.output, target)
   self.loss.gradInput:mul(self.beta)
   self.decoder:updateGradInput(self.encoder.output, self.loss.gradInput)
   self.encoder:updateGradInput(input, self.decoder.gradInput)
   self.gradInput = self.encoder.gradInput
   return self.gradInput
end

function AutoEncoder:accGradParameters(input,target)
   self.decoder:accGradParameters(self.encoder.output, self.loss.gradInput)
   self.encoder:accGradParameters(input, self.decoder.gradInput)
end

function AutoEncoder:zeroGradParameters()
   self.encoder:zeroGradParameters()
   self.decoder:zeroGradParameters()
end

function AutoEncoder:updateDiagHessianInput(input, diagHessianOutput)
   self.loss:updateDiagHessianInput(self.decoder.output, target)
   self.loss.diagHessianInput:mul(self.beta)
   self.decoder:updateDiagHessianInput(self.encoder.output, self.loss.diagHessianInput)
   self.encoder:updateDiagHessianInput(input, self.decoder.diagHessianInput)
   self.diagHessianInput = self.encoder.diagHessianInput
   return self.diagHessianInput
end

function AutoEncoder:accDiagHessianParameters(input, diagHessianOutput)
   self.decoder:accDiagHessianParameters(self.encoder.output, self.loss.diagHessianInput)
   self.encoder:accDiagHessianParameters(input, self.decoder.diagHessianInput)
end

function AutoEncoder:updateParameters(learningRate)
   local eta = {}
   if type(learningRate) ~= 'number' then
      eta = learningRate
   else
      eta[1] = learningRate
      eta[2] = learningRate
   end
   self.encoder:updateParameters(eta[1])
   self.decoder:updateParameters(eta[2])
end
