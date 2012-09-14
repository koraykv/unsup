local SparseAutoEncoder = torch.class('unsup.SparseAutoEncoder','unsup.UnsupModule')

function SparseAutoEncoder:__init(encoder, decoder, beta, lambda, loss)
   self.encoder = encoder
   self.decoder = decoder
   self.sparseCost = nn.L1Cost()
   self.beta = beta
   self.lambda = lambda
   if loss then
      self.loss = nn.MSECriterion()
   else
      self.loss = nn.MSECriterion()
      self.loss.sizeAverage = false
   end
end

function SparseAutoEncoder:parameters()
   local seq = nn.Sequential()
   seq:add(self.encoder)
   seq:add(self.decoder)
   return seq:parameters()
end

function SparseAutoEncoder:initDiagHessianParameters()
   self.encoder:initDiagHessianParameters()
   self.decoder:initDiagHessianParameters()
end

function SparseAutoEncoder:reset(stdv)
   self.decoder:reset(stdv)
   self.encoder:reset(stdv)
end

function SparseAutoEncoder:updateOutput(input,target)
   self.encoder:updateOutput(input)
   self.decoder:updateOutput(self.encoder.output)
   self.output = self.beta * self.loss:updateOutput(self.decoder.output, target)
   self.output = self.output + self.lambda * self.sparseCost(self.encoder.output)
   return self.output
end

function SparseAutoEncoder:updateGradInput(input,target)
   self.loss:updateGradInput(self.decoder.output, target)
   self.loss.gradInput:mul(self.beta)

   self.sparseCost:updateGradInput(self.encoder.output)
   self.sparseCost.gradInput:mul(self.lambda)

   self.decoder:updateGradInput(self.encoder.output, self.loss.gradInput)
   -- accumulate the sparsity
   self.decoder.gradInput:add(self.sparseCost.gradInput)

   self.encoder:updateGradInput(input, self.decoder.gradInput)

   self.gradInput = self.encoder.gradInput
   return self.gradInput
end

function SparseAutoEncoder:accGradParameters(input,target)
   self.decoder:accGradParameters(self.encoder.output, self.loss.gradInput)
   self.encoder:accGradParameters(input, self.decoder.gradInput)
end

function SparseAutoEncoder:zeroGradParameters()
   self.encoder:zeroGradParameters()
   self.decoder:zeroGradParameters()
end

function SparseAutoEncoder:updateDiagHessianInput(input, diagHessianOutput)
   self.loss:updateDiagHessianInput(self.decoder.output, target)
   self.loss.diagHessianInput:mul(self.beta)

   self.sparseCost:updateDiagHessianInput(self.encoder.output)
   self.sparseCost.diagHessianInput:mul(self.lambda)

   self.decoder:updateDiagHessianInput(self.encoder.output, self.loss.diagHessianInput)
   -- accumulate the sparsity
   self.decoder.diagHessianInput:add(self.sparseCost.diagHessianInput)

   self.encoder:updateDiagHessianInput(input, self.decoder.diagHessianInput)

   self.diagHessianInput = self.encoder.diagHessianInput
   return self.diagHessianInput
end

function SparseAutoEncoder:accDiagHessianParameters(input, diagHessianOutput)
   self.decoder:accDiagHessianParameters(self.encoder.output, self.loss.diagHessianInput)
   self.encoder:accDiagHessianParameters(input, self.decoder.diagHessianInput)
end

function SparseAutoEncoder:updateParameters(learningRate)
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
