local AutoEncoder = torch.class('unsup.AutoEncoder','unsup.UnsupModule')

function AutoEncoder:__init(encoder, decoder, beta, loss, lambda, codeloss)
   self.encoder = encoder
   self.decoder = decoder
   self.beta = beta
   if loss then
      self.loss = loss
   else
      self.loss = nn.MSECriterion()
      self.loss.sizeAverage = false
   end
   if lambda and codeloss then
      self.codecost = codeloss
      self.lambda = lambda
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
   if self.lambda then
      self.output = self.output + self.lambda * self.codecost(self.encoder.output)
   end
   return self.output
end

function AutoEncoder:updateGradInput(input,target)
   self.loss:updateGradInput(self.decoder.output, target)
   self.loss.gradInput:mul(self.beta)

   if self.lambda then
      self.codecost:updateGradInput(self.encoder.output)
      self.codecost.gradInput:mul(self.lambda)
   end

   self.decoder:updateGradInput(self.encoder.output, self.loss.gradInput)

   -- accumulate gradients from code cost
   if self.lambda then
      self.decoder.gradInput:add(self.codecost.gradInput)
   end

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

   if self.lambda then
      self.codecost:updateDiagHessianInput(self.encoder.output)
      self.codecost.diagHessianInput:mul(self.lambda)
   end

   self.decoder:updateDiagHessianInput(self.encoder.output, self.loss.diagHessianInput)

   -- accumulate gradients from code cost
   if self.lambda then
      self.decoder.diagHessianInput:add(self.codecost.diagHessianInput)
   end

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

function AutoEncoder:normalize()
   if not self.normalized then return end
   -- normalize the dictionary
   local w = self.decoder.weight
   if not w or w:dim() < 2 then return end

   if w:dim() == 5 then
      for i=1,w:size(1) do
         local keri = w:select(1,i)
         for j=1,w:size(2) do
            local kerj = keri:select(1,j)
            for k=1,w:size(3) do
               local ker = kerj:select(1,k)
               ker:div(ker:norm()+1e-12)
            end
         end
      end
   elseif w:dim() == 4 then
      for i=1,w:size(1) do
         for j=1,w:size(2) do
            local k=w:select(1,i):select(1,j)
            k:div(k:norm()+1e-12)
         end
      end
   elseif w:dim() == 3 then
      for i=1,w:size(1) do
         local k=w:select(1,i)
         k:div(k:norm()+1e-12)
      end
   elseif w:dim() == 2 then
      for i=1,w:size(2) do
         local k=w:select(2,i)
         k:div(k:norm()+1e-12)
      end
   else
      error('I do not know what kind of weight matrix this is')
   end

end
