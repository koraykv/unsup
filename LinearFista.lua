local LinearFista = torch.class('unsup.LinearFista')

function LinearFista:__init(nin, ndic, lambda, maxiter, errthres)

   -- Smooth function modules
   local dictionary = nn.Linear(ndic, nin)
   local reconstruction = nn.MSECriterion()
   reconstruction.sizeAverage = false

   -- put dic + rec into a module
   local decoder = nn.FunctionCost(dictionary, reconstruction)

   -- Non-smooth function modules
   local sparsity = nn.L1Cost()

   -- construct Fista
   self.fista = unsup.Fista(decoder, sparsity)

   if maxiter then self.fista.maxiter = maxiter end
   if errthres then self.fista.errthres = errthres end

   self.lambda = lambda
   self.code = torch.Tensor(ndic):zero()

   self:reset()
   dictionary.bias:zero()
end

function LinearFista:reset(stdv)
   self.fista.smoothFunc:reset(stdv)
end

function LinearFista:forward(input)
   local c,h = self.fista:forward(input, self.code, self.lambda)
   -- I return the code, reconstruction and history
   return self.code,self.fista.smoothFunc.module.output,h
end

function LinearFista:updateGradInput(input)
   return self.fista:updateGradInput(input,self.code,self.lambda)
end

function LinearFista:accGradParameters(input)
   self.fista:accGradParameters(input,self.code,self.lambda)
end

function LinearFista:updateParameters(learningRate)
   self.fista.smoothFunc:updateParameters(learningRate)
   self.fista.smoothFunc.module.bias:fill(0)
end

function LinearFista:zeroGradParameters()
   self.fista.smoothFunc:zeroGradParameters()
end

function LinearFista:normalize()
   local w = self.fista.smoothFunc.module.weight
   for i=1,w:size(2) do
      w:select(2,i):div(w:select(2,i):std()+1e-12)
   end
end

-- function LinearFista:forward(input)
-- end

-- function LinearFista:backward(input)
--    return self.fista:backward(input,self.code,self.lambda)
-- end

-- function LinearFista:write(file)
--    parent.write(self, file)
--    file:writeObject(self.fista)
--    file:writeDouble(self.lambda)
--    file:writeObject(self.code)
-- end

-- function LinearFista:read(file)
--    parent.read(self, file)
--    self.fista = file:readObject()
--    self.lambda = file:readDouble()
--    self.code = file:readObject()
-- end
