local FunctionCost,parent = torch.class('nn.FunctionCost','nn.Criterion')

function FunctionCost:__init(m,c)
   parent.__init(self)
   self.module = m
   self.cost = c
end

function FunctionCost:reset(stdv)
   self.module:reset(stdv)
end

function FunctionCost:forward(input, target)
   local mo = self.module:forward(input)
   self.output = self.cost:forward(mo,target)
   return self.output
end

function FunctionCost:backward(input, target)
   local gi = self.cost:backward(self.module.output,target)
   self.gradInput = self.module:backward(input,gi)
   return self.gradInput
end

function FunctionCost:zeroGradParameters()
   self.module:zeroGradParameters()
end

function FunctionCost:updateParameters(learningRate)
   self.module:updateParameters(learningRate)
end

function FunctionCost:write(file)
   parent.write(self, file)
   file:writeObject(self.module)
   file:writeObject(self.cost)
end

function FunctionCost:read(file)
   parent.read(self, file)
   self.module = file:readObject()
   self.cost = file:readObject()
end
