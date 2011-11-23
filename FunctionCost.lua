local FunctionCost,parent = torch.class('nn.FunctionCost','nn.Criterion')

function FunctionCost:__init(m,c)
   parent.__init(self)
   self.module = m
   self.cost = c
end

function FunctionCost:reset(stdv)
   self.module:reset(stdv)
end

function FunctionCost:updateOutput(input, target)
   local mo = self.module:forward(input)
   self.output = self.cost:forward(mo,target)
   return self.output
end

function FunctionCost:updateGradInput(input, target)
   local gi = self.cost:updateGradInput(self.module.output,target)
   self.gradInput = self.module:updateGradInput(input,gi)
   return self.gradInput
end

function FunctionCost:zeroGradParameters()
   self.module:zeroGradParameters()
end

function FunctionCost:updateParameters(learningRate)
   self.module:updateParameters(learningRate)
end

