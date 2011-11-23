local WeightedMSECriterion, parent = torch.class('nn.WeightedMSECriterion','nn.MSECriterion')

function WeightedMSECriterion:__init(w)
   parent:__init(self)
   self.weight = w:clone()
   self.buffer = torch.Tensor()
end

function WeightedMSECriterion:updateOutput(input,target)
   self.buffer:resizeAs(input):copy(target)
   self.buffer:cmul(self.weight)
   return input.nn.MSECriterion_updateOutput(self, input, self.buffer)
end

function WeightedMSECriterion:updateGradInput(input, target)
   return input.nn.MSECriterion_updateGradInput(self, input, self.buffer)
end
