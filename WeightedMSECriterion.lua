local WeightedMSECriterion, parent = torch.class('nn.WeightedMSECriterion','nn.MSECriterion')

function WeightedMSECriterion:__init(w)
   parent:__init(self)
   self.weight = w:clone()
   self.buffer = torch.Tensor()
end

function WeightedMSECriterion:forward(input,target)
   self.buffer:resizeAs(input):copy(target)
   self.buffer:cmul(self.weight)
   return input.nn.MSECriterion_forward(self, input, self.buffer)
end

function WeightedMSECriterion:backward(input, target)
   return input.nn.MSECriterion_backward(self, input, self.buffer)
end

function WeightedMSECriterion:write(file)
   parent.write(self, file)
   file:writeObject(self.weight)
end

function WeightedMSECriterion:read(file)
   parent.read(self, file)
   self.weight = file:readObject()
   self.buffer = torch.Tensor()
end

