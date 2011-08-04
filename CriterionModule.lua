local CriterionModule,parent = torch.class('nn.CriterionModule','nn.Module')

function CriterionModule:__init(criterion,target)
   parent.__init(self)
   self.criterion = criterion
   self.target = target
   self.output:resize(1)
end

function CriterionModule:forward(input)
   self.output[1] = self.criterion:forward(input, self.target)
   return self.output
end

function CriterionModule:backward(input, gradOutput)
   local gi = self.criterion:backward(input, self.target)
   self.gradInput:resizeAs(gi):copy(gi):mul(gradOutput[1])
   return self.gradInput
end

function CriterionModule:write(file)
   parent.write(self,file)
   file:writeObject(self.criterion)
   file:writeObject(self.target)
end

function CriterionModule:read(file)
   parent.read(self, file)
   self.criterion = file:readObject()
   self.target = file:readObject()
end


