local CriterionModule,parent = torch.class('nn.CriterionModule','nn.Module')

function CriterionModule:__init(criterion,target)
   parent.__init(self)
   self.criterion = criterion
   self.target = target
   self.output:resize(1)
end

function CriterionModule:updateOutput(input)
   self.output[1] = self.criterion:forward(input, self.target)
   return self.output
end

function CriterionModule:updateGradInput(input, gradOutput)
   local gi = self.criterion:updateGradInput(input, self.target)
   self.gradInput:resizeAs(gi):copy(gi):mul(gradOutput[1])
   return self.gradInput
end
