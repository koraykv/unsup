local Diag,parent = torch.class('nn.Diag','nn.Module')

function Diag:__init(nFeature)
   parent.__init(self)
   self.weight = torch.Tensor(nFeature)
   self.gradWeight = torch.Tensor(nFeature)

   self:reset()
end

function Diag:reset(stdv)
   self.weight:fill(1)
end

function Diag:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if input:dim() > 1 then
      for i=1,input:size(1) do
	 self.output[{{i}}]:mul(self.weight[i])
      end
   else
      self.output:cmul(self.weight)
   end
   return self.output
end

function Diag:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   if input:dim() > 1 then
      for i=1,input:size(1) do
	 self.gradInput[{{i}}]:mul(self.weight[i])
      end
   else
      self.gradInput:cmul(self.weight)
   end
   return self.gradInput
end

function Diag:accGradParameters(input, gradOutput, scale)
   for i=1,input:size(1) do
      self.gradWeight[i] = self.gradWeight[i] + scale*gradOutput[{{i}}]:dot(input[{{i}}])
   end
end

