local SpatialBackConvolution, parent = torch.class('nn.SpatialBackConvolution','nn.Module')

function SpatialBackConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH)
   parent:__init(self)
   self.dW = dW
   self.dH = dH

   self.weight = torch.Tensor(nInputPlane, nOutputPlane, kH, kW)
   self.gradWeight = torch.Tensor(nInputPlane, nOutputPlane, kH, kW)
   
   self:reset()
end

function SpatialBackConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      local nInputPlane = self.weight:size(2)
      local kH = self.weight:size(3)
      local kW = self.weight:size(4)
      stdv = 1/math.sqrt(kW*kH*nInputPlane)
   end
   self.weight:apply(function()
                        return random.uniform(-stdv, stdv)
                     end)
end

function SpatialBackConvolution:updateOutput(input)
   return input.nn.SpatialBackConvolution_updateOutput(self, input)
end

function SpatialBackConvolution:updateGradInput(input, gradOutput)
   return input.nn.SpatialBackConvolution_updateGradInput(self, input, gradOutput)
end
function SpatialBackConvolution:accGradParameters(input, gradOutput, scale)
   return input.nn.SpatialBackConvolution_accGradParameters(self, input, gradOutput, scale)
end

