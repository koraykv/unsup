local SpatialBackConvolution, parent = torch.class('nn.SpatialBackConvolution','nn.Module')

function SpatialBackConvolution:__init(nInputPlane, nOutputPlane, kW, kH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
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
      local nInputPlane = self.nInputPlane
      local kH = self.kH
      local kW = self.kW
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
   if self.gradInput then
      return input.nn.SpatialBackConvolution_updateGradInput(self, input, gradOutput)
   end
end
function SpatialBackConvolution:accGradParameters(input, gradOutput, scale)
   return input.nn.SpatialBackConvolution_accGradParameters(self, input, gradOutput, scale)
end

