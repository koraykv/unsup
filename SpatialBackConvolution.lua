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

function SpatialBackConvolution:forward(input)
   return input.nn.SpatialBackConvolution_forward(self, input)
end

function SpatialBackConvolution:backward(input, gradOutput)
   local gi = input.nn.SpatialBackConvolution_backward(self, input, gradOutput)
--    print('input', input:size())
--    print('ginput',gi:size())
--    print('output',self.output:size())
--    print('goutput',gradOutput:size())
--    print('weight',self.weight:size())
--    print('gweight',self.gradWeight:size())
   return gi
end

function SpatialBackConvolution:zeroGradParameters()
   self.gradWeight:zero()
end

function SpatialBackConvolution:updateParameters(learningRate)
   self.weight:add(-learningRate, self.gradWeight)
end

function SpatialBackConvolution:write(file)
   parent.write(self, file)
   file:writeInt(self.dW)
   file:writeInt(self.dH)
   file:writeObject(self.weight)
   file:writeObject(self.gradWeight)
end

function SpatialBackConvolution:read(file)
   parent.read(self, file)
   self.dW = file:readInt()
   self.dH = file:readInt()
   self.weight = file:readObject()
   self.gradWeight = file:readObject()
end
