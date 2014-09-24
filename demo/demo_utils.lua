function nnhacks()

   -- hack Linear
   local Linear = torch.getmetatable("nn.Linear")
   local oldLinearUpdateParameters = Linear.updateParameters
   function Linear:updateParameters(learningRate)
      -- scale the gradients so that we do not add up bluntly like in batch
      oldLinearUpdateParameters(self, learningRate/self.weight:size(2))
   end

   -- hack SpatialFullConvolution
   local SpatialFullConvolution = torch.getmetatable("nn.SpatialFullConvolution")
   local oldSpatialFullConvolutionUpdateParameters = SpatialFullConvolution.updateParameters
   function SpatialFullConvolution:updateParameters(learningRate)
      oldSpatialFullConvolutionUpdateParameters(self, learningRate/(self.nInputPlane))
   end

   -- hack SpatialConvolution
   local SpatialConvolution = torch.getmetatable("nn.SpatialConvolution")
   local oldSpatialConvolutionUpdateParameters = SpatialConvolution.updateParameters
   function SpatialConvolution:updateParameters(learningRate)
      oldSpatialConvolutionUpdateParameters(self, learningRate/(self.kW*self.kH*self.nInputPlane))
   end

   -- hack SpatialFullConvolutionMap
   local SpatialFullConvolutionMap = torch.getmetatable("nn.SpatialFullConvolutionMap")
   local oldSpatialFullConvolutionMapUpdateParameters = SpatialFullConvolutionMap.updateParameters
   function SpatialFullConvolutionMap:updateParameters(learningRate)
      if not self.ninput then
         self.ninput = torch.Tensor(self.nOutputPlane):zero()
         for i=1,self.connTable:size(1) do
            local to = self.connTable[i][2]
            self.ninput[to] = self.ninput[to]+1
         end
      end
      oldSpatialFullConvolutionMapUpdateParameters(self, learningRate/(self.ninput:max()))
   end

   -- hack SpatialConvolutionMap
   local SpatialConvolutionMap = torch.getmetatable("nn.SpatialConvolutionMap")
   local oldSpatialConvolutionMapUpdateParameters = SpatialConvolutionMap.updateParameters
   function SpatialConvolutionMap:updateParameters(learningRate)
      if not self.ninput then
         self.ninput = torch.Tensor(self.nOutputPlane):zero()
         for i=1,self.connTable:size(1) do
            local to = self.connTable[i][2]
            self.ninput[to] = self.ninput[to]+1
         end
      end
      oldSpatialConvolutionMapUpdateParameters(self, learningRate/(self.ninput:max()))
   end

   -- hack SpatialSubSampling
   local SpatialSubSampling = torch.getmetatable("nn.SpatialSubSampling")
   local oldSpatialSubSamplingUpdateParameters = SpatialSubSampling.updateParameters
   function SpatialSubSampling:updateParameters(learningRate)
      oldSpatialSubSamplingUpdateParameters(self, learningRate/(self.kW*self.kH*self.nInputPlane))
   end

end
