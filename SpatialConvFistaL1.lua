local SpatialConvFistaL1, parent = torch.class('unsup.SpatialConvFistaL1','unsup.FistaL1')
-- conntable       : Connection table (ref: nn.SpatialConvolutionMap)
-- kw              : width of convolutional kernel
-- kh              : height of convolutional kernel
-- iw              : width of input patches
-- ih              : height of input patches
-- lambda          : sparsity coefficient
-- params          : optim.FistaLS parameters
function SpatialConvFistaL1:__init(conntable, kw, kh, iw, ih, lambda, params)

   -- parent.__init(self)

   -----------------------------------------
   -- dictionary is a linear layer so that I can train it
   -----------------------------------------
   local D = nn.SpatialFullConvolutionMap(conntable, kw, kh, 1, 1)
   local outputFeatures = conntable:select(2,1):max()
   local inputFeatures = conntable:select(2,2):max()

   -----------------------------------------
   -- L2 reconstruction cost with weighting
   -----------------------------------------
   local tt = torch.Tensor(inputFeatures,ih,iw)
   local utt= tt:unfold(2,kh,1):unfold(3,kw,1)
   tt:zero()
   utt:add(1)
   tt:div(tt:max())
   local Fcost = nn.WeightedMSECriterion(tt)
   Fcost.sizeAverage = false;

   parent.__init(self,D,Fcost,lambda,params)

   -- this is going to be passed to optim.FistaLS
   self.code:resize(outputFeatures, utt:size(2),utt:size(3))
   self.code:fill(0)
end

