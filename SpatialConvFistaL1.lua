require 'nn'

local SpatialConvFistaL1, parent = torch.class('unsup.SpatialConvFistaL1','nn.Module')
-- inputFeatures   : number of input features
-- outputFeatures  : size of code (feature maps)
-- kw              : width of convolutional kernel
-- kh              : height of convolutional kernel
-- iw              : width of input patches
-- ih              : height of input patches
-- lambda          : sparsity coefficient
-- params          : optim.FistaLS parameters
function SpatialConvFistaL1:__init(inputFeatures, outputFeatures, kw, kh, iw, ih, lambda, params)

   parent.__init(self)

   -- sparsity coefficient
   self.lambda = lambda
   -- dictionary is a linear layer so that I can train it
   self.D = nn.SpatialFullConvolution(outputFeatures, inputFeatures, kw, kh, 1, 1)
   -- L2 reconstruction cost with weighting
   local tt = torch.Tensor(ih,iw)
   local utt= tt:unfold(1,kh,1):unfold(2,kw,1)
   tt:zero()
   utt:add(1)
   tt:div(tt:max())
   self.Fcost = nn.WeightedMSECriterion(tt)
   self.Fcost.sizeAverage = false;
   -- L1 sparsity cost
   self.Gcost = nn.L1Cost()

   -- this is going to be set at each forward call.
   self.input = nil
   -- this is going to be passed to optim.FistaLS
   self.code = torch.Tensor(outputFeatures, utt:size(1),utt:size(2)):fill(0)

   -- Now I need a function to pass along as f
   -- input is code, do reconstruction, calculate cost
   -- and possibly derivatives too
   self.f = function(x, mode)
	       local code = x
	       local gradx = nil
	       local input = self.input

	       -- forward function evaluation
	       local reconstruction = self.D:updateOutput(code)
	       local fval = self.Fcost:updateOutput(reconstruction, input)

	       -- derivative wrt code
	       if mode and mode:match('dx') then
		  local gradr = self.Fcost:updateGradInput(reconstruction, input)
		  gradx = self.D:updateGradInput(code, gradr)
	       end
	       return fval, gradx
	    end

   -- Next, we need function g that will be the non-smooth function
   self.g = function(x)
	       local code = x
	       local gradx = nil
	       local fval = self.lambda * self.Gcost:updateOutput(code)
	       if mode and mode:match('dx') then
		  gradx = self.Gcost:updateGradInput(code)
		  gradx:mul(self.lambda)
	       end
	       return fval, gradx
	    end

   -- Finally we need argmin_x Q(x,y)
   self.pl = function(x, L)
		local code = x
		code:shrinkage(self.lambda/L)
	     end

   -- this is for keeping parameters related to fista algorithm
   self.params = params or {}
   -- related to FISTA
   self.params.L = self.params.L or 0.1
   self.params.Lstep = self.params.Lstep or 1.5
   self.params.maxiter = self.params.maxiter or 50
   self.params.maxline = self.params.maxline or 20
   self.params.errthres = self.params.errthres or 1e-4
   self.params.doFistaUpdate = true

   self:reset()
   self.gradInput = nil
end

function SpatialConvFistaL1:reset(stdv)
   self.D:reset(stdv)
   --self.D.bias:fill(0)
end

function SpatialConvFistaL1:parameters()
   return {self.D.weight},{self.D.gradWeight}
end

-- we do inference in forward
function SpatialConvFistaL1:updateOutput(input)
   self.input = input
   -- init code to all zeros
   self.code:fill(0)
   -- do fista solution
   local oldL = self.params.L
   local code, h = optim.FistaLS(self.f, self.g, self.pl, self.code, self.params)
   local fval = h[#h].F

   -- let's just half the params.L (eq. to double learning rate)
   if oldL == self.params.L then
      self.params.L = self.params.L / 2
   end

   return fval, h
end

-- no grad output, because we are unsup
-- d(||Ax-b||+lam||x||_1)/dx
function SpatialConvFistaL1:updateGradInput(input)
   -- calculate grad wrt to (x) which is code.
   if self.gradInput then
      local fval, gradf = self.fista.f(self.code,'dx')
      local gval, gradg = self.fista.g(self.code,'dx')
      self.gradInput:resizeAs(gradf):copy(gradf):add(gradg)
   end
   return self.gradInput
end

function SpatialConvFistaL1:zeroGradParameters()
   self.D:zeroGradParameters()
end

-- no grad output, because we are unsup
-- d(||Ax-b||+lam||x||_1)/dA
function SpatialConvFistaL1:accGradParameters(input)
   self.Fcost:updateGradInput(self.D.output,input)
   self.D:accGradParameters(self.code, self.Fcost.gradInput)
end

function SpatialConvFistaL1:updateParameters(learningRate)
   self.D:updateParameters(learningRate)
   -- normalize the dictionary
   local w = self.D.weight
   for i=1,w:size(1) do
      for j=1,w:size(2) do
	 local k=w:select(1,i):select(1,j)
	 k:div(k:norm()+1e-12)
      end
   end
   --self.D.bias:fill(0)
end
