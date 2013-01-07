require 'nn'
require 'optim'
require 'libunsup'

-- extra modules
torch.include('unsup', 'Diag.lua')
-- classes that implement algorithms
torch.include('unsup', 'UnsupModule.lua')
torch.include('unsup', 'AutoEncoder.lua')
torch.include('unsup', 'SparseAutoEncoder.lua')
torch.include('unsup', 'FistaL1.lua')
torch.include('unsup', 'LinearFistaL1.lua')
torch.include('unsup', 'SpatialConvFistaL1.lua')
torch.include('unsup', 'psd.lua')
torch.include('unsup', 'LinearPsd.lua')
torch.include('unsup', 'ConvPsd.lua')
torch.include('unsup', 'UnsupTrainer.lua')
torch.include('unsup', 'pca.lua')
torch.include('unsup', 'kmeans.lua')

local oldhessian = nn.hessian.enable
function nn.hessian.enable()
	oldhessian() -- enable Hessian usage
	----------------------------------------------------------------------
	-- Diag
	----------------------------------------------------------------------
	local accDiagHessianParameters = nn.hessian.accDiagHessianParameters
	local updateDiagHessianInput = nn.hessian.updateDiagHessianInput
	local updateDiagHessianInputPointWise = nn.hessian.updateDiagHessianInputPointWise
	local initDiagHessianParameters = nn.hessian.initDiagHessianParameters

	function nn.Diag.updateDiagHessianInput(self, input, diagHessianOutput)
	   updateDiagHessianInput(self, input, diagHessianOutput, {'weight'}, {'weightSq'})
	   return self.diagHessianInput
	end

	function nn.Diag.accDiagHessianParameters(self, input, diagHessianOutput)
	   accDiagHessianParameters(self,input, diagHessianOutput, {'gradWeight'}, {'diagHessianWeight'})
	end

	function nn.Diag.initDiagHessianParameters(self)
	   initDiagHessianParameters(self,{'gradWeight'},{'diagHessianWeight'})
	end
end
