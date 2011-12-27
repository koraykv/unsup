
local mytester = torch.Tester()
local jac = nn.Jacobian

local precision = 1e-5

local nntest = {}

function nntest.SpatialFullConvolution()
   local from = math.random(2,5)
   local to = math.random(2,7)
   local ki = math.random(2,7)
   local kj = math.random(2,7)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local ini = math.random(10,13)
   local inj = math.random(10,13)
   local module = nn.SpatialFullConvolution(from, to, ki, kj, si, sj)
   local input = torch.Tensor(from, inj, ini):zero()
   
   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')
   
   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')
   
   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.WeightedMSECriterion()
   local from  = math.random(100,200)
   local input = torch.Tensor(from):zero()
   local target = lab.randn(from)
   local weight = lab.randn(from)
   local cri = nn.WeightedMSECriterion(weight)
   local module = nn.CriterionModule(cri,target)

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')
   
   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

mytester:add(nntest)

function unsup.module_test()
   mytester:run()
end
