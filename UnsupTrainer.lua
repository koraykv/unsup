local UnsupTrainer = torch.class('unsup.UnsupTrainer')

function UnsupTrainer:__init(module,data)

   local x,dx,ddx = module:getParameters()
   self.parameters = {x,dx,ddx}
   if not self.parameters or #self.parameters == 0 then
      error(' I could not get parameters from module...')
   end
   self.module = module
   self.data = data
end

function UnsupTrainer:train(params)
   -- essential stuff
   local eta = params.eta
   local etadecay = params.etadecay or 0
   local maxiter = params.maxiter
   local statinterval = params.statinterval or math.ceil(maxiter/100)
   -- optional hessian stuff
   local dohessian = params.hessian or false
   local hessianinterval = params.hessianinterval or statinterval

   local age = 1
   local err = 0
   while age <= maxiter do
      
      -- HESSIAN
      if dohessian and (age-1) % hessianinterval == 0 then
	 print('Computing Hessian')
	 params.di = age
	 self:computeDiagHessian(params)
	 print('done')
      end

      -- DATA
      local ex = data[age]

      -- SGD UPDATE
      local sres = self:trainSample(ex,eta)
      local serr = sres[1]
      err = err + serr

      -- HOOK SAMPLE
      if self.hookSample then self.hookSample(self,age,ex,sres) end

      if age % statinterval == 0 then
	 -- HOOK EPOCH
	 if self.hookEpoch then self.hookEpoch(self,age/statinterval) end

	 print('# iter= ' .. age .. ' eta= ' .. eta .. ' current error= ' .. err)
	 
	 -- ETA DECAY
	 eta = params.eta/(1+(age/statinterval)*etadecay)
	 err = 0
      end

      age = age + 1
   end
end

function UnsupTrainer:computeDiagHessian(params)
   local hessiansamples = params.hessiansamples or 500
   local minhessian = params.minhessian or 0.02
   local di = params.di

   local parameters = self.parameters

   local data = self.data
   local module = self.module

   local x = parameters[1]
   local dx = parameters[2]
   local ddx = parameters[3]

   local knew = 1/hessiansamples
   local kold = 1

   self.ddeltax = self.ddeltax or ddx.new():resizeAs(ddx)
   local ddeltax = self.ddeltax
   ddeltax:zero()

   for i=1,hessiansamples do
      local ex = data[di+i]
      local input = ex[1]
      local target = ex[2]
      module:updateOutput(input, target)

      -- gradient
      dx:zero()
      module:updateGradInput(input, target)
      module:accGradParameters(input, target)

      -- hessian
      ddx:zero()
      module:updateDiagHessianInput(input, target)
      module:accDiagHessianParameters(input, target)

      if ddx:min() < 0 then
	 error('Negative ddx')
      end

      ddeltax:mul(kold)
      ddeltax:add(knew,ddx)
   end
   ddeltax:add(minhessian)
   ddx:copy(ddeltax)
end

function UnsupTrainer:trainSample(ex, eta)
   local module = self.module
   local parameters = self.parameters

   local input = ex[1]
   local target = ex[2]

   local x = parameters[1]
   local dx = parameters[2]
   local ddx = parameters[3]

   local res = {module:updateOutput(input, target)}
   -- clear derivatives
   dx:zero()
   module:updateGradInput(input, target)
   module:accGradParameters(input, target)

   -- do update
   if not ddx then
      -- regular sgd
      x:add(-eta,dx)
   else
      -- diag hessian
      x:addcdiv(-eta,dx,ddx)
   end
   module:normalize()
   return res
end



