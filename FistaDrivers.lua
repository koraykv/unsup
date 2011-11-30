-- input : 1D vector, input sample OR 2D batch samples, each sample in a row
-- D     : dictionary, each column is a dictionary element
-- lambda: sparsity penalty coefficient
-- params: **optional** set of params to pass to FISTA and possibly temp allocation.
function unsup.FistaL1(input, D, lambda, params)

   if D:dim() ~= 2 then
      error ('Dictionary is supposed to 2D')
   end
   if input:dim() > 2 then
      error ('input can be 1D for a single sample or 2D for a batch')
   end
   if input:dim() == 1 and input:size(1) ~= D:size(1) then
      error ('input size should be same as row size of Dictionary')
   end
   if input:dim() == 2 and input:size(2) ~= D:size(1) then
      error ('input size[2] should be same as row size of Dictionary')
   end
   local params = params or {}

   -- related to FISTA
   params.L = params.L or 0.1
   params.Lstep = params.Lstep or 1.5
   params.maxiter = params.maxiter or 50
   params.maxline = params.maxline or 20
   params.errthres = params.errthres or 1e-4
   
   -- temporary stuff that might be good to keep around
   params.reconstruction = params.reconstruction or torch.Tensor()
   params.gradf = params.gradf or torch.Tensor()
   params.code = params.code or torch.Tensor()
   params.tcode = params.tcode or torch.Tensor()

   -- resize temporary stuff
   if input:dim() == 2 then
      params.code:resize(input:size(1),D:size(2))
   else
      params.code:resize(D:size(2))
   end
   params.gradf:resizeAs(params.code)
   params.reconstruction:resizeAs(input)

   -- CREATE FUNCTION CLOSURES
   -- smooth function
   local function f(x,mode)
      -- function evaluation
      if input:dim() == 1 then
	 --print(D:size(),x:size())
	 params.reconstruction:addmv(0,1,D,x)
      else
	 params.reconstruction:addmm(0,1,x,D:t())
      end
      local fval = input:dist(params.reconstruction)^2

      -- derivative calculation
      if mode and mode:match('dx') then
	 params.reconstruction:add(-1,input):mul(2)
	 params.gradf:resizeAs(x)
	 if input:dim() == 1 then
	    params.gradf:addmv(0,1,D:t(),params.reconstruction)
	 else
	    params.gradf:addmm(0,1,params.reconstruction, D)
	 end
	 return fval, params.gradf
      end

      return fval
   end

   -- non-smooth function L1
   local function g(x)
      local temp = params.tcode
      temp:resizeAs(x)
      lab.abs(temp,x)
      return lambda*temp:sum()
   end

   -- argmin_x Q(x,y), just shrinkage for L1
   local function pl(x,L,gfx)
      x:add(-1/L,gfx)
      x:shrinkage(lambda/L)
   end

   params.f = f
   params.g = g
   params.pl = pl

   return unsup.FistaLS(f, g, pl, params.code, params)
end

-- input : 1D vector, input sample OR 2D batch samples, each sample in a row
-- D     : dictionary, each column is a dictionary element
-- lambda: sparsity penalty coefficient
-- params: **optional** set of params to pass to FISTA and possibly temp allocation.
function unsup.ConvFistaL1(input, D, lambda, params)

   if D:dim() ~= 3 then
      error ('Dictionary is supposed to 2D')
   end
   if input:dim() > 2 then
      error ('input can be 1D for a single sample or 2D for a batch')
   end
   if input:dim() == 1 and input:size(1) ~= D:size(1) then
      error ('input size should be same as row size of Dictionary')
   end
   if input:dim() == 2 and input:size(2) ~= D:size(1) then
      error ('input size[2] should be same as row size of Dictionary')
   end
   local params = params or {}

   -- related to FISTA
   params.L = params.L or 0.1
   params.Lstep = params.Lstep or 1.5
   params.maxiter = params.maxiter or 50
   params.maxline = params.maxline or 20
   params.errthres = params.errthres or 1e-4
   
   -- temporary stuff that might be good to keep around
   params.reconstruction = params.reconstruction or torch.Tensor()
   params.gradf = params.gradf or torch.Tensor()
   params.code = params.code or torch.Tensor()
   params.tcode = params.tcode or torch.Tensor()

   -- resize temporary stuff
   if input:dim() == 2 then
      params.code:resize(input:size(1),D:size(2))
   else
      params.code:resize(D:size(2))
   end
   params.gradf:resizeAs(params.code)
   params.reconstruction:resizeAs(input)

   -- CREATE FUNCTION CLOSURES
   -- smooth function
   local function f(x,mode)
      -- function evaluation
      if input:dim() == 1 then
	 --print(D:size(),x:size())
	 params.reconstruction:addmv(0,1,D,x)
      else
	 params.reconstruction:addmm(0,1,x,D:t())
      end
      local fval = input:dist(params.reconstruction)^2

      -- derivative calculation
      if mode and mode:match('dx') then
	 params.reconstruction:add(-1,input):mul(2)
	 params.gradf:resizeAs(x)
	 if input:dim() == 1 then
	    params.gradf:addmv(0,1,D:t(),params.reconstruction)
	 else
	    params.gradf:addmm(0,1,params.reconstruction, D)
	 end
	 return fval, params.gradf
      end

      return fval
   end

   -- non-smooth function L1
   local function g(x)
      local temp = params.tcode
      temp:resizeAs(x)
      lab.abs(temp,x)
      return lambda*temp:sum()
   end

   -- argmin_x Q(x,y), just shrinkage for L1
   local function pl(x,L,gfx)
      x:add(-1/L,gfx)
      x:shrinkage(lambda/L)
   end

   params.f = f
   params.g = g
   params.pl = pl

   return unsup.FistaLS(f, g, pl, params.code, params)
end
