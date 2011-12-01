
-- D     : dictionary, each column is a dictionary element
-- params: set of params to pass to FISTA and possibly temp allocation (**optional**)
--         on output some temporary stuff and the following function definitions will be added
--         params.f : smooth function
--         params.g : non-smooth function
--         params.pl: argmin( Q ) solution
--         one can pass these functions too, if not passed the default is to solve L1
--         with a linear dictionary ||Ax-b|| + \lambda ||x||_1
function unsup.FistaL1(D, params)

   -- this is for keeping parameters related to fista algorithm
   local params = params or {}
   -- this is for temporary variables and such
   local fista = {}

   -- related to FISTA
   params.L = params.L or 0.1
   params.Lstep = params.Lstep or 1.5
   params.maxiter = params.maxiter or 50
   params.maxline = params.maxline or 20
   params.errthres = params.errthres or 1e-4
   
   -- temporary stuff that might be good to keep around
   fista.reconstruction = torch.Tensor()
   fista.gradf = torch.Tensor()
   fista.code = torch.Tensor()

   -- these will be assigned in run(x)
   -- fista.input points to the last input that was run
   -- fista.lambda is the lambda value from the last run
   fista.input = nil
   fista.lambda = nil

   -- CREATE FUNCTION CLOSURES
   -- smooth function
   fista.f = function (x,mode)

		local reconstruction = fista.reconstruction
		local input = fista.input
		local gradf = fista.gradf
		-- -------------------
		-- function evaluation
		if x:dim() == 1 then
		   --print(D:size(),x:size())
		   reconstruction:resize(D:size(1))
		   reconstruction:addmv(0,1,D,x)
		elseif x:dim(2) then
		   reconstruction:resize(x:size(1),D:size(1))
		   reconstruction:addmm(0,1,x,D:t())
		end
		local fval = input:dist(reconstruction)^2
		
		-- ----------------------
		-- derivative calculation
		if mode and mode:match('dx') then
		   reconstruction:add(-1,input):mul(2)
		   gradf:resizeAs(x)
		   if input:dim() == 1 then
		      gradf:addmv(0,1,D:t(),reconstruction)
		   else
		      gradf:addmm(0,1,reconstruction, D)
		   end
		   ---------------------------------------
		   -- return function value and derivative
		   return fval, gradf
		end
		
		------------------------
		-- return function value
		return fval
	     end

   -- non-smooth function L1
   fista.g =  function (x)
		 return fista.lambda*x:norm(1)
	      end
   
   -- argmin_x Q(x,y), just shrinkage for L1
   fista.pl = function (x,L)
		 x:shrinkage(fista.lambda/L)
	      end
   
   fista.run = function(x, lam)
		  local code = fista.code
		  fista.input = x
		  fista.lambda = lam
		  if x:dim() == 1 then
		     code:resize(D:size(2)):fill(0)
		  elseif x:dim() == 2 then
		     code:resize(x:size(1),D:size(2)):fill(0)
		  else
		     error(' I do not know how to handle ' .. x:dim() .. ' dimensional input')
		  end
		  return unsup.FistaLS(fista.f, fista.g, fista.pl, fista.code, params)
	       end

   return fista
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
