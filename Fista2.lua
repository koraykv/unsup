require 'torch'

function unsup.FistaL1(input, D, lambda, params)

   if D:dim() ~= 2 then
      error ('Dictionary is supposed to 2D')
   end
   if x:dim() > 2 then
      error ('input can be 1D for a single sample or 2D for a batch')
   end
   if x:dim() == 1 and x:size(1) ~= D:size(2) then
      error ('input size should be same as row size of Dictionary')
   end
   if x:dim() == 2 and x:size(2) ~= D:size(2) then
      error ('input size[2] should be same as row size of Dictionary')
   end
   local params = params or {}
   -- related to FISTA
   params.L = params.L or 0.1
   params.Lstep = params.Lstep or 1.5
   params.maxiter = params.maxiter or 50
   params.maxline = params.maxline or 20
   params.errthres = params.errthres or 1e-4
   params.doFistaUpdate = params.doFistaUpdate or true
   
   -- temporary stuff that might be good to keep around
   params.reconstruction = params.reconstruction or torch.Tensor()
   params.gradf = params.gradf or torch.Tensor()
   params.code = params.code or torch.Tensor()

   -- resize temporary stuff
   params.reconstruction:resizeAs(input)
   if x:dim() == 2 then
      params.code:resize(x:size(1),D:size(1))
   else
      params.code:resize(D:size(1))
   end
   params.gradf:resizeAs(params.code)

   local temp = torch.Tensor()

   -- CREATE FUNCTION CLOSURES
   -- smooth function
   local function f(x,mode)
      -- function evaluation
      if x:dim() == 1 then
	 params.reconstruction:addmv(0,1,D,x)
      else
	 params.reconstruction:addmm(0,1,x,D:t())
      end
      local fval = input:dist(params.reconstruction)

      -- derivative calculation
      if mode and mode:match('dx') then
	 params.reconstruction:add(-1,input):mul(2)
	 params.gradf:resizeAs(x)
	 if x:dim() == 1 then
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

-- FISTA with backtracking line search
-- f  smooth function
-- g  non-smooth function
-- pl minimizer of intermediate problem Q(x,y)
-- xinit initial point
function unsup.FistaLS(f, g, pl, xinit, params)
   
   local params = params or {}
   local L = params.L or 0.1
   local Lstep = params.Lstep or 1.5
   local maxiter = params.maxiter or 50
   local maxline = params.maxline or 20
   local errthres = params.errthres or 1e-4
   local doFistaUpdate = params.doFistaUpdate
   local verbose = params.verbose 
   
   local xk = xinit
   -- we start from all zeros
   local xkm = torch.Tensor():resizeAs(xk):zero() -- previous iteration
   local y = torch.Tensor():resizeAs(xk):zero()   -- fista iteration
   local ply = torch.Tensor():resizeAs(y)         -- soft shrinked y


   local history = {} -- keep track of stuff
   local niter = 0    -- number of iterations done
   local converged = false  -- are we done?
   local tk = 1      -- momentum param for FISTA
   local tkp = 0


   local gy = g(y)
   local fval = math.huge -- fval = f+g
   while not converged and niter < maxiter do

      -- run through smooth function (code is input, input is target)
      local fy = f(y)
      -- get derivatives from smooth function
      local gfy = df(y)
      
      local fply = 0
      local gply = 0
      local Q = 0
      
      ----------------------------------------------
      -- do line search to find new current location starting from fista loc
      local nline = 0
      local linesearchdone = false
      while not linesearchdone do
	 -- take a step in gradient direction of smooth function
	 ply:copy(y)
	 --ply:add(-1/L,GFy)
	 -- soft shrink
	 --ply:shrinkage(lambda/L)
	 pl(ply,L,gfy)
	 xk:copy(ply) -- this is candidate for new current iteration
	 -- evaluate this point F(ply)
	 fply = f(ply)
	 -- evaluate approximation Q(beta,y)
	 -- non smooth function
	 --Gply = lambda*self.nonSmoothFunc:forward(ply)

	 -- ply - y
	 ply:add(-1, y)
	 -- <ply-y , \Grad(f(y))>
	 local Q2 = gfy:dot(ply)
	 -- L/2 ||beta-y||^2
	 local Q3 = L/2 * ply:dot(ply)
	 -- Q(beta,y) = F(y) + <beta-y , \Grad(F(y))> + L/2||beta-y||^2 + G(beta)
	 Q = fy + Q2 + Q3
	 -- check if F(beta) < Q(pl(y),\t)
	 if fply <= Q then --and Fply + Gply <= F then
	    linesearchdone = true
	 elseif  nline >= maxline then
	    linesearchdone = true
	    xk:copy(xkm) -- if we can't find a better point, current iter = previous iter
	    fply = f(xk)
	    gply = g(xk)
	    --print('oops')
	 else
	    L = L * Lstep
	 end
	 nline = nline + 1
	 --print(linesearchdone,nline,L,Fy,Q2,Q3,Q,Fbeta,Gbeta)
      end
      -- end line search
      ---------------------------------------------
      gply = g(xk)
      niter = niter + 1

      -- bookeeping
      fval = fply + gply
      history[niter] = {}
      history[niter].nline = nline
      history[niter].L  = L
      history[niter].F  = fval
      history[niter].Fply = fply
      history[niter].Gply = gply
      history[niter].Q  = Q
      if verbose then
	 history[niter].xk = xk:clone()
	 history[niter].y  = y:clone()
      end

      -- are we done?
      if niter > 1 and math.abs(history[niter].F - history[niter-1].F) <= errthres then
	 converged = true
	 return xk,history
      end

      if niter >= maxiter then
	 return xk,history
      end

      --if niter > 1 and history[niter].F > history[niter-1].F then
      --print(niter, 'This was supposed to be a convex function, we are going up')
      --converged = true
      --return xk,history
      --end

      if doFistaUpdate then
	 -- do the FISTA step
	 tkp = (1 + math.sqrt(1 + 4*tk*tk)) / 2
	 -- x(k-1) = x(k-1) - x(k)
	 xkm:add(-1,xk)
	 -- y(k+1) = x(k) + (1-t(k)/t(k+1))*(x(k-1)-x(k))
	 y:copy(xk)
	 y:add( (1-tk)/tkp , xkm)
	 -- store for next iterations
	 -- x(k-1) = x(k)
 	 local t = xkm
 	 xkm = xk
 	 xk = t
      else
	 y:copy(xk)
      end
      -- t(k) = t(k+1)
      tk = tkp
   end
   error('not supposed to be here')
end

