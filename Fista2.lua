require 'torch'

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
      -- get derivatives from smooth function
      local fy,gfy = f(y,'dx')
      --local gfy = f(y)
      
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
	 pl(ply,L,gfy)

	 -- this is candidate for new current iteration
	 xk:copy(ply)

	 -- evaluate this point F(ply)
	 fply = f(ply)

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
	 if verbose then
	    print(niter,linesearchdone,nline,L,fy,Q2,Q3,Q,fply)
	 end
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

