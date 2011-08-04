local Fista = torch.class('unsup.Fista')

function Fista:__init(F, G)
   self.smoothFunc = F
   self.nonSmoothFunc = G
   -- default params, override if you want
   self.verbose = false
   self.L = 0.1
   self.Lstep = 1.5
   self.maxiter = 50
   self.maxline = 20
   self.errthres = 1e-4
   self.doFistaUpdate = true
end

local function softshrink(v,alpha)
   v:apply(function(x) 
	      if x > alpha then
		 return x - alpha
	      elseif x < -alpha then
		 return x + alpha
	      else
		 return 0
	      end
	   end)
end

function Fista:forward(input, code, lambda, maxiter, errthres)

   local xk = code

   maxiter = maxiter or self.maxiter
   errthres = errthres or self.errthres


   local history = {} -- keep track of stuff
   local niter = 0    -- number of iterations done
   local converged = false  -- are we done?

   local L = self.L  -- inverse step size
   local tk = 1      -- momentum param for FISTA
   local tkp = 0

   -- we start from all zeros
   local xkm = torch.Tensor():resizeAs(xk):zero() -- previous iteration
   local y = torch.Tensor():resizeAs(xk)          -- fista iteration
   local ply = torch.Tensor():resizeAs(y)         -- soft shrinked y
   y:copy(xkm)                                    -- fista location

   -- run through smooth function (code is input, input is target)
   local Fy = self.smoothFunc:forward(y,input)
   local Gy = self.nonSmoothFunc:forward(y)
   local F = math.huge
   while not converged and niter < self.maxiter do

      -- get derivatives from smooth function
      local GFy = self.smoothFunc:backward(y,input)
      
      local Fply = 0
      local Gply = 0
      local Q = 0
      
      ----------------------------------------------
      -- do line search to find new current location starting from fista loc
      local nline = 0
      local linesearchdone = false
      while not linesearchdone do
	 -- take a step in gradient direction of smooth function
	 ply:copy(y)
	 ply:add(-1/L,GFy)
	 -- soft shrink
	 softshrink(ply, lambda/L)
	 xk:copy(ply) -- this is candidate for new current iteration
	 -- evaluate this point F(ply)
	 Fply = self.smoothFunc:forward(ply,input)
	 -- evaluate approximation Q(beta,y)
	 -- non smooth function
	 --Gply = lambda*self.nonSmoothFunc:forward(ply)

	 -- ply - y
	 ply:add(-1, y)
	 -- <ply-y , \Grad(f(y))>
	 local Q2 = GFy:dot(ply)
	 -- L/2 ||beta-y||^2
	 local Q3 = L/2 * ply:dot(ply)
	 -- Q(beta,y) = F(y) + <beta-y , \Grad(F(y))> + L/2||beta-y||^2 + G(beta)
	 Q = Fy + Q2 + Q3 + Gply
	 -- check if F(beta) < Q(pl(y),\t)
	 if Fply <= Q then --and Fply + Gply <= F then
	    linesearchdone = true
	 elseif  nline >= self.maxline then
	    linesearchdone = true
	    xk:copy(xkm) -- if we can't find a better point, current iter = previous iter
	    Fply = self.smoothFunc:forward(xk,input)
	    Gply = lambda*self.nonSmoothFunc:forward(xk)
	    print('oops')
	 else
	    L = L * self.Lstep
	 end
	 nline = nline + 1
	 --print(linesearchdone,nline,L,Fy,Q2,Q3,Q,Fbeta,Gbeta)
      end
      -- end line search
      ---------------------------------------------
      Gply = lambda*self.nonSmoothFunc:forward(xk)
      niter = niter + 1

      -- bookeeping
      F = Fply + Gply
      history[niter] = {}
      history[niter].nline = nline
      history[niter].L  = L
      history[niter].F  = F
      history[niter].Fply = Fply
      history[niter].Gply = Gply
      history[niter].Q  = Q
      if self.verbose then
	 history[niter].xk = xk:clone()
	 history[niter].y  = y:clone()
      end

      -- are we done?
      if niter > 1 and math.abs(history[niter].F - history[niter-1].F) <= errthres then
	 converged = true
	 return xk,history
      end

      if niter >= self.maxiter then
	 return xk,history
      end

      --if niter > 1 and history[niter].F > history[niter-1].F then
      --print(niter, 'This was supposed to be a convex function, we are going up')
      --converged = true
      --return xk,history
      --end

      if self.doFistaUpdate then
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

-- input is data, code is sparse representation
function Fista:backward(input, code, lambda)
   -- code is input to smoothFunc and input is the target for reconstruction
   local gF = self.smoothFunc:backward(code, input)
   local gG = self.nonSmoothFunc:backward(code)
   gF:add(lambda,gG)
   return gF
end

function Fista:write(file)
   file:writeBool(self.verbose)
   file:writeDouble(self.L)
   file:writeDouble(self.Lstep)
   file:writeInt(self.maxiter)
   file:writeDouble(self.errthres)
end

function Fista:read(file)
   self.verbose = file:readBool()
   self.L = file:readDouble()
   self.Lstep = file:readDouble()
   self.maxiter = file:readInt()
   self.errthres = file:readDouble()
end

