require 'unsup'

function gauss1d(n,mean,std)
   mean = mean or 0
   std = std or 1
   local x = torch.randn(n)
   x:add(mean)
   x:mul(std)
   return x
end

function gauss2d(n,mean,std)
   mean = mean or {0,0}
   std = std or {1,1}
   
   local x = gauss1d(n,mean[1],std[1])
   local y = gauss1d(n,mean[2],std[2])
   return torch.cat(x,y,2)
end

function rotmat(deg)
   local rad = deg/180*math.pi
   local cos = math.cos
   local sin = math.sin
   return torch.Tensor{{cos(rad), sin(rad)},
		                 {-sin(rad),cos(rad)}}
end

function rotate(x,deg)
   local rm = rotmat(deg)
   if x:dim() == 1 then
      local n = x:size(1)
      x=x:clone():resize(1,n)
   end
   return torch.mm(x,rm)
end

x=gauss2d(10000,{0,0},{1,4})
xr=rotate(x,-45)
e,v=unsup.pca(xr)
vv=v*torch.diag(e)
vv=torch.cat(torch.zeros(2,2),vv:t())

gnuplot.figure(1)
gnuplot.axis('equal')
gnuplot.plot({'orginal',x},{'rotated',xr},{'PC1',vv[{ {1,1} , {} }],'v'},{'PC2',vv[{ {2,2} , {} }],'v'})
gnuplot.axis('equal')
gnuplot.grid(true)

x1=gauss2d(10000,{-3,0},{1,4})
x2=gauss2d(10000,{ 3,0},{1,4})
x=torch.cat(x1,x2,1)
e,v=unsup.pca(x)
vv=v*torch.diag(e)
vv=torch.cat(torch.zeros(2,2),vv:t())

gnuplot.figure(2)
gnuplot.axis('equal')
gnuplot.plot({'x1',x1},{'x2',x2},{'PC1',vv[{ {1,1} , {} }],'v'},{'PC2',vv[{ {2,2} , {} }],'v'})
gnuplot.axis('equal')
gnuplot.grid(true)

