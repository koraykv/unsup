require 'lab'
require 'unsup'

random.manualSeed(123)
math.randomseed(123)
nc = 100
ns = 1000
ndim = 2
x = lab.randn(ns*nc,ndim):mul(ns/10);
d = torch.Tensor(2,nc)
for i=0,nc-1 do
   for j=1,ndim do
      x:narrow(1,i*ns+1,ns):select(2,j):add(math.random(ns)*2)
   end
end

for i=1,x:size(2) do
   x:select(2,i):add(-1*x:select(2,i):mean())
   x:select(2,i):div(x:select(2,i):std())
end

-- random.manualSeed(123)
-- math.randomseed(123)
-- km = unsup.Kmeans()
-- ndist,nind,cc=km:run_slow(x,nc,100)

random.manualSeed(123)
math.randomseed(123)
km2 = unsup.Kmeans()
ndist2,nind2,cc2=km2:run(x,nc,100)
-- ndist,nind,cc=km:run(x,nc*3,1)
-- for i=1,30 do
--    ndist,nind,cc = km:run(x,nc*3,1,km.dictionary)
--    lab.plot({'data',x:select(2,1),x:select(2,2),'.'},
-- 	    {'centers',km.dictionary:select(1,1),km.dictionary:select(1,2)})
-- end
--print(km.dictionary:dist(km2.dictionary))

for i=1,ndim-1,2 do
   lab.plot({'data',x:select(2,i),x:select(2,i+1),'.'},
	    {'centers',km2.dictionary:select(1,i),km2.dictionary:select(1,i+1),'+'})
   lab.title('Dims ' .. i .. ' vs ' .. i+1)
   os.execute('sleep 2')
end