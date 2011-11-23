
require 'unsup'
require 'lab'
require 'plot'

lab.setgnuplotterminal('x11')

function gettableval(tt,v)
   local x = torch.Tensor(#tt)
   for i=1,#tt do x[i] = tt[i][v] end
   return x
end
function doplots(v)
   v = v or 'F'
   local fistaf = torch.DiskFile('fista.bin'):binary()
   local istaf = torch.DiskFile('ista.bin'):binary()
   
   local hfista = fistaf:readObject()
   fistaf:close()
   local hista = istaf:readObject()
   istaf:close()

   lab.figure()
   plot.plot({'fista ' .. v,gettableval(hfista,v)},{'ista ' .. v, gettableval(hista,v)})
end

seed = seed or 123
if dofista == nil then
   dofista = true
else
   dofista = not dofista
end

random.manualSeed(seed)
math.randomseed(seed)
nc = 3
ni = 30
no = 100
x = torch.Tensor(ni):zero()
fista = unsup.LinearFista(ni,no,0.1,200)
fista:normalize()
fista.fista.verbose = true
fista.fista.doFistaUpdate = dofista
--fista.fista.maxiter = 3

fista.fista.smoothFunc.module.weight:copy(lab.randn(ni,no))
fista:normalize()


mixi = torch.Tensor(nc)
mixj = torch.Tensor(nc)
for i=1,nc do
   local ii = math.random(1,no)
   local cc = random.uniform(0,1/nc)
   mixi[i] = ii;
   mixj[i] = cc;
   --fista.fista.smoothFunc.module.weight:select(2,ii):copy(lab.randn(ni))
   print(ii,cc)
   x:add(cc, fista.fista.smoothFunc.module.weight:select(2,ii))
end

code,rec,h = fista:forward(x);

plot.figure(1)
plot.plot({'data',mixi,mixj,'+'},{'code',lab.linspace(1,no,no),code,'+'})
lab.title('Fista = ' .. tostring(fista.fista.doFistaUpdate))

plot.figure(2)
plot.plot({'input',lab.linspace(1,ni,ni),x,'+-'},{'reconstruction',lab.linspace(1,ni,ni),rec,'+-'});
plot.title('Reconstruction Error : ' ..  x:dist(rec) .. ' ' .. 'Fista = ' .. tostring(fista.fista.doFistaUpdate))
--w2:axis(0,ni+1,-1,1)

if fista.fista.doFistaUpdate then
   print('Running FISTA')
   fname = 'fista.bin'
else
   print('Running ISTA')
   fname = 'ista.bin'
end
ff = torch.DiskFile(fname,'w'):binary()
ff:writeObject(h)
ff:close()

