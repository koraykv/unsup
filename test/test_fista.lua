
require 'unsup'
require 'torch'
require 'gnuplot'

-- gnuplot.setgnuplotexe('/usr/bin/gnuplot44')
-- gnuplot.setgnuplotterminal('x11')

function gettableval(tt,v)
   local x = torch.Tensor(#tt)
   for i=1,#tt do x[i] = tt[i][v] end
   return x
end
function doplots(v)
   v = v or 'F'
   local fistaf = torch.DiskFile('fista3.bin'):binary()
   local istaf = torch.DiskFile('ista3.bin'):binary()
   
   local hfista = fistaf:readObject()
   fistaf:close()
   local hista = istaf:readObject()
   istaf:close()

   gnuplot.figure()
   gnuplot.plot({'fista ' .. v,gettableval(hfista,v)},{'ista ' .. v, gettableval(hista,v)})
end

seed = seed or 123
if dofista == nil then
   dofista = true
else
   dofista = not dofista
end

torch.manualSeed(seed)
math.randomseed(seed)
nc = 3
ni = 30
no = 100
x = torch.Tensor(ni):zero()

fistaparams = {}
fistaparams.doFistaUpdate = dofista
fistaparams.maxline = 10
fistaparams.maxiter = 200
fistaparams.verbose = true
fista = unsup.LinearFistaL1(ni,no,0.1, fistaparams)

D=torch.randn(ni,no)
for i=1,D:size(2) do
   D:select(2,i):div(D:select(2,i):std()+1e-12)
end
fista.D.weight:copy(D)

mixi = torch.Tensor(nc)
mixj = torch.Tensor(nc)
for i=1,nc do
   local ii = math.random(1,no)
   local cc = torch.uniform(0,1/nc)
   mixi[i] = ii;
   mixj[i] = cc;
   print(ii,cc)
   x:add(cc, D:select(2,ii))
end
err = fista:forward(x)
code = fista.code

rec = fista.D.output
--code,rec,h = fista:forward(x);

gnuplot.figure(1)
gnuplot.plot({'data',mixi,mixj,'+'},{'code',torch.linspace(1,no,no),code,'+'})
gnuplot.title('Fista = ' .. tostring(fistaparams.doFistaUpdate))

gnuplot.figure(2)
gnuplot.plot({'input',torch.linspace(1,ni,ni),x,'+-'},{'reconstruction',torch.linspace(1,ni,ni),rec,'+-'});
gnuplot.title('Reconstruction Error : ' ..  x:dist(rec) .. ' ' .. 'Fista = ' .. tostring(fistaparams.doFistaUpdate))
--w2:axis(0,ni+1,-1,1)

if dofista then
   print('Running FISTA')
   fname = 'fista3.bin'
else
   print('Running ISTA')
   fname = 'ista3.bin'
end
ff = torch.DiskFile(fname,'w'):binary()
ff:writeObject(h)
ff:close()

