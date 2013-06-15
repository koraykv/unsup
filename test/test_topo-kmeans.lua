
----------------------------------------------------------------------
-- E. Culurciello May 2013
-- Run k-means on Berkeley image and generate TOPOGRAPHIC layers filters
-- topographic clustering learning technique
-- with custom k-mean code: topo-kmeans.lua
----------------------------------------------------------------------

require 'image'
require 'unsup'
require 'nnx'
require 'topo-kmeans'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Get k-means templates on directory of images')
cmd:text()
cmd:text('Options')
cmd:option('-datafile', 'http://data.neuflow.org/data/tr-berkeley-N5K-M56x56-lcn.bin', 'Dataset URL')
cmd:option('-visualize', true, 'display kernels')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 8, 'threads')
cmd:option('-inputsize', 9, 'size of each input patches') -- OL: 7x7
cmd:option('-nkernels1', 256, 'number of kernels 1st layer') -- OL: 16
cmd:option('-niter1', 1, 'nb of k-means iterations') -- ned fewer now because we 
cmd:option('-batchsize', 1000, 'batch size for k-means\' inner loop')
cmd:option('-nsamples', 10*1000, 'nb of random training samples')
cmd:option('-initstd1', 0.1, 'standard deviation to generate random initial templates')
cmd:text()
opt = cmd:parse(arg or {}) -- pass parameters to rest of file:

--if not qt then
--   opt.visualize = false
--end

torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

is = opt.inputsize
nk1 = opt.nkernels1

normkernel = image.gaussian1D(7)

print 'TOPOGRAPHIC Clustering Learning test on the Berkely background dataset!'

----------------------------------------------------------------------
-- loading and processing dataset:
dofile '1_data.lua'

filename = '../../datasets/'..paths.basename(opt.datafile)
if not paths.filep(filename) then
   os.execute('wget ' .. opt.datafile .. '; '.. 'tar xvf ' .. filename)
end
dataset = getdata(filename, opt.inputsize)

trsize = 256--dataset:size()

trainData = {
   data = torch.Tensor(trsize, 3, dataset[1][3]:size(1), dataset[1][3]:size(2)),
   size = function() return trsize end
}
for t = 1,trsize do
   trainData.data[t][1] = dataset[t][3]
   trainData.data[t][2] = trainData.data[t][1]
   trainData.data[t][3] = trainData.data[t][1]
   xlua.progress(t, trainData:size())
end

f256S = trainData.data[{{1,256}}]
image.display{image=f256S, nrow=16, nrow=16, padding=2, zoom=1, legend='Input images'}


-- verify dataset statistics:
trainMean = trainData.data:mean()
trainStd = trainData.data:std()
print('1st layer training data mean: ' .. trainMean)
print('1st layer training data standard deviation: ' .. trainStd)


----------------------------------------------------------------------
print '==> generating 1st layer filters:'
print '==> extracting patches' -- only extract on Y channel (or R if RGB) -- all ok
data1 = torch.Tensor(opt.nsamples,is*is)
for i = 1,opt.nsamples do
   img = math.random(1,dataset:size())
   img2 = dataset[i][3]
   x = math.random(1,dataset[1][3]:size(1)-is+1)
   y = math.random(1,dataset[1][3]:size(2)-is+1)
   randompatch = img2[{{y,y+is-1},{x,x+is-1} }]
   -- normalize patches to 0 mean and 1 std:
   randompatch:add(-randompatch:mean())
   randompatch:div(randompatch:std())
   data1[i] = randompatch
end

-- show a few patches:
f256S = data1[{{1,256}}]:reshape(256,is,is)
image.display{image=f256S, nrow=16, nrow=16, padding=2, zoom=2, legend='Patches for 1st layer learning'}

print '==> running k-means'
 function cb (kernels1)
    if opt.visualize then
       win1 = image.display{image=kernels1:reshape(nk1,is,is), padding=1, symmetric=true, 
       zoom=2, win=win1, nrow=math.floor(math.sqrt(nk1)), legend='1st layer filters'}
    end
end                    
kernels1 = topokmeans(data1, nk1, nil, opt.initstd1, opt.niter1, 'topo+', cb, true)
 
-- clear nan kernels if kmeans initstd is not right!
for i=1,nk1 do   
   if torch.sum(kernels1[i]-kernels1[i]) ~= 0 then 
      print('Found NaN kernels!') 
      kernels1[i] = torch.zeros(kernels1[1]:size()) 
   end
 
   -- normalize kernels to 0 mean and 1 std:  
	kernels1[i]:add(-kernels1[i]:mean())
	kernels1[i]:div(kernels1[i]:std())
end

-- visualize final kernels:
image.display{image=kernels1:reshape(nk1,is,is), padding=1, symmetric=true, 
       zoom=2, win=win1, nrow=math.floor(math.sqrt(nk1)), legend='1st layer filters'}





