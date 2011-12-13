require 'unsup'
require 'image'
require 'plot'

dofile 'demo_data.lua'
if not arg then arg = {} end

cmd = torch.CmdLine()

cmd:text()
cmd:text()
cmd:text('Training a simple sparse coding dictionary on Berkeley images')
cmd:text()
cmd:text()
cmd:text('Options')
cmd:option('-dir','outputs', 'subdirectory to save experimens in')
cmd:option('-seed', 123211, 'initial random seed')
cmd:option('-nfiltersin', 1, 'number of input convolutional filters')
cmd:option('-nfiltersout', 32, 'number of output convolutional filters')
cmd:option('-kernelsize', 9, 'size of convolutional kernels')
cmd:option('-inputsize', 9, 'size of each input patch')
cmd:option('-lambda', 1, 'sparsity coefficient')
cmd:option('-datafile', '/net/ml56/mnt/local/koray/tr-berkeley-N5K-M56x56-lcn.bin','Data set file')
cmd:option('-eta',0.01,'learning rate')
cmd:option('-momentum',0,'gradient momentum')
cmd:option('-decay',0,'weigth decay')
cmd:option('-maxiter',1000000,'max number of updates')
cmd:option('-statinterval',5000,'interval for saving stats and models')
cmd:option('-v', false, 'be verbose')
cmd:option('-wcar', '', 'additional flag to differentiate this run')
cmd:text()

local params = cmd:parse(arg)

local rundir = cmd:string('unsup', params, {dir=true})
params.rundir = params.dir .. '/' .. rundir

if paths.dirp(params.rundir) then
   error('This experiment is already done!!!')
end

os.execute('mkdir -p ' .. params.rundir)
cmd:log(params.rundir .. '/log', params)

-- init random number generator
random.manualSeed(params.seed)

-- create the dataset
data = getdata(params.datafile, params.inputsize)

-- creat unsup stuff
mlp = unsup.LinearFistaL1(params.inputsize*params.inputsize, params.nfiltersout, params.lambda )
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
local Linear = torch.getmetatable("nn.Linear")
local oldLinearUpdateParameters = Linear.updateParameters
function Linear:updateParameters(learningRate)
   -- scale the gradients so that we do not add up bluntly like in batch
   oldLinearUpdateParameters(self, learningRate/self.weight:size(2))
end
local oldLinearzeroGradParameters = Linear.zeroGradParameters
function Linear:zeroGradParameters()
   self.gradWeight:mul(params.momentum)
   self.gradBias:mul(params.momentum)
end

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

function train(module,dataset)

   local avTrainingError = torch.FloatTensor(math.ceil(params.maxiter/params.statinterval)):zero()
   local avFistaIterations = torch.FloatTensor(math.ceil(params.maxiter/params.statinterval)):zero()
   local currentLearningRate = params.eta

   local function updateSample(input, target, eta)
      local err,h = module:forward(input, target)
      module:zeroGradParameters()
      module:updateGradInput(input, target)
      module:accGradParameters(input, target)
      module:updateParameters(eta)
      return err, #h
   end

   local err = 0
   local iter = 0
   for t = 1,params.maxiter do

      local example = dataset[t]

      local serr, siter = updateSample(example[1], example[2] ,currentLearningRate)
      err = err + serr
      iter = iter + siter
      
      if math.fmod(t , params.statinterval) == 0 then
	 avTrainingError[t/params.statinterval] = err/params.statinterval
	 avFistaIterations[t/params.statinterval] = iter/params.statinterval

	 -- report
	 print('# iter=' .. t .. ' eta = ' .. currentLearningRate .. ' current error = ' .. err)

	 -- plot training error
	 plot.pngfigure(params.rundir .. '/error.png')
	 plot.plot(avTrainingError:narrow(1,1,math.max(t/params.statinterval,2)))
	 plot.title('Training Error')
	 plot.xlabel('# iterations / ' .. params.statinterval)
	 plot.ylabel('Cost')
	 -- plot training error
	 plot.pngfigure(params.rundir .. '/iter.png')
	 plot.plot(avFistaIterations:narrow(1,1,math.max(t/params.statinterval,2)))
	 plot.title('Fista Iterations')
	 plot.xlabel('# iterations / ' .. params.statinterval)
	 plot.ylabel('Fista Iterations')
	 plot.plotflush()
	 plot.closeall()

	 -- plot filters
	 local dd = image.toDisplayTensor{input=mlp.D.weight:transpose(1,2):unfold(2,9,9),padding=1,nrow=8,symmetric=true}
	 image.savePNG(params.rundir .. '/filters_' .. t .. '.png',dd)
	 
	 -- store model
	 local mf = torch.DiskFile(params.rundir .. '/model_' .. t .. '.bin','w'):binary()
	 mf:writeObject(module)
	 mf:close()

	 -- write training error
	 local tf = torch.DiskFile(params.rundir .. '/error.mat','w'):binary()
	 tf:writeObject(avTrainingError:narrow(1,1,t/params.statinterval))
	 tf:close()

	 -- write # of iterations
	 local ti = torch.DiskFile(params.rundir .. '/iter.mat','w'):binary()
	 ti:writeObject(avFistaIterations:narrow(1,1,t/params.statinterval))
	 ti:close()

	 -- update learning rate with decay
	 currentLearningRate = params.eta/(1+(t/params.statinterval)*params.decay)
	 err = 0
	 iter = 0
      end
   end
end

train(mlp,data)
