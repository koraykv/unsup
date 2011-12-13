
local data_verbose = false

function getdata(datafile, inputsize, std)
   local data = torch.DiskFile(datafile,'r'):binary():readObject()
   local dataset ={}

   local std = std or 0.2
   local nsamples = data:size(1)
   local nrows = data:size(2)
   local ncols = data:size(3)

   function dataset:size()
      return nsamples
   end

   function dataset:selectPatch(nr,nc)
      local imageok = false
      if simdata_verbose then
	 print('selectPatch')
      end
      while not imageok do
	 --image index
	 local i = math.ceil(random.uniform(1e-12,nsamples))
	 local im = data:select(1,i)
	 -- select some patch for original that contains original + pos
	 local ri = math.ceil(random.uniform(1e-12,nrows-nr))
	 local ci = math.ceil(random.uniform(1e-12,ncols-nc))
	 local patch = im:narrow(1,ri,nr)
	 patch = patch:narrow(2,ci,nc)
	 local patchstd = patch:std()
	 if data_verbose then
	    print('Image ' .. i .. ' ri= ' .. ri .. ' ci= ' .. ci .. ' std= ' .. patchstd)
	 end
	 if patchstd > std then
	    return patch,i
	 end
      end
   end

   local dsample = torch.Tensor(inputsize*inputsize)
   setmetatable(dataset, {__index = function(self, index)
				       local sample = self:selectPatch(inputsize, inputsize)
				       dsample:copy(sample)
				       return {dsample}
				    end})
   return dataset
end

-- dataset, dataset=createDataset(....)
-- nsamples, how many samples to display from dataset
-- nrow, number of samples per row for displaying samples
-- scale, scale at which to draw dataset
function displayData(dataset, nsamples, nrow, scale)
   require 'image'
   local nsamples = nsamples or 100
   local scale = scale or 1
   local nrow = nrow or 10

   local win = nil

   cntr = 1
   local ex = {}
   for i=1,nsamples do
      local exx = dataset[1]
      ex[cntr] = exx[1]:clone():unfold(1,math.sqrt(exx[1]:size(1)),math.sqrt(exx[1]:size(1)))
      cntr = cntr + 1
   end
   --return ex
   win = image.display{image=ex, padding=1, symmetric=true, scale=scale, win=win, nrow=nrow}
   return win
end
