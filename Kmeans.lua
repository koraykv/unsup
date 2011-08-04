local Kmeans = torch.class('unsup.Kmeans')

function Kmeans:__init()

   self.dictionary = torch.Tensor()

end

function Kmeans:reset(m,n)
   self.dictionary = lab.randn(m,n)
end

function Kmeans:normalize()
   -- normalize the dictionary
   for i=1,self.dictionary:size(2) do
      local col = self.dictionary:select(2,i)
      col:div(col:std()+1e-12)
   end
end

-- data (mxn) each row is data
-- k number of centers
-- itermax max number of iterations
function Kmeans:run_slow(data, k, itermax, dict)

   -- init the dictionary
   if not dict then
      self:reset(data:size(2),k)
   else
      if k ~= dict:size(2) then
	 error('k and warm start dictionary size do not match')
      end
      self.dictionary:resizeAs(dict):copy(dict)
   end

   --print(self.dictionary)
   local dic = self.dictionary
   local dist,ind = nil,torch.LongTensor(data:size(1)):fill(0)
   local iter,nmove = 0,data:size(1)
   local counts = torch.Tensor(k):fill(0)
   
   local avtime_encode = 0
   local avtime_iter = 0
   local avtime_update = 0
   while iter < itermax and nmove > 0 do

      local t1i = lab.tic()
      counts:fill(0)
      
      -- assign data to centers
      local t1=lab.tic()
      local ndist,nind = self:encode(data)
      avtime_encode = avtime_encode + lab.toc(t1)

      -- calculate new centers
      dic:zero()
      nmove = 0
      local t2=lab.tic()
      for i=1,data:size(1) do
	 --print(i,nind[i])
	 dic:select(2,nind[i]):add(data:select(1,i))
	 counts[nind[i]] = counts[nind[i]]+1
	 if ind and nind[i] ~= ind[i] then
	    nmove = nmove + 1
	 end
      end
      avtime_update = avtime_update + lab.toc(t2)
      

      -- if someone is not assigned anything, reinitialize
      for i=1,dic:size(2) do
	 if counts[i] == 0 then
	    dic:select(2,i):copy(lab.randn(dic:size(1)))
	 else
	    dic:select(2,i):div(counts[i])
	 end
      end

      --print('Iteration ' .. iter .. ' dist=' .. ndist:sum() .. ' nmove=' .. nmove)
      ind = nind
      dist = ndist
      iter = iter + 1
      avtime_iter = avtime_iter + lab.toc(t1i)
   end
   print('Number of Iterations='..iter)
   print('Time iter='.. avtime_iter, 'Time encode='.. avtime_encode, 'Time update='..avtime_update)
   return dist,ind,counts
end
-- data (mxn) each row is data
-- k number of centers
-- itermax max number of iterations
function Kmeans:run(data, k, itermax, dict)

   -- init the dictionary
   if not dict then
      self:reset(data:size(2),k)
   else
      if k ~= dict:size(2) then
	 error('k and warm start dictionary size do not match')
      end
      self.dictionary:resizeAs(dict):copy(dict)
   end

   --print(self.dictionary)
   local dic = self.dictionary
   local dist,ind = nil,torch.LongTensor(data:size(1)):fill(0)
   local iter,nmove = 0,data:size(1)
   local counts = torch.LongTensor(k):fill(0)
   
   local avtime_encode = 0
   local avtime_iter = 0
   local avtime_update = 0
   while iter < itermax and nmove > 0 do

      local t1i = lab.tic()
      counts:fill(0)
      
      -- assign data to centers
      local t1=lab.tic()
      local ndist,nind = self:encode2(data)
      avtime_encode = avtime_encode + lab.toc(t1)

      -- calculate new centers
      local t2=lab.tic()
      nmove = data.unsup.Kmeans_update(self,data,ind,nind,counts)
      avtime_update = avtime_update + lab.toc(t2)
      

      -- if someone is not assigned anything, reinitialize
      for i=1,dic:size(2) do
	 if counts[i] == 0 then
	    dic:select(2,i):copy(lab.randn(dic:size(1)))
	 else
	    dic:select(2,i):div(counts[i])
	 end
      end

      --print('Iteration ' .. iter .. ' dist=' .. ndist:sum() .. ' nmove=' .. nmove)
      ind = nind
      dist = ndist
      iter = iter + 1
      avtime_iter = avtime_iter + lab.toc(t1i)
   end
   print('Number of Iterations='..iter)
   print('Time iter='.. avtime_iter, 'Time encode='.. avtime_encode, 'Time update='..avtime_update)
   return dist,ind,counts
end

function Kmeans:encode(data)
   
   local dic = self.dictionary
   local alldist = torch.Tensor(data:size(1),dic:size(2)):zero()

   for i=1,dic:size(2) do
      local dici = dic:select(2,i):clone()
      local dicirep = torch.Tensor(dici:storage(),1,data:size(1),0,dici:size(1),1)
      local dist = data - dicirep
      dist:cmul(dist)
      --print(dist)
      alldist:select(2,i):copy(lab.sum(dist))
      --print(alldist)
   end
   -- get the closest kernel distance and index
   local dist,ind = lab.min(alldist)
   dist = dist:select(2,1)
   ind = ind:select(2,1)
   return dist,ind
end   

function Kmeans:encode2(data)
   
   alldist = data.unsup.Kmeans_getdist(self,data)
--    local dic = self.dictionary
--    local alldist = torch.Tensor(data:size(1),dic:size(2)):zero()

--    for i=1,dic:size(2) do
--       local dici = dic:select(2,i):clone()
--       local dicirep = torch.Tensor(dici:storage(),1,data:size(1),0,dici:size(1),1)
--       local dist = data - dicirep
--       dist:cmul(dist)
--       alldist:select(2,i):copy(lab.sum(dist))
--    end
   -- get the closest kernel distance and index
   local dist,ind = lab.min(alldist)
   dist = dist:select(2,1)
   ind = ind:select(2,1)
   return dist,ind
end   


function Kmeans:write(file)
   file:writeObject(self.dictionary)
end

function Kmeans:read(file)
   self.dictionary = file:readObject()
end
