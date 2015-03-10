--
-- The k-means algorithm.
--
--   > x: is supposed to be an MxN matrix, where M is the nb of samples and each sample is N-dim
--   > k: is the number of kernels
--   > niter: the number of iterations
--   > batchsize: the batch size [large is good, to parallelize matrix multiplications]
--   > callback: optional callback, at each iteration end
--   > verbose: prints a progress bar...
--
--   < returns the k means (centroids) + the counts per centroid
--
function unsup.kmeans(x, k, niter, batchsize, callback, verbose)
   -- args
   local help = 'centroids,count = unsup.kmeans(Tensor(npoints,dim), k [, niter, batchsize, callback, verbose])'
   x = x or error('missing argument: ' .. help)
   k = k or error('missing argument: ' .. help)
   niter = niter or 1
   batchsize = batchsize or math.min(1000, (#x)[1])

   -- resize data
   local k_size = x:size()
   k_size[1] = k
   if x:dim() > 2 then
      x = x:reshape(x:size(1), x:nElement()/x:size(1))
   end

   -- some shortcuts
   local sum = x.sum
   local max = x.max
   local pow = x.pow

   -- dims
   local nsamples = (#x)[1]
   local ndims = (#x)[2]

   -- initialize means
   local x2 = sum(pow(x,2),2)
   local centroids = x.new(k,ndims):normal()
   for i = 1,k do
      centroids[i]:div(centroids[i]:norm())
   end
   local totalcounts = x.new(k):zero()
      
   -- callback?
   if callback then callback(0,centroids:reshape(k_size),totalcounts) end

   -- do niter iterations
   for i = 1,niter do
      -- progress
      if verbose then xlua.progress(i,niter) end

      -- sums of squares
      local c2 = sum(pow(centroids,2),2)*0.5

      -- init some variables
      local summation = x.new(k,ndims):zero()
      local counts = x.new(k):zero()
      local loss = 0

      -- process batch
      for i = 1,nsamples,batchsize do
         -- indices
         local lasti = math.min(i+batchsize-1,nsamples)
         local m = lasti - i + 1

         -- k-means step, on minibatch
         local batch = x[{ {i,lasti},{} }]
         local batch_t = batch:t()
         local tmp = centroids * batch_t
         for n = 1,(#batch)[1] do
            tmp[{ {},n }]:add(-1,c2)
         end
         local val,labels = max(tmp,1)
         loss = loss + sum(x2[{ {i,lasti} }]*0.5 - val:t())

         -- count examplars per template
         local S = x.new(m,k):zero()
         for i = 1,(#labels)[2] do
            S[i][labels[1][i]] = 1
         end
         summation:add( S:t() * batch )
         counts:add( sum(S,1) )
      end

      -- normalize
      for i = 1,k do
         if counts[i] ~= 0 then
            centroids[i] = summation[i]:div(counts[i])
         end
      end

      -- total counts
      totalcounts:add(counts)

      -- callback?
      if callback then 
         local ret = callback(i,centroids:reshape(k_size),totalcounts) 
         if ret then break end
      end
   end

   -- done
   return centroids:reshape(k_size),totalcounts
end
