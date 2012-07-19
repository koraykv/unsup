
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
--   < returns the k means (centroids)
--
function unsup.kmeans(x, k, std, niter, batchsize, callback, verbose)
   -- args
   batchsize = batchsize or 1000
   std = std or 0.1

   -- some shortcuts
   local sum = torch.sum
   local max = torch.max
   local pow = torch.pow
   local randn = torch.randn
   local zeros = torch.zeros

   -- dims
   local nsamples = (#x)[1]
   local ndims = (#x)[2]

   -- initialize means
   local x2 = sum(pow(x,2),2)
   local centroids = randn(k,ndims)*std

   -- do niter iterations
   for i = 1,niter do
      -- progress
      if verbose then xlua.progress(i,niter) end

      -- sums of squares
      local c2 = sum(pow(centroids,2),2)*0.5

      -- init some variables
      local summation = zeros(k,ndims)
      local counts = zeros(k)
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
         local S = zeros(m,k)
         for i = 1,(#labels)[2] do
            S[i][labels[1][i]] = 1
         end
         summation:add( S:t() * batch )
         counts:add( sum(S,1) )
      end

      -- normalize
      centroids[{}] = summation
      for i = 1,k do
         centroids[i]:div(counts[i])
      end

      -- callback?
      if callback then callback(centroids) end
   end

   -- done
   return centroids
end
