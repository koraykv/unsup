-- ZCA-Whitening
--
-- Input: 
--  - data tensor M x N1 [x N2 x ...] (required); at least 2D.
--  - means: 1D tensor of size N = N1 x N2 x ... (flattned).
--  - P: ZCA-transfor matrix of size N x N.
--
-- Behavior: 
--  - if both means and P are provided, the ZCA-transformed data is returned, alongside means and P (unchanged). 
--  - otherwise, means and P are computed and returned, preceded by the transformed data. 
--
-- Input arguments are never changed.
--
function unsup.zca_whiten(data, means, P, invP)
    local auxdata = data:clone()
    local dims = data:size()
    local nsamples = dims[1]
    local n_dimensions = data:nElement() / nsamples
    if data:dim() >= 3 then
        auxdata:resize(nsamples, n_dimensions)
    end
    if not means or not P or not invP then 
        -- compute mean vector if not provided 
        means = torch.mean(auxdata, 1):squeeze()
        -- compute transformation matrix P if not provided
        local ce, cv = unsup.pcacov(auxdata)
        ce:add(1e-5):sqrt()
        local invce = ce:clone():pow(-1)
        local invdiag = torch.diag(invce)
        P = torch.mm(cv, invdiag)
        P = torch.mm(P, cv:t())

        -- compute inverse of the transformation
        local diag = torch.diag(ce)
        invP = torch.mm(cv, diag)
        invP = torch.mm(invP, cv:t())
    end
    -- remove the means
    auxdata:add(torch.ger(torch.ones(nsamples), means):mul(-1))
    -- transform in ZCA space
    auxdata = torch.mm(auxdata, P)

    auxdata:resizeAs(data)
    return auxdata, means, P, invP
end

function unsup.zca_colour(data, means, P, invP)
    local auxdata = data:clone()
    local dims = data:size()
    local nsamples = dims[1]
    local n_dimensions = data:nElement() / nsamples
    assert(means)
    assert(invP)
    if data:dim() >= 3 then
        auxdata:resize(nsamples, n_dimensions)
    end
    -- transform in ZCA space
    auxdata = torch.mm(auxdata, invP)
    -- remove the means
    auxdata:add(torch.ger(torch.ones(nsamples), means))

    data:copy(auxdata:resizeAs(data))
    return auxdata, means, P, invP
end


-- Function computes return a linear layer which applies a ZCA transform 
-- to its input using a precomputed (static) transformation matrix. 
-- if not specified, the transformation parameters are computed from data
function unsup.zca_layer(data, means, P, invP)
    local auxdata
    if not means or not P or not invP then 
        auxdata, means, P, invP  = unsup.zca_whiten(data)
    end
    local n_dimensions = data:nElement() / data:size(1)
    local linear = nn.Linear(n_dimensions, n_dimensions)
    linear.weight:copy(P:t())
    linear.bias:fill(0)
    linear.bias:copy(linear:forward(means):mul(-1))
    local layer
    if data:nDimension() > 2 then 
        layer = nn.Sequential()
        layer:add(nn.Reshape(data:size(1), n_dimensions))
        layer:add(linear)
        layer:add(nn.Reshape(data:size()))
    else
        layer = linear
    end
    return layer, means, P, invP
end

-- PCA-Whitening
--
-- Input: 
--  - data tensor M x N1 [x N2 x ...] (required); at least 2D.
--  - means: 1D tensor of size N = N1 x N2 x ... (flattned).
--  - P: PCA-transfor matrix of size N x N.
--
-- Behavior: 
--  - if both means and P are provided, the PCA-transformed data is returned, alongside means, P and invP (unchanged). 
--  - otherwise, means, P and invP are computed and returned, preceded by the transformed data. 
--
-- Input arguments are never changed.
--

function unsup.pca_whiten(data, means, P, invP)
    local auxdata = data:clone()
    local dims = data:size()
    local nsamples = dims[1]
    local n_dimensions = data:nElement() / nsamples
    if data:dim() >= 3 then
        auxdata:resize(nsamples, n_dimensions)
    end
    if not means or not P then
        -- compute mean vector if not provided 
        means = torch.mean(auxdata, 1):squeeze()
        -- compute transformation matrix P if not provided
        local ce, cv = unsup.pcacov(auxdata)
        ce:add(1e-5):sqrt()
        local invce = ce:clone():pow(-1)
        local invdiag = torch.diag(invce)
        P = torch.mm(cv, invdiag)

        -- compute inverse of the transformation
        local diag = torch.diag(ce)
        invP = torch.mm(diag, cv:t())
    end
    -- remove the means
    auxdata:add(torch.ger(torch.ones(nsamples), means):mul(-1))
    -- transform in ZCA space
    auxdata = torch.mm(auxdata, P)

    auxdata:resizeAs(data)
    return auxdata, means, P, invP
end

function unsup.pca_colour(data, means, P, invP)
    local auxdata = data:clone()
    local dims = data:size()
    local nsamples = dims[1]
    local n_dimensions = data:nElement() / nsamples
    assert(means)
    assert(invP)
    if data:dim() >= 3 then
        auxdata:resize(nsamples, n_dimensions)
    end
    -- transform in PCA space
    auxdata = torch.mm(auxdata, invP)
    -- add back the means
    auxdata:add(torch.ger(torch.ones(nsamples), means))

    data:copy(auxdata:resizeAs(data))
    return auxdata, means, P, invP
end


-- Function computes return a linear layer which applies a PCA transform 
-- to its input using a prec-computed (static) transformation matrix. 
-- if not specified, the transformation parameters are computed from data
function unsup.pca_layer(data, means, P, invP)
    local auxdata
    if not means or not P or not invP then 
        auxdata, means, P, invP  = unsup.pca_whiten(data)
    end
    local n_dimensions = data:nElement() / data:size(1)
    local linear = nn.Linear(n_dimensions, n_dimensions)
    linear.weight:copy(P:t())
    linear.bias:fill(0)
    linear.bias:copy(linear:forward(means):mul(-1))
    local layer
    if data:nDimension() > 2 then 
        layer = nn.Sequential()
        layer:add(nn.Reshape(data:size(1), n_dimensions))
        layer:add(linear)
        layer:add(nn.Reshape(data:size()))
    else
        layer = linear
    end
    return layer, means, P, invP
end

