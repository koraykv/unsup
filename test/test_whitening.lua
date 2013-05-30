require 'torch'
require 'unsup'
require 'image'


torch.manualSeed(0)

mytest = {}

local function get_correlated_data(sizes)
    if type(sizes) == 'table' then
        sizes = torch.LongStorage(sizes)
    end
    local n_dimensions = 1 
    for i=2,sizes:size() do
        n_dimensions = n_dimensions * sizes[i]
    end
    local gaussian_white_data = torch.randn(sizes[1], n_dimensions)
    local l = nn.Linear(n_dimensions,n_dimensions)
    l.weight:copy(torch.lt(torch.rand(l.weight:size()), 1/math.sqrt(n_dimensions)):double())
    l.bias:fill(0)
    local linearly_correlated_data = l:forward(gaussian_white_data):resize(sizes)
    return linearly_correlated_data, gaussian_white_data:resize(sizes)
end



local function pearson_correlation_coefficient(data)
    local auxdata = data:clone():resize(data:size(1), data:nElement() / data:size(1))
    local corr = torch.Tensor(auxdata:size(2), auxdata:size(2))

    local means = torch.mean(auxdata, 1):squeeze()
    local stds = torch.std(auxdata, 1):squeeze():add(1e-30)

    for i=1,auxdata:size(2) do
        for j=i,auxdata:size(2) do
            corr[j][i] = ((auxdata[{{}, i}] - means[i]) * (auxdata[{{}, j}] - means[j])) / stds[i] / stds[j] / auxdata:size(1)
            corr[i][j] = corr[j][i]
        end
    end
    return corr
end
 



local function zca_whiten_test_data(linearly_correlated_data, gaussian_white_data)
    local zca_whitened_data, means, P, invP  = unsup.zca_whiten(linearly_correlated_data)
    local stat = torch.mean(torch.pow(pearson_correlation_coefficient(zca_whitened_data)
                                            - pearson_correlation_coefficient(gaussian_white_data), 2))
    tester:assertlt(stat, 1e-3, 'corr_diff < 1e-2')
end


function mytest.zca_whiten()
    do 
        local linearly_correlated_data, gaussian_white_data = get_correlated_data({10000, 60})
        zca_whiten_test_data(linearly_correlated_data, gaussian_white_data)
    end
    
    do 
        local linearly_correlated_data, gaussian_white_data = get_correlated_data({10000, 10, 6})
        zca_whiten_test_data(linearly_correlated_data, gaussian_white_data)
    end
    
    do 
        local linearly_correlated_data, gaussian_white_data = get_correlated_data({10000, 3, 4, 5})
        zca_whiten_test_data(linearly_correlated_data, gaussian_white_data)
    end

end


local function zca_colour_test_data(linearly_correlated_data, gaussian_white_data)
    local zca_whitened_data
    zca_whitened_data, means, P, invP  = unsup.zca_whiten(linearly_correlated_data)
   
    -- colour data
    local coloured_data
    coloured_data = unsup.zca_colour(zca_whitened_data, means, P, invP)
    local stat = torch.max(torch.abs(linearly_correlated_data - coloured_data))
    tester:assertlt(stat, 1e-10, 'rec_diff < 1e-10')
end

function mytest.zca_colour()
    do
        local linearly_correlated_data, gaussian_white_data = get_correlated_data({10, 3})
        zca_colour_test_data(linearly_correlated_data, gaussian_white_data)
    end
    
    do
        local linearly_correlated_data, gaussian_white_data = get_correlated_data({10, 3, 4})
        zca_colour_test_data(linearly_correlated_data, gaussian_white_data)
    end
    
    do
        local linearly_correlated_data, gaussian_white_data = get_correlated_data({10, 3, 2, 3})
        zca_colour_test_data(linearly_correlated_data, gaussian_white_data)
    end
end


local function zca_layer_test_data(linearly_correlated_data, gaussian_white_data)
    local zca_whitened_data, means, P, invP  = unsup.zca_whiten(linearly_correlated_data)
    local layer = unsup.zca_layer(linearly_correlated_data)
    local layer_output = layer:forward(linearly_correlated_data)
    local stat = torch.max(torch.abs(zca_whitened_data - layer_output))
    tester:assertlt(stat, 1e-10, 'rec_diff < 1e-10')
    tester:asserteq(linearly_correlated_data:nDimension(), layer_output:nDimension(), 'input and output have the same number of dimensions')
    for i=1,linearly_correlated_data:nDimension() do
        tester:asserteq(linearly_correlated_data:size(i), layer_output:size(i), 'input and output match on dimension '..tostring(i))
    end
end

function mytest.zca_layer()
    do
        local linearly_correlated_data, gaussian_white_data = get_correlated_data({10, 3})
        zca_layer_test_data(linearly_correlated_data, gaussian_white_data)
    end
    
    do
        local linearly_correlated_data, gaussian_white_data = get_correlated_data({10, 3, 4})
        zca_layer_test_data(linearly_correlated_data, gaussian_white_data)
    end
    
    do
        local linearly_correlated_data, gaussian_white_data = get_correlated_data({10, 3, 2, 3})
        zca_layer_test_data(linearly_correlated_data, gaussian_white_data)
    end
end


function mytest.pca_whiten()
    local gaussian_white_data = torch.randn(1000, 300)
    
    local l = nn.Linear(gaussian_white_data:size(2),gaussian_white_data:size(2))
    l.weight:copy(torch.lt(torch.rand(l.weight:size()), 1/math.sqrt(gaussian_white_data:size(2))):double())
    l.bias:fill(0)
    local linearly_correlated_data = l:forward(gaussian_white_data)
    
    local pca_whitened_data
    pca_whitened_data, means, P, invP  = unsup.pca_whiten(linearly_correlated_data)
    local stat = torch.mean(torch.pow(pearson_correlation_coefficient(pca_whitened_data)
                                            - pearson_correlation_coefficient(gaussian_white_data), 2))
    tester:assertlt(stat, 1e-2, 'corr_diff < 1e-2')
end


function mytest.pca_colour()
    local gaussian_white_data = torch.randn(1000, 300)
    
    local l = nn.Linear(gaussian_white_data:size(2),gaussian_white_data:size(2))
    l.weight:copy(torch.lt(torch.rand(l.weight:size()), 1/math.sqrt(gaussian_white_data:size(2))):double())
    l.bias:fill(0)
    local linearly_correlated_data = l:forward(gaussian_white_data)
    
    local pca_whitened_data
    pca_whitened_data, means, P, invP  = unsup.pca_whiten(linearly_correlated_data)
   
    -- colour data
    local coloured_data
    coloured_data = unsup.pca_colour(pca_whitened_data, means, P, invP)
    local stat = torch.max(torch.abs(linearly_correlated_data - coloured_data))
    tester:assertlt(stat, 1e-10, 'rec_diff < 1e-10')
end


function mytest.pca_layer()
    local gaussian_white_data = torch.randn(1000, 300)
    local l = nn.Linear(gaussian_white_data:size(2),gaussian_white_data:size(2))
    l.weight:copy(torch.lt(torch.rand(l.weight:size()), 1/math.sqrt(gaussian_white_data:size(2))):double())
    l.bias:fill(0)
    local linearly_correlated_data = l:forward(gaussian_white_data)
    local pca_whitened_data
    pca_whitened_data, means, P, invP  = unsup.pca_whiten(linearly_correlated_data)
    local layer = unsup.pca_layer(linearly_correlated_data)
    local layer_output = layer:forward(linearly_correlated_data)
    local stat = torch.max(torch.abs(pca_whitened_data - layer_output))
    tester:assertlt(stat, 1e-10, 'rec_diff < 1e-10')
end


function main()
    torch.manualSeed(os.time())
    tester = torch.Tester()
    tester:add(mytest)
    tester:run()
end


do 
    main()
end

