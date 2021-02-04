using MLDatasets
using ImageFiltering
using Images

cifar_train,_  = CIFAR10.traindata()

function const_model!(n)
    return ones(Float64,n-1,n-1)
end

function cifar_model!(n; smooth=true)
    index = rand(1:size(cifar_train,4))
    sample = Gray.(colorview(RGB, cifar_train[:,:,1,index],cifar_train[:,:,2,index],cifar_train[:,:,3,index]))
    sample = imresize(sample, (n-1,n-1))
    if smooth == true
        sample = imfilter(sample, Kernel.gaussian(3))
    end
    sample = Float64.(sample)
    threshold = 0.01 * rand(50:99)
    sample_normal = threshold .+ (((1.0 - threshold) .* sample) ./ maximum(sample))
	return sample_normal;
end
