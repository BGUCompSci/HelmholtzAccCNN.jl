using MLDatasets
using ImageFiltering
using Images

cifar_test,  _  = CIFAR10.testdata()

function const_model!(n)
    return ones(Float64,n-1,n-1)
end

function cifar_model!(n; smooth=true)
    index = rand(1:1000) #size(cifar_test,4)
    sample = Gray.(colorview(RGB, cifar_test[:,:,1,index],cifar_test[:,:,2,index],cifar_test[:,:,3,index]))
    sample = imresize(sample, (n-1,n-1))
    if smooth == true
        sample = imfilter(sample, Kernel.gaussian(3))
    end
    sample_normal = 0.5 .+ (Float64.(sample) ./ (2*maximum(Float64.(sample))))
	return sample_normal;
end
