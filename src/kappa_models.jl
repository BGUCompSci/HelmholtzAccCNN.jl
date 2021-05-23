using MLDatasets
using ImageFiltering
using Images

cifar_train,_  = CIFAR10.traindata()

function load_96_unlabeled_stl10()
    train_images = Array{UInt8}(undef, 96, 96, 3, 100000)
    read!(joinpath("data","STL10", "unlabeled_X.bin"), train_images)
    return (train_images)
end

function load_stl10()
    train_images = Array{UInt8}(undef, 96, 96, 3, 8000)
    read!(joinpath("data","STL10", "test_X.bin"), train_images)
    return (train_images)
end

stl10_train = load_stl10()

function generate_kappa!(n; type=0, smooth=false, threshold=50, kernel=3)
    if type == 0 # Uniform Model
        return ones(r_type, n-1, n-1)
    else
        if type == 2 # STL10 Model
            index = rand(1:size(stl10_train,4))
            sample = Gray.(colorview(RGB, stl10_train[:,:,1,index] / 255,stl10_train[:,:,2,index] / 255,stl10_train[:,:,3,index] / 255))
        else # CIFAR10 Model
            index = 7706 # Const Model - 4
            if type == 1
                index = rand(1:size(cifar_train,4))
            end
            sample = Gray.(colorview(RGB, cifar_train[:,:,1,index],cifar_train[:,:,2,index],cifar_train[:,:,3,index]))
        end

        # Resize
        sample = imresize(sample, (n-1,n-1))

        # Smooth
        if smooth == true
            sample = imfilter(sample, Kernel.gaussian(kernel))
        end

        # Sample ∈ [random threshold, 1]
        sample = Float64.(sample)
        threshold = 0.01 * rand(threshold:90)
        sample_normal = threshold .+ (((1.0 - threshold) .* sample) ./ maximum(sample))

    	return sample_normal
    end
end
