using MLDatasets
using ImageFiltering
using Images
using DelimitedFiles

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

stl10_train = zeros(10,10,10,10) #load_stl10()

function generate_kappa!(n, m; type=0, smooth=false, threshold=50, kernel=3)
    if type == 0 # Uniform Model
        return ones(r_type, n-1, m-1)
    end
    if type > 4 # Geographical Model
        if type == 5
            full_kappa = readdlm("SEGmodel2Dsalt.dat")
        elseif type == 6
            full_kappa = readdlm("overthrust_XZ_slice.dat")
        elseif type == 7
            full_kappa = readdlm("MarmousiModelWithoutPad.dat")
        end
        sample = full_kappa[n+ceil(Int64,(size(full_kappa,1)-(n-1))/2-1):-1:ceil(Int64,(size(full_kappa,1)-(n-1))/2)+1,
                         ceil(Int64,(size(full_kappa,2)-(m-1))/2)+1:m+ceil(Int64,(size(full_kappa,2)-(m-1))/2)-1]
        sample = 1.0 ./ sample
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
        sample = imresize(sample, (n-1,m-1))

        # Smooth
        if smooth == true
            sample = imfilter(sample, Kernel.gaussian(kernel))
        end
    end

    # Sample ∈ [random threshold, 1]
    threshold = 0.01 * threshold #  rand(threshold:90)
    sample_normal = threshold .+ (((1.0 - threshold) .* (sample .- minimum(sample))) ./ (maximum(sample) - minimum(sample)))
    sample_normal1 = r_type.(sample_normal)

    return sample_normal1
end
