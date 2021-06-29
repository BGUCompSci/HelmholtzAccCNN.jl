using DataFrames, CSV
using Plots
using LaTeXStrings
using BSON: @load
using Dates
pyplot()
include("../test/unet/unet_test_utils.jl")

# Fonts:

bbfont = Plots.font("Helvetica", 15)
bfont = Plots.font("Helvetica", 13)
mfont = Plots.font("Helvetica", 12)
sfont = Plots.font("Helvetica", 10)
fonts = Dict(:guidefont=>bfont, :xtickfont=>sfont, :ytickfont=>sfont, :legendfont=>sfont)

# Loss graphs:

function loss_csv_to_graph!(title, path)

    fonts = Dict(:guidefont=>bfont, :xtickfont=>bfont, :ytickfont=>bfont, :legendfont=>bfont)
    df = DataFrame(CSV.File(path))
    iter = range(1, length=length(df.Train))
    p = plot(iter, df.Train, label="train", color="blue",size=(500,400),w=1.5; fonts...)
    plot!(iter, df.Test, label="validation",color="blue",w=1.5,linestyle=:dot)

    ylabel!("loss")
    xlabel!("epochs")
    savefig("C:/Siga/Repositories/HelmholtzAccCNN.jl/paper/temp/$(title)")
    savefig("C:/Siga/Repositories/HelmholtzAccCNN.jl/paper/temp/$(title).eps")
end

loss_csv_to_graph!("08_39_27_0_r", "C:/Siga/Repositories/HelmholtzAccCNN.jl/graphs/16.06/08_39_27 RADAM ND SDNUnet NaN SResidualBlock 10 elu 3 5 g=-1 t=Float32 g=t e=f r=f k=0 25 n=128 f=10_0 m=20000 bs=20 lr=0_0001 each=48 i=80 loss.csv")

function multi_errors_loss_csv_to_graph!(title, path)
    df = CSV.read(path)

    iter = range(1, length=length(df.Train))
    p = plot(iter, df.Train, label="Full Train Loss")
    # plot!(iter, df.Test, label="Full Test Loss")
    plot!(iter, df.Train_U1, label="U1 Train Loss")
    # plot!(iter, df.Test_U1, label="U1 Test Loss")
    plot!(iter, df.Train_J1, label="J1 Train Loss")
    # plot!(iter, df.Test_J1, label="J1 Test Loss")
    plot!(iter, df.Train_U2, label="U2 Train Loss")
    # plot!(iter, df.Test_U2, label="U2 Test Loss")
    plot!(iter, df.Train_J2, label="J2 Train Loss")
    # plot!(iter, df.Test_J2, label="J2 Test Loss")
    plot!(iter, df.Train_U3, label="U3 Train Loss")
    # plot!(iter, df.Test_U3, label="U3 Test Loss")
    plot!(iter, df.Train_J3, label="J3 Train Loss")
    # plot!(iter, df.Test_J3, label="J3 Test Loss")
    yaxis!(L"\Vert e^{true} -e^{net} \Vert_2", :log10)
    xlabel!("iterations")
    savefig("test/unet/results/$(title) residual graph")
end

multi_errors_loss_csv_to_graph!("23_30_11 Full Loss 4 50 2 bs=25", "graphs/23_30_11 DNUnet j=f axb=f norm=f to 1 t=Float32 k=3 50 g=t e=f da=f k=2 n=128 f=10_0 m=20000 bs=25 opt=ADAM lr=5_0e-5 each=25 i=100 loss.csv")

function multi_loss_csv_to_graph!(title, l25, l50)
    fonts = Dict(:guidefont=>bfont, :xtickfont=>bfont, :ytickfont=>bfont, :legendfont=>bfont)
    iterations = 125
    iter = range(1, length=iterations)
    p = plot(iter, min.(l25[1:iterations],l25[1]), label=L"\quad \kappa^2 \in [0.25,1]",color="blue",w=1.5,size=(500,400); fonts...)
    plot!(iter, min.(l50[1:iterations],l50[1]), label=L"\quad \kappa^2 \in [0.5,1]",color="red",w=1.5)

    ylabel!("loss")
    xlabel!("epochs")
    savefig("C:/Siga/Repositories/HelmholtzAccCNN.jl/paper/temp/$(title)")
    savefig("C:/Siga/Repositories/HelmholtzAccCNN.jl/paper/temp/$(title).eps")
end

path50 = "C:/Siga/Repositories/HelmholtzAccCNN.jl/graphs/csv/09_40_06 SDNUnet1 g=-1 t=Float32 g=t e=f k=0 50 n=128 f=10_0 m=20000 bs=20 lr=0_0001 each=70 i=155 loss.csv"
path25 = "C:/Siga/Repositories/HelmholtzAccCNN.jl/graphs/csv/13_48_25 SDNUnet1 NaN -1 g=-1 t=Float32 g=t e=f k=0 25 n=128 f=10_0 m=25000 bs=20 lr=0_0001 each=60 i=125 loss.csv"
title = "09_40_06_13_48_25_4_r"
df25 = DataFrame(CSV.File(path25))
df50 = DataFrame(CSV.File(path50))
multi_loss_csv_to_graph!(title, df25.Train, df50.Train)

function loss_error_residual_to_graph!(title, path)
    fonts = Dict(:guidefont=>bfont, :xtickfont=>bfont, :ytickfont=>bfont, :legendfont=>bfont)
    df = DataFrame(CSV.File(path))
    iter = range(1, length=length(df.Train))
    p = plot(iter, df.Train, label="total loss",size=(500,400); fonts...)
    plot!(iter, df.Residual, label="residual loss")
    plot!(iter, df.Error, label="error loss")

    ylabel!("loss")
    xlabel!("epochs")
    savefig("C:/Siga/Repositories/HelmholtzAccCNN.jl/paper/temp/$(title) residual graph")
end

loss_error_residual_to_graph!("23_10_00 RADAM ND SDNUnet NaN SResidualBlock 0 r=t", "graphs/16.06/23_10_00 RADAM ND SDNUnet NaN SResidualBlock 10 elu 3 5 g=-1 t=Float32 g=t e=f r=t k=0 25 n=128 f=10_0 m=20000 bs=20 lr=0_0001 each=24 i=50 loss.csv")

# Heatmaps:

function csv_to_heatmap!(title, path, n)
    df = DataFrame(CSV.File(path))
    heatmap(reshape(df.E, n-1, n-1), color=:jet,size=(240,160))
    savefig("C:/Siga/Repositories/HelmholtzAccCNN.jl/paper/$(title)")
end

csv_to_heatmap!("13_48_25 SDNUnet1 NaN 128 4 25 e vcycle unet", "C:/Siga/Repositories/HelmholtzAccCNN.jl/graphs/14.06/13_48_25 SDNUnet1 NaN -1 g=-1 t=Float32 g=t e=f k=0 25 n=128 f=10_0 m=25000 bs=20 lr=0_0001 each=60 i=125 128 e vcycle unet.csv", 128)

# Preconditioner tset graphs:


function unet_vs_vcycle_graph!(title, vu, v, ju; after_vcycle=false, e_vcycle_input=false)
    iterations = 15 #length(v)
    iter = range(1, length=iterations)
    fonts = Dict(:guidefont=>bfont, :xtickfont=>bfont, :ytickfont=>bfont, :legendfont=>bfont)
    factor = convergence_factor_106!(v[1:iterations])
    factor_text = "v=$(factor)"
    p = plot(iter,v[1:iterations],label=L"\rho_V = " * "$(factor)",legend=:bottomleft,color="green",size=(500,400); fonts...)
    # Jacobi Unet
    factor,_ = convergence_factor_106!(ju[1:iterations])
    factor_text = "$(factor_text) ju=$(factor)"
    plot!(iter,ju[1:iterations],label=L"\rho_{JU} = " * "$(factor)",color="blue",linestyle=:dot)
    # Vcycle Unet
    factor,_ = convergence_factor_106!(vu[1:iterations])
    factor_text = "$(factor_text) vu=$(factor)"
    plot!(iter,vu[1:iterations],label=L"\rho_{VU} = " * "$(factor)",color="blue")

    yaxis!(L"\Vert b - Hx \Vert_2", :log10)
    xlabel!("iterations")

    savefig("C:/Siga/Repositories/HelmholtzAccCNN.jl/paper/temp/$(title)")
    savefig("C:/Siga/Repositories/HelmholtzAccCNN.jl/paper/temp/$(title).eps")
end

path = "C:/Siga/Repositories/HelmholtzAccCNN.jl/graphs/22.06/08_39_27 RADAM ND SDNUnet NaN SResidualBlock 10 elu 3 5 g=-1 t=Float32 g=t e=f r=f k=0 25 n=128 f=10_0 m=20000 bs=20 lr=0_0001 each=48 i=80 t_n=128 t_axb=t t_norm=f preconditioner test.csv"
title = "08_39_27 SDNUnet 0 128"
df = DataFrame(CSV.File(path))
unet_vs_vcycle_graph!("$(title) preconditioner test", df.VU, df.V, df.JU)

function convergence_factor_106!(vector)
    val0 = vector[1]
    index = length(vector)
    for i=2:length(vector)
        if vector[i] < (val0 ./ (10^6))
            index = i
            break
        end
    end
    # length = argmin(vector)[1]
    if index > 200
        return round(((vector[index] / vector[index-30])^(1.0 / 30)), digits=3), index
    else
        return round(((vector[index] / vector[1])^(1.0 / index)), digits=3), index
    end
end

function multi_unet_vs_vcycle_graph!(title, vu25, v25, ju25, vu50, v50, ju50; after_vcycle=false, e_vcycle_input=false)
    iterations = length(vc_unet_res)
    iter = range(1, length=iterations)
    fonts = Dict(:guidefont=>bfont, :xtickfont=>bfont, :ytickfont=>bfont, :legendfont=>sfont)

    factor = convergence_factor_106!(v25)
    factor_text = "v25=$(factor)"
    p = plot(iter,v25[1:iterations],label=L"\rho_V = " * "$(factor)",legend=:bottomright,color="green", w = 2,size=(500,400); fonts...)
    factor,_ = convergence_factor_106!(ju25)
    factor_text = "$(factor_text) ju25=$(factor)"
    plot!(iter,ju25[1:iterations],label=L"\rho_{JU} = " * "$(factor); " *L"\kappa^2 \in [0.25,1]", w = 2,color="blue",linestyle=:dot)
    factor,_= convergence_factor_106!(vu25)
    factor_text = "$(factor_text) vu=$(factor)"
    plot!(iter,vu25[1:iterations],label=L"\rho_{VU} = " * "$(factor); " * L"\kappa^2 \in [0.25,1]", w = 2,color="blue")
    factor,_ = convergence_factor_106!(ju50)
    factor_text = "$(factor_text) ju25=$(factor)"
    plot!(iter,ju50[1:iterations],label=L"\rho_{JU} = " * "$(factor); " * L"\kappa^2 \in [0.5,1]", w = 2,color="red",linestyle=:dot)
    factor,_ = convergence_factor_106!(vu50)
    factor_text = "$(factor_text) vu=$(factor)"
    plot!(iter,vu50[1:iterations],label=L"\rho_{VU} = " * "$(factor); " * L"\kappa^2 \in [0.5,1]", w = 2,color="red")

    yaxis!("relative residual norm", :log10)
    ylims!((10^-10,10^-1))
    xlabel!("iterations")

    savefig("C:/Siga/Repositories/HelmholtzAccCNN.jl/paper/temp/$(title)")
    savefig("C:/Siga/Repositories/HelmholtzAccCNN.jl/paper/temp/$(title).eps")
end

path50 = "C:/Siga/Repositories/HelmholtzAccCNN.jl/graphs/24.06/single/09_40_06 4 50 10 blocks 3 1 t_n=128 t_axb=f t_norm=f preconditioner test.csv"
path25 = "C:/Siga/Repositories/HelmholtzAccCNN.jl/graphs/24.06/single/13_48_25 4 25 10 blocks 3 3 t_n=128 t_axb=f t_norm=f preconditioner test.csv"

n = 128
title = "09_40_06_13_48_25_4_$(n)_3"
df25 = DataFrame(CSV.File(path25))
df50 = DataFrame(CSV.File(path50))
multi_unet_vs_vcycle_graph!("$(title)", df25.VU, df25.V, df25.JU, df50.VU, df50.V, df50.JU)

function retrain_unet_vs_vcycle_graph!(title, vu, v, ju, vu_r, v_r, ju_r; after_vcycle=false, e_vcycle_input=false, len=60, v_factor=0.92)
    iterations = len # length(vu)
    iter = range(1, length=iterations)
    fonts = Dict(:guidefont=>bfont, :xtickfont=>bfont, :ytickfont=>bfont, :legendfont=>sfont)

    factor = convergence_factor_106!(v_r)
    factor_text = "v=$(factor)"
    p = plot(iter,v_r[1:iterations],label=L"\rho_V = " * "$(factor)",legend=:topright, w = 1.8,color="green",size=(500,400); fonts...)
    factor = convergence_factor_106!(ju)
    factor_text = "$(factor_text) ju=$(factor)"
    plot!(iter,ju[1:iterations],label=L"\rho_{JU} = " * "$(factor)", w = 1.8,color="blue")
    factor = convergence_factor_106!(ju_r)
    factor_text = "$(factor_text) ju_r=$(factor)"
    plot!(iter,ju_r[1:iterations],label=L"\rho_{JU} = " * "$(factor);" * " re-train", w = 1.8,color="blue",linestyle=:dot)
    factor = convergence_factor_106!(vu)
    factor_text = "$(factor_text) vu=$(factor)"
    plot!(iter,vu[1:iterations],label=L"\rho_{VU} = " * "$(factor)", w = 1.8,color="red")
    factor  = convergence_factor_106!(vu_r)
    factor_text = "$(factor_text) vu_r=$(factor)"
    plot!(iter,vu_r[1:iterations],label=L"\rho_{VU} = " * "$(factor);" * " re-train", w = 1.8,color="red",linestyle=:dot)

    yaxis!("relative residual norm", :log10)
    ylims!((10^-10,10^-1))
    xlabel!("iterations")
    savefig("C:/Siga/Repositories/HelmholtzAccCNN.jl/paper/temp/$(title)")
    savefig("C:/Siga/Repositories/HelmholtzAccCNN.jl/paper/temp/$(title).eps")
end

original_p = "C:/Siga/Repositories/HelmholtzAccCNN.jl/graphs/24.06/uniform/09_40_06 t_n=128 t_axb=f t_norm=f preconditioner test.csv"
retrain_p = "C:/Siga/Repositories/HelmholtzAccCNN.jl/graphs/24.06/uniform/09_40_06 t_n=128 t_axb=f t_norm=f preconditioner test.csv"

n = 128
axb = false
thresh = 25

title = "09_40_06_vcycle_factors"

# Const Model 0.887 118
original = DataFrame(CSV.File(original_p))
retrain = DataFrame(CSV.File(retrain_p))

retrain_unet_vs_vcycle_graph!("$(title)",original.VU, original.V,original.JU, retrain.VU,retrain.V,retrain.JU;len=200)

if thresh == 25
    if n == 128
        if axb == true
            factor = 0.927 # 182
        else
            factor = 0.916 # 159
        end
    elseif n == 256
        if axb == true
            factor = 0.979 # 637
        else
            factor = 0.972 # 493
        end
    elseif n == 512
        if axb == true
            factor = 0.991 # 1520
        else
            factor = 0.987 # 1017
        end
    end
else
    if n == 128
        if axb == true
            factor = 0.887 # 115
        else
            factor = 0.879 # 108
        end
    elseif n == 256
        if axb == true
            factor = 0.979 # 637
        else
            factor = 0.972 # 493
        end
    end
end

function gmres_unet_vs_vcycle_graph!(title, gmres5vu, gmres5ju, gmres5v, gmres10vu, gmres10ju, gmres10v,gmres20vu, gmres20ju,gmres20v)
    iterations = length(gmres5v)
    iter = range(1, length=iterations)
    fonts = Dict(:guidefont=>bfont, :xtickfont=>bfont, :ytickfont=>bfont, :legendfont=>sfont)

    p = plot(iter,gmres5v,label=L"M_{V}"*"; GMRES(5)",legend=:outerright, w = 1.8,size=(600,400),color="green"; fonts...)
    plot!(iter,gmres10v,label=L"M_{V}"*"; GMRES(10)",color="green",linestyle=:dash, w = 1.8)
    plot!(iter,gmres20v,label=L"M_{V}"*"; GMRES(20)",color="green",linestyle=:dot, w = 1.8)
    plot!(iter,gmres5ju,label=L"M_{JU}"*"; GMRES(5)",color="blue", w = 1.8)
    plot!(iter,gmres10ju,label=L"M_{JU}"*"; GMRES(10)",color="blue",linestyle=:dash, w = 1.8)
    plot!(iter,gmres20ju,label=L"M_{JU}"*"; GMRES(20)",color="blue",linestyle=:dot, w = 1.8)
    plot!(iter,gmres5vu,label=L"M_{VU}"*"; GMRES(5)",color="red", w = 1.8)
    plot!(iter,gmres10vu,label=L"M_{VU}"*"; GMRES(10)",color="red",linestyle=:dash, w = 1.8)
    plot!(iter,gmres20vu,label=L"M_{VU}"*"; GMRES(20)",color="red",linestyle=:dot, w = 1.8)

    yaxis!("relative residual norm", :log10)
    ylims!((10^-10,10^-1))
    xlabel!("iterations")
    savefig("C:/Siga/Repositories/HelmholtzAccCNN.jl/paper/gmres_blocks/$(title)")
    savefig("C:/Siga/Repositories/HelmholtzAccCNN.jl/paper/gmres_blocks/$(title).eps")
end

gmres5_p = "C:/Siga/Repositories/HelmholtzAccCNN.jl/graphs/24.06/gmres/23_48_23_GMRES5 t_n=256 t_axb=t t_norm=f preconditioner test.csv"
gmres10_p = "C:/Siga/Repositories/HelmholtzAccCNN.jl/graphs/24.06/gmres/23_48_23_GMRES10 t_n=256 t_axb=t t_norm=f preconditioner test.csv"
gmres20_p= "C:/Siga/Repositories/HelmholtzAccCNN.jl/graphs/24.06/gmres/23_48_23_GMRES20 t_n=256 t_axb=t t_norm=f preconditioner test.csv"

n = 256
axb = false

title = "23_48_23_GMRES_$(n)"

gmres5 = DataFrame(CSV.File(gmres5_p))
gmres10 = DataFrame(CSV.File(gmres10_p))
gmres20 = DataFrame(CSV.File(gmres20_p))

gmres_unet_vs_vcycle_graph!("$(title)", gmres5.VU, gmres5.JU, gmres5.V, gmres10.VU, gmres10.JU, gmres10.V, gmres20.VU, gmres20.JU, gmres20.V)

function blocks_unet_vs_vcycle_graph!(title, blocks2vu, blocks2ju, blocks2v, blocks5vu, blocks5ju, blocks5v, blocks10vu, blocks10ju, blocks10v, blocks20vu, blocks20ju, blocks20v)
    iterations = length(blocks2vu)
    iter = range(1, length=iterations)
    fonts = Dict(:guidefont=>bfont, :xtickfont=>bfont, :ytickfont=>bfont, :legendfont=>sfont)

    p = plot(iter,blocks2v,label=L"M_{V}"*"; 2 RHS's",legend=:outerright, w = 1.8,color="green",size=(600,400); fonts...)
    plot!(iter,blocks5v,label=L"M_{V}"*"; 5 RHS's",color="green",linestyle=:dash, w = 1.8)
    plot!(iter,blocks10v,label=L"M_{V}"*"; 10 RHS's",color="green",linestyle=:dashdot, w = 1.8)
    plot!(iter,blocks20v,label=L"M_{V}"*"; 20 RHS's",color="green",linestyle=:dot, w = 1.8)
    plot!(iter,blocks2ju,label=L"M_{JU}"*"; 2 RHS's",color="blue", w = 1.8)
    plot!(iter,blocks5ju,label=L"M_{JU}"*"; 5 RHS's",color="blue",linestyle=:dash, w = 1.8)
    plot!(iter,blocks10ju,label=L"M_{JU}"*"; 10 RHS's",color="blue",linestyle=:dashdot, w = 1.8)
    plot!(iter,blocks20ju,label=L"M_{JU}"*"; 20 RHS's",color="blue",linestyle=:dot, w = 1.8)
    plot!(iter,blocks2vu,label=L"M_{VU}"*"; 2 RHS's",color="red", w = 1.8)
    plot!(iter,blocks5vu,label=L"M_{VU}"*"; 5 RHS's",color="red",linestyle=:dash, w = 1.8)
    plot!(iter,blocks10vu,label=L"M_{VU}"*"; 10 RHS's",color="red",linestyle=:dashdot, w = 1.8)
    plot!(iter,blocks20vu,label=L"M_{VU}"*"; 20 RHS's",color="red",linestyle=:dot, w = 1.8)

    yaxis!("relative residual norm", :log10)
    ylims!((10^-10,10^-1))
    xlabel!("iterations")
    savefig("C:/Siga/Repositories/HelmholtzAccCNN.jl/paper/gmres_blocks/$(title)")
    savefig("C:/Siga/Repositories/HelmholtzAccCNN.jl/paper/gmres_blocks/$(title).eps")
end

blocks2_p = "C:/Siga/Repositories/HelmholtzAccCNN.jl/graphs/24.06/gmres/23_48_23_2b t_n=256 t_axb=f t_norm=f preconditioner test.csv"
blocks5_p = "C:/Siga/Repositories/HelmholtzAccCNN.jl/graphs/24.06/gmres/23_48_23_5b t_n=256 t_axb=f t_norm=f preconditioner test.csv"
blocks10_p = "C:/Siga/Repositories/HelmholtzAccCNN.jl/graphs/24.06/gmres/23_48_23_10b t_n=256 t_axb=f t_norm=f preconditioner test.csv"
blocks20_p = "C:/Siga/Repositories/HelmholtzAccCNN.jl/graphs/24.06/gmres/23_48_23_20b t_n=256 t_axb=f t_norm=f preconditioner test.csv"

n = 256
axb = false

title = "23_48_23_blocks_$(n)"

blocks2 = DataFrame(CSV.File(blocks2_p))
blocks5 = DataFrame(CSV.File(blocks5_p))
blocks10 = DataFrame(CSV.File(blocks10_p))
blocks20 = DataFrame(CSV.File(blocks20_p))

blocks_unet_vs_vcycle_graph!("$(title)", blocks2.VU, blocks2.JU, blocks2.V, blocks5.VU, blocks5.JU, blocks5.V, blocks10.VU, blocks10.JU, blocks10.V, blocks20.VU, blocks20.JU, blocks20.V)

function multi_error_vs_residual_graph!(title, evu, rvu, eu, ru, eju, rju, ev, rv; after_vcycle=false, e_vcycle_input=false)
    iterations = length(eu)
    iter = range(1, length=iterations)
    fonts = Dict(:guidefont=>bfont, :xtickfont=>bfont, :ytickfont=>bfont, :legendfont=>bfont)
    # Vcycle
    p = plot(iter,ev[1:iterations],label=L"e_v",legend=:bottomleft,color="green",size=(500,400); fonts...)
    # plot!(iter,rv[1:iterations],label=L"r_v",color="green",linestyle=:dot)
    plot!(iter,eu[1:iterations],label=L"e_u",color="red")
    # plot!(iter,ru[1:iterations],label=L"r_u",color="red",linestyle=:dot)
    plot!(iter,eju[1:iterations],label=L"e_{ju}",color="orange")
    # plot!(iter,rju[1:iterations],label=L"r_{ju}",color="orange",linestyle=:dot)
    plot!(iter,evu[1:iterations],label=L"e_{vu}",color="blue")
    # plot!(iter,rvu[1:iterations],label=L"r_{vu}",color="blue",linestyle=:dot)

    yaxis!(L"\Vert b - Hx \Vert_2", :log10)
    xlabel!("iterations")

    savefig("C:/Siga/Repositories/HelmholtzAccCNN.jl/paper/$(title)")
end

pathu = "C:/Siga/Repositories/HelmholtzAccCNN.jl/graphs/16.06/18_13_51 RADAM ND FFSDNUnet FFKappa SResidualBlock 10 elu 3 5 g=-1 t=Float32 g=t e=f r=f k=1 25 n=128 f=10_0 m=20000 bs=20 lr=0_0001 each=32 i=80 t_n=128 t_axb=f t_norm=f unet error.csv"
pathv = "C:/Siga/Repositories/HelmholtzAccCNN.jl/graphs/16.06/18_13_51 RADAM ND FFSDNUnet FFKappa SResidualBlock 10 elu 3 5 g=-1 t=Float32 g=t e=f r=f k=1 25 n=128 f=10_0 m=20000 bs=20 lr=0_0001 each=32 i=80 t_n=128 t_axb=f t_norm=f vcycle error.csv"
title = "18_13_51 RADAM ND FFSDNUnet FFKappa SResidualBlock 1 25 error"
dfu = DataFrame(CSV.File(pathu))
dfv = DataFrame(CSV.File(pathv))
multi_error_vs_residual_graph!(title, dfu.EA[1:20], dfu.RA[1:20], dfu.EA[21:40], dfu.RA[21:40], dfu.EA[41:60], dfu.RA[41:60], dfv.EA[21:40], dfv.RA[21:40])
