using DataFrames, CSV
using Plots
using LaTeXStrings
using BSON: @load
using Dates
pyplot()

# Fonts:

bbfont = Plots.font("Helvetica", 15)
bfont = Plots.font("Helvetica", 13)
mfont = Plots.font("Helvetica", 12)
sfont = Plots.font("Helvetica", 10)
fonts = Dict(:guidefont=>bfont, :xtickfont=>sfont, :ytickfont=>sfont, :legendfont=>sfont)

# Loss graphs:

function loss_graph!(title, path)

    fonts = Dict(:guidefont=>bfont, :xtickfont=>bfont, :ytickfont=>bfont, :legendfont=>bfont)
    df = DataFrame(CSV.File(path))
    iter = range(1, length=length(df.Train))
    p = plot(iter, df.Train, label="train", color="blue",size=(500,400),w=1.5; fonts...)
    plot!(iter, df.Test, label="validation",color="blue",w=1.5,linestyle=:dot)

    ylabel!("loss")
    xlabel!("epochs")
    savefig("../paper/$(title)")
    savefig("../paper/$(title).eps")
end

path = ""
title = ""
loss_graph!(title, path)

function kappa_threshold_loss_graph!(title, l25, l50)
    fonts = Dict(:guidefont=>bfont, :xtickfont=>bfont, :ytickfont=>bfont, :legendfont=>bfont)
    iterations = 125
    iter = range(1, length=iterations)
    p = plot(iter, min.(l25[1:iterations],l25[1]), label=L"\quad \kappa^2 \in [0.25,1]",color="blue",w=1.5,size=(500,400); fonts...)
    plot!(iter, min.(l50[1:iterations],l50[1]), label=L"\quad \kappa^2 \in [0.5,1]",color="red",w=1.5)

    ylabel!("loss")
    xlabel!("epochs")
    savefig("../paper/$(title)")
    savefig("../paper/$(title).eps")
end

path = ""
title = ""
df25 = DataFrame(CSV.File(path))
df50 = DataFrame(CSV.File(path))
kappa_threshold_loss_graph!(title, df25.Train, df50.Train)

function loss_error_residual_to_graph!(title, path)
    fonts = Dict(:guidefont=>bfont, :xtickfont=>bfont, :ytickfont=>bfont, :legendfont=>bfont)
    df = DataFrame(CSV.File(path))
    iter = range(1, length=length(df.Train))
    p = plot(iter, df.Train, label="total loss",size=(500,400); fonts...)
    plot!(iter, df.Residual, label="residual loss")
    plot!(iter, df.Error, label="error loss")

    ylabel!("loss")
    xlabel!("epochs")
    savefig("../paper/$(title) residual graph")
end

path = ""
title = ""
loss_error_residual_to_graph!(title, path)

# Heatmaps:

function csv_to_heatmap!(title, path, n, m)
    df = DataFrame(CSV.File(path))
    heatmap(reshape(df.K, n-1, m-1), color=:jet,size=(200*m/n,200))
    savefig("C:/Siga/Repositories/HelmholtzAccCNN.jl/temp/graphs/$(title)")
end

path = ""
csv_to_heatmap!("kappa_7", path, 256, 512)
csv_to_heatmap!("kappa_6", path, 176, 400)
csv_to_heatmap!("kappa_5", path, 128, 256)

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

    savefig("../paper/$(title)")
    savefig("../paper/$(title).eps")
end

path = ""
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

function kappa_threshold_unet_vs_vcycle_graph!(title, vu25, v25, ju25, vu50, v50, ju50; after_vcycle=false, e_vcycle_input=false)
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

    savefig("../paper/$(title)")
    savefig("../paper/$(title).eps")
end

path50 = ""
n = 128
title = "09_40_06_13_48_25_4_$(n)_3"
df25 = DataFrame(CSV.File(path))
df50 = DataFrame(CSV.File(path))
kappa_threshold_unet_vs_vcycle_graph!("$(title)", df25.VU, df25.V, df25.JU, df50.VU, df50.V, df50.JU)

function retrain_unet_vs_vcycle_graph!(title, vu, v, ju, vu_r, v_r, ju_r; after_vcycle=false, e_vcycle_input=false, len=60, v_factor=0.92)
    iterations = len # length(vu)
    iter = range(1, length=iterations)
    fonts = Dict(:guidefont=>bfont, :xtickfont=>bfont, :ytickfont=>bfont, :legendfont=>sfont)

    factor,_ = convergence_factor_106!(v_r)
    factor_text = "v=$(factor)"
    p = plot(iter,v_r[1:iterations],label=L"\rho_V = " * "$(factor)",legend=:topright, w = 1.8,color="green",size=(500,400); fonts...)
    factor,_ = convergence_factor_106!(ju)
    factor_text = "$(factor_text) ju=$(factor)"
    plot!(iter,ju[1:iterations],label=L"\rho_{JU} = " * "$(factor)", w = 1.8,color="blue")
    factor,_ = convergence_factor_106!(ju_r)
    factor_text = "$(factor_text) ju_r=$(factor)"
    plot!(iter,ju_r[1:iterations],label=L"\rho_{JU} = " * "$(factor);" * " re-train", w = 1.8,color="blue",linestyle=:dot)
    factor,_ = convergence_factor_106!(vu)
    factor_text = "$(factor_text) vu=$(factor)"
    plot!(iter,vu[1:iterations],label=L"\rho_{VU} = " * "$(factor)", w = 1.8,color="red")
    factor,_  = convergence_factor_106!(vu_r)
    factor_text = "$(factor_text) vu_r=$(factor)"
    plot!(iter,vu_r[1:iterations],label=L"\rho_{VU} = " * "$(factor);" * " re-train", w = 1.8,color="red",linestyle=:dot)

    yaxis!("relative residual norm", :log10)
    ylims!((10^-10,10^-1))
    xlabel!("iterations")
    savefig("../paper/$(title)")
    savefig("../paper/$(title).eps")
end

n = 128
path = ""
original = DataFrame(CSV.File(path))
retrain = DataFrame(CSV.File(path))
title = "23_48_23_32_bit_cpu_m10r10_128"
retrain_unet_vs_vcycle_graph!("$(title)",original.VU, original.V,original.JU, original.VU_T, original.V_T,original.JU_T;len=100)

function retrain_unet_vs_vcycle_graph_times!(title, vu, v, ju, vu_r, v_r, ju_r, vu_t, v_t, ju_t, vu_r_t, v_r_t, ju_r_t; xlim=20, sec=1, blocks=10)
    iterations = length(vu)
    iter = range(1, length=iterations)
    fonts = Dict(:guidefont=>bfont, :xtickfont=>bfont, :ytickfont=>bfont, :legendfont=>sfont)

    factor,T = convergence_factor_106!(v_r)
    if T == 1001
        T = 1018
    end
    factor_text = "v=$(factor)"
    p = plot(v_r_t[1:iterations]./(blocks*sec),v_r[1:iterations],label=L"\rho_V = " * "$(factor);" * L" T = " * "$(T)",legend=:topright, w = 1.8,color="green",size=(500,400); fonts...)
    factor,T = convergence_factor_106!(ju)
    factor_text = "$(factor_text) ju=$(factor)"
    plot!(ju_t[1:iterations]./(blocks*sec),ju[1:iterations],label=L"\rho_{JU} = " * "$(factor);" * L" T = " * "$(T)", w = 1.8,color="blue")
    factor,T = convergence_factor_106!(ju_r)
    factor_text = "$(factor_text) ju_r=$(factor)"
    plot!(ju_r_t[1:iterations]./(blocks*sec),ju_r[1:iterations],label=L"\rho_{JU} = " * "$(factor);" * L" T = " * "$(T);"* " re-train", w = 1.8,color="blue",linestyle=:dot)
    factor,T= convergence_factor_106!(vu)
    factor_text = "$(factor_text) vu=$(factor)"
    plot!(vu_t[1:iterations]./(blocks*sec),vu[1:iterations],label=L"\rho_{VU} = " * "$(factor);" * L" T = " * "$(T)", w = 1.8,color="red")
    factor,T  = convergence_factor_106!(vu_r)
    factor_text = "$(factor_text) vu_r=$(factor)"
    plot!(vu_r_t[1:iterations]./(blocks*sec),vu_r[1:iterations],label=L"\rho_{VU} = " * "$(factor);" * L" T = " * "$(T);" * " re-train", w = 1.8,color="red",linestyle=:dot)

    yaxis!("relative residual norm", :log10)
    ylims!((10^-7,10^-1))
    xlims!((-(xlim ./ 90),xlim))
    if (sec == 1000)
        xlabel!("seconds per RHS")
    else
        xlabel!("milliseconds per RHS")
    end
    savefig("../graphs/times/$(title)")
    savefig("../graphs/times/$(title).eps")
end

function retrain_unet_vs_vcycle_graph_times_details!(title, vu, v, ju, vu_r, v_r, ju_r, vu_t, v_t, ju_t, vu_r_t, v_r_t, ju_r_t; xlim=20, sec=1, blocks=10)
    iterations = length(vu)
    iter = range(1, length=iterations)
    fonts = Dict(:guidefont=>bfont, :xtickfont=>bfont, :ytickfont=>bfont, :legendfont=>sfont)

    factor,T = convergence_factor_106!(v_r)
    t = T
    if T == 1001
        T = 1018
        t = 1000
    end
    factor_text = "v=$(factor)"
    p = plot(v_r_t[1:iterations]./(blocks*sec),v_r[1:iterations],label=L"\rho_V = " * "$(factor);" * L" T = " * "$(T); $(v_r_t[t]./(blocks*sec))",legend=:topright, w = 1.8,color="green",size=(500,400); fonts...)
    factor,T = convergence_factor_106!(ju)
    factor_text = "$(factor_text) ju=$(factor)"
    plot!(ju_t[1:iterations]./(blocks*sec),ju[1:iterations],label=L"\rho_{JU} = " * "$(factor);" * L" T = " * "$(T); $(ju_t[T]./(blocks*sec))", w = 1.8,color="blue")
    factor,T = convergence_factor_106!(ju_r)
    factor_text = "$(factor_text) ju_r=$(factor)"
    plot!(ju_r_t[1:iterations]./(blocks*sec),ju_r[1:iterations],label=L"\rho_{JU} = " * "$(factor);" * L" T = " * "$(T); $(ju_r_t[T]./(blocks*sec));"* " re-train", w = 1.8,color="blue",linestyle=:dot)
    factor,T= convergence_factor_106!(vu)
    factor_text = "$(factor_text) vu=$(factor)"
    plot!(vu_t[1:iterations]./(blocks*sec),vu[1:iterations],label=L"\rho_{VU} = " * "$(factor);" * L" T = " * "$(T); $(vu_t[T]./(blocks*sec))", w = 1.8,color="red")
    factor,T  = convergence_factor_106!(vu_r)
    factor_text = "$(factor_text) vu_r=$(factor)"
    plot!(vu_r_t[1:iterations]./(blocks*sec),vu_r[1:iterations],label=L"\rho_{VU} = " * "$(factor);" * L" T = " * "$(T); $(vu_r_t[T]./(blocks*sec));" * " re-train", w = 1.8,color="red",linestyle=:dot)

    yaxis!("relative residual norm", :log10)
    ylims!((10^-7,10^-1))
    xlims!((-(xlim ./ 90),xlim))
    if (sec == 1000)
        xlabel!("seconds per RHS")
    else
        xlabel!("milliseconds per RHS")
    end
    savefig("../graphs/$(title)")
    savefig("../graphs/$(title).eps")
end

function unet_vs_vcycle_graph_iterations!(title, vu, v, ju; after_vcycle=false, e_vcycle_input=false, len=60, v_factor=0.92)
    iterations = len # length(vu)
    iter = range(1, length=iterations)
    fonts = Dict(:guidefont=>bfont, :xtickfont=>bfont, :ytickfont=>bfont, :legendfont=>sfont)

    factor,_ = convergence_factor_106!(v)
    factor_text = "v=$(factor)"
    p = plot(iter,v[1:iterations],label=L"\rho_V = " * "$(factor)",legend=:topright, w = 1.8,color="green",size=(500,400); fonts...)
    factor,_ = convergence_factor_106!(ju)
    factor_text = "$(factor_text) ju=$(factor)"
    plot!(iter,ju[1:iterations],label=L"\rho_{JU} = " * "$(factor)", w = 1.8,color="blue")
    factor,_ = convergence_factor_106!(vu)
    factor_text = "$(factor_text) vu=$(factor)"
    plot!(iter,vu[1:iterations],label=L"\rho_{VU} = " * "$(factor)", w = 1.8,color="red")

    yaxis!("relative residual norm", :log10)
    ylims!((10^-7,10^-1))
    xlabel!("iterations")
    savefig("../graphs/$(title)")
end

function retrain_unet_vs_vcycle_graph_iterations!(title, vu, v, ju, vu_r, v_r, ju_r; xlim=100)
    iterations = length(vu)
    iter = range(1, length=iterations)
    fonts = Dict(:guidefont=>bfont, :xtickfont=>bfont, :ytickfont=>bfont, :legendfont=>sfont)

    factor,_ = convergence_factor_106!(v_r)
    factor_text = "v=$(factor)"
    p = plot(iter,v_r[1:iterations],label=L"\rho_V = " * "$(factor)",legend=:topright, w = 1.8,color="green",size=(500,400); fonts...)
    factor,_ = convergence_factor_106!(ju)
    factor_text = "$(factor_text) ju=$(factor)"
    plot!(iter,ju[1:iterations],label=L"\rho_{JU} = " * "$(factor)", w = 1.8,color="blue")
    factor,_ = convergence_factor_106!(ju_r)
    factor_text = "$(factor_text) ju_r=$(factor)"
    plot!(iter,ju_r[1:iterations],label=L"\rho_{JU} = " * "$(factor);" * " re-train", w = 1.8,color="blue",linestyle=:dot)
    factor,_ = convergence_factor_106!(vu)
    factor_text = "$(factor_text) vu=$(factor)"
    plot!(iter,vu[1:iterations],label=L"\rho_{VU} = " * "$(factor)", w = 1.8,color="red")
    factor,_  = convergence_factor_106!(vu_r)
    factor_text = "$(factor_text) vu_r=$(factor)"
    plot!(iter,vu_r[1:iterations],label=L"\rho_{VU} = " * "$(factor);" * " re-train", w = 1.8,color="red",linestyle=:dot)

    yaxis!("relative residual norm", :log10)
    ylims!((10^-10,10^-1))
    xlims!((-(xlim ./ 90),xlim))
    xlabel!("iterations")
    savefig("../graphs/$(title)")
    savefig("../graphs/$(title).eps")
end

n = 512
blocks = 10
path = ""
df = DataFrame(CSV.File(path))
df_r= DataFrame(CSV.File(path))

title = "23_48_23_64_cpu_block10_model4_$(n)_3090"
retrain_unet_vs_vcycle_graph_iterations!(title, df.VU, df.V, df.JU, df_r.VU, df_r.V, df_r.JU; xlim=400)

# df = (df1.+df2.+df3.+df4.+df5.+df6.+df7.+df8.+df9.+df10)./10
# df_r = (df_r1.+df_r2.+df_r3.+df_r4.+df_r5.+df_r6.+df_r7.+df_r8.+df_r9.+df_r10)./10
title = "23_48_23_32_gpu_model4_$(n)_b$(blocks)_retrain_times_mean10_details"
retrain_unet_vs_vcycle_graph_times!(title, df.VU, df.V, df.JU, df_r.VU, df_r.V, df_r.JU, df.VU_T, df.V_T, df.JU_T, df_r.VU_T, df_r.V_T, df_r.JU_T; xlim=1000, sec=1, blocks=blocks)
retrain_unet_vs_vcycle_graph_times_details!(title, df.VU, df.V, df.JU, df_r.VU, df_r.V, df_r.JU, df.VU_T, df.V_T, df.JU_T, df_r.VU_T, df_r.V_T, df_r.JU_T; xlim=8, sec=1000, blocks=blocks)

blues = palette(:Blues_5)
greens = palette(:Greens_5)
reds = palette(:Reds_5)
function gpu_cpu_32_64_bit_unet_vs_vcycle_graph_times!(title, vu32c, v32c, ju32c, vu32g, v32g, ju32g, vu64c, v64c, ju64c,
    vu32c_t, v32c_t, ju32c_t, vu32g_t, v32g_t, ju32g_t, vu64c_t, v64c_t, ju64c_t; after_vcycle=false, e_vcycle_input=false, len=60, v_factor=0.92)
    iterations = len # length(vu)
    iter = range(1, length=iterations)
    fonts = Dict(:guidefont=>bfont, :xtickfont=>bfont, :ytickfont=>bfont, :legendfont=>sfont)

    p = plot(v64c_t[1:iterations]./1000,v64c[1:iterations],label=L"\rho_V"*" 64 BIT CPU",legend=:topright, w = 1,color=greens[2],size=(800,400), marker=(:circle,5); fonts...)
    plot!(v32c_t[1:iterations]./1000,v32c[1:iterations],label=L"\rho_V"*" 32 BIT CPU",legend=:topright, w = 1.8,color=greens[3])
    plot!(v32g_t[1:iterations]./1000,v32g[1:iterations],label=L"\rho_V"*" 32 BIT GPU",legend=:topright, w = 1.8,color=greens[4])
    # plot!(v64g_t[1:iterations]./1000,v64g[1:iterations],label=L"\rho_V"*" 64 BIT GPU",legend=:topright, w = 1.8,color=greens[5])
    plot!(ju64c_t[1:iterations]./1000,ju64c[1:iterations],label=L"\rho_{JU}"*" 64 BIT CPU",legend=:topright, w = 1.8,color=blues[2])
    plot!(ju32c_t[1:iterations]./1000,ju32c[1:iterations],label=L"\rho_{JU}"*" 32 BIT CPU",legend=:topright, w = 1.8,color=blues[3])
    plot!(ju32g_t[1:iterations]./1000,ju32g[1:iterations],label=L"\rho_{JU}"*" 32 BIT GPU",legend=:topright, w = 1.8,color=blues[4])
    # plot!(ju64g_t[1:iterations]./1000,ju64g[1:iterations],label=L"\rho_{JU}"*" 64 BIT GPU",legend=:topright, w = 1.8,color=blues[5])
    plot!(vu64c_t[1:iterations]./1000,vu64c[1:iterations],label=L"\rho_{VU}"*" 64 BIT CPU",legend=:topright, w = 1.8,color=reds[2])
    plot!(vu32c_t[1:iterations]./1000,vu32c[1:iterations],label=L"\rho_{VU}"*" 32 BIT CPU",legend=:topright, w = 1.8,color=reds[3])
    plot!(vu32g_t[1:iterations]./1000,vu32g[1:iterations],label=L"\rho_{VU}"*" 32 BIT GPU",legend=:topright, w = 1.8,color=reds[4])
    # plot!(vu64g_t[1:iterations]./1000,vu64g[1:iterations],label=L"\rho_{VU}"*" 64 BIT GPU",legend=:topright, w = 1.8,color=reds[5])
    yaxis!("relative residual norm", :log10)
    ylims!((10^-7,10^-1))
    xlabel!("seconds")
    xlabel!("iterations")
    savefig("../graphs/$(title)")
end

function gpu_cpu_32_64_bit_unet_vs_vcycle_graph_iterations!(title, vu32c, v32c, ju32c, vu32g, v32g, ju32g, vu64c, v64c, ju64c, vu64g, v64g, ju64g; after_vcycle=false, e_vcycle_input=false, len=60, v_factor=0.92)
    iterations = len # length(vu)
    iter = range(1, length=iterations)
    fonts = Dict(:guidefont=>bfont, :xtickfont=>bfont, :ytickfont=>bfont, :legendfont=>sfont)

    p = plot(iter,v32c[1:iterations],label=L"\rho_V"*" 32 BIT CPU",legend=:topright, w = 1.8,color=greens[2],size=(700,700); fonts...)
    plot!(iter,v32g[1:iterations],label=L"\rho_V"*" 32 BIT GPU",legend=:topright, w = 1.8,color=greens[3])
    plot!(iter,v64c[1:iterations],label=L"\rho_V"*" 64 BIT CPU",legend=:topright, w = 1.8,color=greens[4])
    # plot!(iter,v64g[1:iterations],label=L"\rho_V"*" 64 BIT GPU",legend=:topright, w = 1.8,color=greens[5])
    plot!(iter,ju32c[1:iterations],label=L"\rho_{JU}"*" 32 BIT CPU",legend=:topright, w = 1.8,color=blues[2])
    plot!(iter,ju32g[1:iterations],label=L"\rho_{JU}"*" 32 BIT GPU",legend=:topright, w = 1.8,color=blues[3])
    plot!(iter,ju64c[1:iterations],label=L"\rho_{JU}"*" 64 BIT CPU",legend=:topright, w = 1.8,color=blues[4])
    # plot!(iter,ju64g[1:iterations],label=L"\rho_{JU}"*" 64 BIT GPU",legend=:topright, w = 1.8,color=blues[5])
    plot!(iter,vu32c[1:iterations],label=L"\rho_{VU}"*" 32 BIT CPU",legend=:topright, w = 1.8,color=reds[2])
    plot!(iter,vu32g[1:iterations],label=L"\rho_{VU}"*" 32 BIT GPU",legend=:topright, w = 1.8,color=reds[3])
    plot!(iter,vu64c[1:iterations],label=L"\rho_{VU}"*" 64 BIT CPU",legend=:topright, w = 1.8,color=reds[4])
    # plot!(iter,vu64g[1:iterations],label=L"\rho_{VU}"*" 64 BIT GPU",legend=:topright, w = 1.8,color=reds[5])
    yaxis!("relative residual norm", :log10)
    xlabel!("iterations")
    savefig("../graphs/$(title)")
end

title = "23_48_23_32_64_cpu_gpu_iterations"

path = ""
c32_df = DataFrame(CSV.File(path))
g32_df = DataFrame(CSV.File(path))
c64_df = DataFrame(CSV.File(path))
g64_df = DataFrame(CSV.File(path))

gpu_cpu_32_64_bit_unet_vs_vcycle_graph_iterations!("$(title)",c32_df.VU, c32_df.V,c32_df.JU,g32_df.VU, g32_df.V,g32_df.JU,c64_df.VU, c64_df.V,c64_df.JU,g64_df.VU, g64_df.V,g64_df.JU;len=400)
title = "23_48_23_32_64_cpu_gpu_times"
iter = 800

c32_df = DataFrame(CSV.File(path))
g32_df = DataFrame(CSV.File(path))
c64_df = DataFrame(CSV.File(path))

gpu_cpu_32_64_bit_unet_vs_vcycle_graph_times!("$(title)",c32_df.VU, c32_df.V,c32_df.JU,g32_df.VU, g32_df.V,g32_df.JU,c64_df.VU, c64_df.V,c64_df.JU,
c32_df.VU_T, c32_df.V_T,c32_df.JU_T,g32_df.VU_T, g32_df.V_T,g32_df.JU_T,c64_df.VU_T, c64_df.V_T,c64_df.JU_T;len=iter)

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
    savefig("../graphs/$(title)")
    savefig("../graphs/$(title).eps")
end

n = 256
title = "23_48_23_10blocks_GMRES_$(n)"

path = ""
gmres5 = DataFrame(CSV.File(path))
gmres10 = DataFrame(CSV.File(path))
gmres20 = DataFrame(CSV.File(path))

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
    savefig("../graphs/$(title)")
    savefig("../graphs/$(title).eps")
end

n = 256
title = "23_48_23_blocks_$(n)"

path = ""
blocks2 = DataFrame(CSV.File(path))
blocks5 = DataFrame(CSV.File(path))
blocks10 = DataFrame(CSV.File(path))
blocks20 = DataFrame(CSV.File(path))

blocks_unet_vs_vcycle_graph!("$(title)", blocks2.VU, blocks2.JU, blocks2.V, blocks5.VU, blocks5.JU, blocks5.V, blocks10.VU, blocks10.JU, blocks10.V, blocks20.VU, blocks20.JU, blocks20.V)
