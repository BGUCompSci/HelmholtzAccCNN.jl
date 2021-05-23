using DataFrames, CSV
using Plots
using LaTeXStrings
using BSON: @load
using Dates
pyplot()

include("../test/unet/unet_test_utils.jl")

function unet_vs_vcycle_csv_to_graph!(title, path)
    df = CSV.read(path)
    unet_vs_vcycle_graph!("$(title) preconditioner test", df.VU, df.U, df.VV, df.V, df.JU)
end

unet_vs_vcycle_csv_to_graph!("20_12_55 SDNUnet1 256 0","graphs/csv/20_12_55 SDNUnet1 d=f axb=f g=-1 norm=f to 1 t=Float32 k=3 50 g=t e=f da=f k=0 n=256 f=20_0 m=25000 bs=5 opt=ADAM lr=0_0001 each=50 i=60 t_n=256 t_axb=f t_norm=f t_j=t preconditioner test.csv")

function loss_csv_to_graph!(title, path)
    df = CSV.read(path)

    iter = range(1, length=length(df.Train))
    p = plot(iter, df.Train, label="train loss")
    plot!(iter, df.Test, label="test loss")
    yaxis!(L"\Vert e^{true} -e^{net} \Vert_2", :log10)
    xlabel!("iterations")
    savefig("test/unet/results/$(title) residual graph")
end

loss_csv_to_graph!("08_39_20 GMRES-R 50 2", "graphs/08_39_20 DNUnet j=f axb=f norm=f to 1 t=Float32 k=3 50 g=t e=f da=f k=2 n=128 f=10_0 m=25000 bs=5 opt=ADAM lr=0_0005 each=25 i=150 loss.csv")

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
