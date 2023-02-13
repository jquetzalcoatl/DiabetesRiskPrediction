using Flux, Images #, Flux.Data.MNIST
using Flux, Plots, Statistics
using Flux: @epochs
using BSON: @load, @save
using Dates
using BSON

# cd(dirname(pwd()))

include(pwd() * "/Diabetes/ML/parseTimeSeries.jl")
include(pwd() * "/Diabetes/ML/plots.jl")
include(pwd() * "/Diabetes/statistics/patientDayStruc.jl")
include(pwd() * "/Diabetes/preproc/dataParsing.jl")

invBGriskFunc(x) = exp((x / γ + β)^(1/α))


function getTimeSeries(; lim = 10, MAXLENGTH = 10000, bg=true, idx=75)
    lim = lim == 0 ? size(collect(keys(PTS)),1) : lim
    datA = Array{Array{Float32}}(undef, lim)
    for i in 1:lim        
        pt_id = collect(keys(PTS))[idx+i] 
        df = loadCSV(pt_id, linear_interpolation=true)
        test = createPerDayStruc(df, pt_id)
        dat = bg ? vcat([test[pt_id][i].BGrisk / √10 for i in 1:size(test[pt_id],1)]...) : vcat([test[pt_id][i].data[:,2] for i in 1:size(test[pt_id],1)]...)
        datA[i] = size(dat,1) > MAXLENGTH ? dat[1:MAXLENGTH] : dat
    end
    datA
end



function loadData(;BS=20, init=16, dt=3, idx=75, lim=2)
    datA = getTimeSeries(bg=true, lim=lim, idx=idx)
    parseX, parseY = [], []
    for i in 1:size(datA,1)
        pX, pY = gen4CNN(datA[i], init=init, dt=dt)
        parseX = vcat(parseX, pX)
        parseY = vcat(parseY, pY)
    end

    X = reshape(Float32.(float.(hcat(vec.(parseX)...))), 4, 4, 1, :)
    Y = reshape(parseY,1,:)

    dataA = [(X[:,:,:,i], Y[:,i]) for i in Iterators.partition(1:size(X, 4),BS)];

    dataA, datA
end

function loadData4Test(datA;init=16, dt=3)
    parseX, parseY = gen4CNN(datA, init=init, dt=dt)

    X = reshape(Float32.(float.(hcat(vec.(parseX)...))), 4, 4, 1, :)
    Y = reshape(parseY,1,:)


    dataA = (X, Y);

    dataA, datA
end

struct CNN
    NN1
    # NN2
    # NN3
end

function gen_NN(args)
    lrelu(x) = leakyrelu(x, 0.02) |> gpu
    NN1 = Chain(Conv((2,2), 1 => 4, tanh; stride = 1, pad = 0),
        # x->leakyrelu.(x, 0.02),
        # Dropout(args.Drop1, dims = :),
        # BatchNorm(4),
        Conv((2,2), 4 => 8, tanh; stride = 1, pad = 0),
        # x->leakyrelu.(x, 0.02),
        # MeanPool((2,2); pad = 0, stride = 2),

        # BatchNorm(8),
        Conv((2,2), 8 => 16, tanh; stride = 1, pad = 0),
        # x->leakyrelu.(x, 0.02),
        # MeanPool((2,2); pad = 0, stride = 2),
        # BatchNorm(16),
        x-> reshape(x, :, size(x, 4)),

        Dense(16,1, tanh)
    ) |> gpu

    return CNN(NN1)
end

# nn = gen_NN(1)
# nn(rand(4,4,1,2))

function (m::CNN)(x)
    x1 = m.NN1(x)
    x1
end



function my_custom_train!(loss, ps, dataA, opt, nn)
  ps = Flux.Params(ps)
  lost_list = []
#   plt=scatter(legend=false)
#   plt = plot!(loss.(data))
  ps = Flux.Params(ps)
  for (x,y) in dataA
    x = x |>gpu
    y = y |>gpu
    gs = gradient(() -> loss(x,y, nn), ps)
    Flux.update!(opt, ps, gs)
    append!(lost_list, loss(x,y, nn))
    # println(lost_list)
#     plt=scatter!(lost_list)
  end
#   plt=hline!([0])
  return mean(lost_list)#plt
end

function train(dataA; epochs=10)
    nn = gen_NN(1)
    loss(x,y, nn) = (ŷ = nn(x); Flux.mse(y,ŷ))
    # loss(x,y, nn) = (ŷ = nn(x); Flux.mae(y,ŷ))
    opt = ADAM(0.0001, (0.9, 0.8))
    ps = params(nn.NN1)

    # dataA = loadData(;BS=M, init=16, dt=3)

    #Test
    @info "Testing functions..."
    loss(dataA[1][1]|>gpu, dataA[1][2]|>gpu, nn)
    gs = gradient(() -> loss(dataA[1][1]|>gpu, dataA[1][2]|>gpu, nn), ps)
    Flux.update!(opt, ps, gs)
    @info "Testing grads"
    my_custom_train!(loss, ps, dataA, opt, nn)
    @info "Done testing"
    plt = zeros(epochs)
    for ep = 1:epochs
        @info "Epoch $ep"
        plt[ep] = my_custom_train!(loss, ps, dataA, opt, nn)
        @info plt[ep]
        if ep > 1
            @info plt[ep] - plt[ep-1]
        end
        plotCNN(plt)

    end

    plt, nn
end

function trainSamples(;dt=3, path = "/Diabetes/models/15/CNN/")
    BS = 100
    dataA, _ = loadData(;BS=BS, init=16, dt=dt, lim=10)
    for i in 12:15
        @info "Model $i"

        plt, nn = train(dataA, epochs=200)

        savePath = pwd() * path * string(i)
        modelPath = savePath * "/model.bson"
        isdir(savePath) || mkpath(savePath)
        @save modelPath nn

        isdir(savePath * "/Plots") || mkdir(savePath * "/Plots")

        savefig(plot(plt), savePath * "/Plots/$i")

    end
end

BS = 100
dataA, _ = loadData(;BS=BS, init=16, dt=3, lim=1)
_, nn = train(dataA, epochs=50)


# datA = getTimeSeries(bg=true, lim=10, idx=0)
dataA_test, datA_test = loadData4Test(datA[2]; init=16, dt=3)
plot(vcat(nn(dataA_test[1][:,:,:,1:end])...))
plot!(dataA_test[2][1,1:end])

plot(invBGriskFunc.(dataA_test[2][1,1:end] .* √10))
plot!( invBGriskFunc.(vcat(nn(dataA_test[1][:,:,:,1:end])...) .* √10) )


trainSamples(dt=3, path = "/Diabetes/models/15/CNN/")
trainSamples(dt=6, path = "/Diabetes/models/30/CNN/")
trainSamples(dt=12, path = "/Diabetes/models/60/CNN/")




using StatsPlots


function getError(datA; path = "/Diabetes/models/15/CNN/", pred=3, MAX=10, idx=0)
    dataA_test, _ = loadData4Test(datA[1]; init=16, dt=pred)

    errorDict = Dict()
    errorDict["ModelName"] = path

    errorDict["model"] = [sqrt(mean(abs2, dataA_test[2][1,1:end] .- vcat(nn(dataA_test[1][:,:,:,1:end])...) ))]

    errorDict["modelCGM"] = [sqrt(mean(abs2, invBGriskFunc.(dataA_test[2][1,1:end] .* √10) .- invBGriskFunc.(vcat(nn(dataA_test[1][:,:,:,1:end])...) .* √10)))]

    errorDict["modelBGrisk"] = [sqrt(mean(abs2, (dataA_test[2][1,1:end] .- vcat(nn(dataA_test[1][:,:,:,1:end])...)) .* ((dataA_test[2][1,1:end] .* √10) .^ 2 .* 10)))]

    errorDict["modelBGriskCGM"] = [sqrt(mean(abs2, (invBGriskFunc.(dataA_test[2][1,1:end] .* √10) .- invBGriskFunc.(vcat(nn(dataA_test[1][:,:,:,1:end])...) .* √10)) .* ((dataA_test[2][1,1:end] .* √10) .^ 2 .* 10)))]


    for i in 2:MAX
        if i ∉ collect(76:86) 
            @info i
            dataA_test, _ = loadData4Test(datA[i]; init=16, dt=pred)

            append!(errorDict["model"],[sqrt(mean(abs2, dataA_test[2][1,1:end] .- vcat(nn(dataA_test[1][:,:,:,1:end])...) ))])

            append!(errorDict["modelCGM"], [sqrt(mean(abs2, invBGriskFunc.(dataA_test[2][1,1:end] .* √10) .- invBGriskFunc.(vcat(nn(dataA_test[1][:,:,:,1:end])...) .* √10)))])

            append!(errorDict["modelBGrisk"], [sqrt(mean(abs2, (dataA_test[2][1,1:end] .- vcat(nn(dataA_test[1][:,:,:,1:end])...)) .* ((dataA_test[2][1,1:end] .* √10) .^ 2 .* 10)))])

            append!(errorDict["modelBGriskCGM"], [sqrt(mean(abs2, (invBGriskFunc.(dataA_test[2][1,1:end] .* √10) .- invBGriskFunc.(vcat(nn(dataA_test[1][:,:,:,1:end])...) .* √10)) .* ((dataA_test[2][1,1:end] .* √10) .^ 2 .* 10)))])

        end
    end
    save(pwd() * path * "$idx/error-4.jld", "data", errorDict)
    errorDict
end


datA = getTimeSeries(bg=true, lim=0, idx=0)
for i in 5:10
    BSON.@load pwd() * "/Diabetes/models/15/CNN/$i/model.bson" nn
    errorDict = getError(datA, path = "/Diabetes/models/15/CNN/", pred = 3, MAX = 139, idx=i)
end


#Error Grid
BSON.@load pwd() * "/Diabetes/models/15/CNN10/1/model.bson" nn



begin
    dataA_test, _ = loadData4Test(datA[1]; init=16, dt=3)
    yhat = nn(dataA_test[1])
    f = plot(invBGriskFunc.(dataA_test[2][1,:] .* √10), invBGriskFunc.(dataA_test[1][end,end,1,:] .* √10), frame=:box, ms=1.8, s=:auto, markershapes = :circle, lw=0, markerstrokewidth=0, xlabel="CGM Ground Truth", ylabel="CGM Prediction", size=(500,500), label="CNN10")
    f = plot!(invBGriskFunc.(dataA_test[2][1,:] .* √10), invBGriskFunc.(yhat[1,:] .* √10), alpha=0.5, frame=:box, ms=1.8, s=:auto, markershapes = :circle, lw=0, markerstrokewidth=0, xlabel="CGM Ground Truth", ylabel="CGM Prediction", size=(500,500), label="LM")
    # plot!(39:400, x->x, label="y=x", lw=2)
    plot!(f,lims=(0,400), legend=:bottomright)
end


begin
    #Plot zone lines
    plot!([0,400], [0,400], c=:black, label=:none)                      #Theoretical 45 regression line
    plot!([0, 175/3], [70, 70], c=:black, label=:none)
    #plot!([175/3, 320], [70, 400], '-', c=:black)
    plot!([175/3, 400/1.2], [70, 400], c=:black, label=:none)           #Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
    plot!([70, 70], [84, 400], c=:black, label=:none)
    plot!([0, 70], [180, 180], c=:black, label=:none)
    plot!([70, 290],[180, 400], c=:black, label=:none)
    # plot!([70, 70], [0, 175/3], c=:black)
    plot!([70, 70], [0, 56], c=:black, label=:none)                     #Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
    # plot!([70, 400],[175/3, 320], c=:black)
    plot!([70, 400], [56, 320], c=:black, label=:none)
    plot!([180, 180], [0, 70], c=:black, label=:none)
    plot!([180, 400], [70, 70], c=:black, label=:none)
    plot!([240, 240], [70, 180], c=:black, label=:none)
    plot!([240, 400], [180, 180], c=:black, label=:none)
    f = plot!([130, 180], [0, 70], c=:black, label=:none)

    plot!(f, ann=[(30,15,text("A", 15,:black, font))])
    plot!(f, ann=[(370, 260,text("B", 15,:black, font))])
    plot!(f, ann=[(280, 370,text("B", 15,:black, font))])
    plot!(f, ann=[(160, 370,text("C", 15,:black, font))])
    plot!(f, ann=[(160, 15,text("C", 15,:black, font))])
    plot!(f, ann=[(30, 140,text("D", 15,:black, font))])
    plot!(f, ann=[(370, 120,text("D", 15,:black, font))])
    plot!(f, ann=[(30, 370,text("E", 15,:black, font))])
    plot!(f, ann=[(270, 15,text("E", 15,:black, font))])
end
savefig(f, "/Users/javier/Library/CloudStorage/Dropbox/Aplicaciones/Overleaf/GlucosePrediction/Figs/ClarkeGridCNN10-15min-2.png")

#Statistics from the data
ref_values, pred_values = invBGriskFunc.(dataA_test[2][1,:] .* √10), invBGriskFunc.(yhat[1,:] .* √10)
zone = zeros(5)
for i in 1:size(yhat,2)
    if (ref_values[i] <= 70 && pred_values[i] <= 70) || (pred_values[i] <= 1.2*ref_values[i] && pred_values[i] >= 0.8*ref_values[i])
        zone[1] = zone[1] + 1    #Zone A

    elseif (ref_values[i] >= 180 && pred_values[i] <= 70) || (ref_values[i] <= 70 && pred_values[i] >= 180)
        zone[5] = zone[5] + 1    #Zone E

    elseif ((ref_values[i] >= 70 && ref_values[i] <= 290) && pred_values[i] >= ref_values[i] + 110) || ((ref_values[i] >= 130 && ref_values[i] <= 180) && (pred_values[i] <= (7/5)*ref_values[i] - 182))
        zone[3] = zone[3] + 1    #Zone C
    elseif (ref_values[i] >= 240 && (pred_values[i] >= 70 && pred_values[i] <= 180)) || (ref_values[i] <= 175/3 && pred_values[i] <= 180 && pred_values[i] >= 70) || ((ref_values[i] >= 175/3 && ref_values[i] <= 70) && pred_values[i] >= (6/5)*ref_values[i])
        zone[4] = zone[4] + 1    #Zone D
    else
        zone[2] = zone[2] + 1    #Zone B
    end
end
fractionCNN10_15 = zone ./ sum(zone)


ref_values, pred_values = invBGriskFunc.(dataA_test[2][1,:] .* √10), invBGriskFunc.(dataA_test[1][end,end,1,:] .* √10)
zone = zeros(5)
for i in 1:size(yhat,2)
    if (ref_values[i] <= 70 && pred_values[i] <= 70) || (pred_values[i] <= 1.2*ref_values[i] && pred_values[i] >= 0.8*ref_values[i])
        zone[1] = zone[1] + 1    #Zone A

    elseif (ref_values[i] >= 180 && pred_values[i] <= 70) || (ref_values[i] <= 70 && pred_values[i] >= 180)
        zone[5] = zone[5] + 1    #Zone E

    elseif ((ref_values[i] >= 70 && ref_values[i] <= 290) && pred_values[i] >= ref_values[i] + 110) || ((ref_values[i] >= 130 && ref_values[i] <= 180) && (pred_values[i] <= (7/5)*ref_values[i] - 182))
        zone[3] = zone[3] + 1    #Zone C
    elseif (ref_values[i] >= 240 && (pred_values[i] >= 70 && pred_values[i] <= 180)) || (ref_values[i] <= 175/3 && pred_values[i] <= 180 && pred_values[i] >= 70) || ((ref_values[i] >= 175/3 && ref_values[i] <= 70) && pred_values[i] >= (6/5)*ref_values[i])
        zone[4] = zone[4] + 1    #Zone D
    else
        zone[2] = zone[2] + 1    #Zone B
    end
end
fractionLM_15 = zone ./ sum(zone)

f = plot(fractionCNN10_60 ./ fractionLM_60, frame=:box, ms=6.8, s=:auto, 
    markershapes = :circle, lw=0, markerstrokewidth=0, c=:blue, 
    xlabel="Clarke Error Grid Zones", ylabel="Ratio Between CNN10 and LM", 
    size=(500,500), label="60-min", xticks=(1:5, ["A", "B", "C", "D", "E"]), legend=:topleft)
f = plot!(fractionCNN10_30 ./ fractionLM_30,  ms=6.8, s=:auto, 
    markershapes = :square, lw=0, markerstrokewidth=0, label="30-min", c=:red)
f = plot!(fractionCNN10_15 ./ fractionLM_15,  ms=6.8, s=:auto, 
    markershapes = :star, lw=0, markerstrokewidth=0, label="15-min", c=:purple)
savefig(f, "/Users/javier/Library/CloudStorage/Dropbox/Aplicaciones/Overleaf/GlucosePrediction/Figs/ClarkeGridratio.png")
fractionLM_15


sum(fractionCNN10_60[3:5]) / sum(fractionLM_60[3:5])

sum(fractionCNN10_30[3:5]) / sum(fractionLM_30[3:5])

sum(fractionCNN10_15[3:5]) / sum(fractionLM_15[3:5])

