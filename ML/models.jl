##Machine Learning
using Flux
include(pwd() * "/Diabetes/ML/parseTimeSeries.jl")
include(pwd() * "/Diabetes/ML/plots.jl")
include(pwd() * "/Diabetes/statistics/patientDayStruc.jl")
include(pwd() * "/Diabetes/preproc/dataParsing.jl")

invBGriskFunc(x) = exp((x / γ + β)^(1/α))

function encode(x)
    out = [eval_model(x[i:end])[1] for i in 1:size(x,1)]
    μ, logσ = mean(out), log(std(out))
    z = μ .+ exp.( logσ) .* randn()
    return [z], μ, logσ
end

struct Encoder
    e_μ_layer
    e_σ_layer
end

function enc(in, out)
    e_μ_layer = Chain(Dense(in,out, x->500*σ(x)))
    e_σ_layer = Chain(Dense(in,out, x->σ(x)))
    Encoder(e_μ_layer, e_σ_layer)
end

function (m::Encoder)(x)
    μ = m.e_μ_layer(x)
    σ = m.e_σ_layer(x)
    return μ, σ
end

function vencode(x)
    out = [eval_model(x[i:end])[1] for i in 1:size(x,1)]
    # μ, logσ = mean(out), log(std(out))
    μ, logσ = encoder(reshape(out, 1,:))
    z = mean(μ) .+ exp.( mean(logσ)) .* randn()
    return [z], μ[1], logσ[1]
end

struct Loss
    mse
    mae
    enc
    venc
    gmse
    gmae
    genc
    gvenc
end

expWeight(y) = exp(-abs(1 - y))
stepWeight(y; w=1, wt=3.1) = abs(w*tanh(wt*y))

kl_q_p(μ, logσ) = 0.5f0 * sum(exp.(2f0 .* logσ) + μ.^2 .- 1f0 .- (2 .* logσ))

function selectLossFunction(; w=50, λ=1, β=1)
    return Loss((x,y)->mean(abs2,(eval_model(x) .- y) .* (1 .+ β .* stepWeight.(y))) - λ * mean(memLoss.(eval_model(x), x[end] )),
        (x,y)->mean(abs,(eval_model(x) .- y) .* (1 .+ β .* stepWeight.(y))) - λ * mean(memLoss.(eval_model(x), x[end] )),
        (x,y)->((z, μ, logσ) = encode(x); mean(abs2,(z .- y) ) - λ * mean(memLoss.(mean(z), x[end] )) + (kl_q_p(μ, logσ)) * w ),
        (x,y)->((z, μ, logσ) = vencode(x); mean(abs2,(z .- y) ) - λ * mean(memLoss.(mean(z), x[end] )) + (kl_q_p(μ, logσ)) * w ),
        (x,y)->mean(abs2,(eval_model(x) .- y) .* pen.(eval_model(x),y)) - λ * mean(memLoss.(eval_model(x), x[end] )),
        (x,y)->mean(abs,(eval_model(x) .- y) .* pen.(eval_model(x),y)) - λ * mean(memLoss.(eval_model(x), x[end] )),
        (x,y)->((z, μ, logσ) = encode(x); mean(abs2,(z .- y) .* pen.(mean(z),y) ) - λ * mean(memLoss.(mean(z), x[end] )) + (kl_q_p(μ, logσ)) * w ),
        (x,y)->((z, μ, logσ) = vencode(x); mean(abs2,(z .- y) .* pen.(mean(z),y) ) - λ * mean(memLoss.(mean(z), x[end] )) + (kl_q_p(μ, logσ)) * w ))
end

function eval_model(x)
  out = simple_rnn.(x)[end]
  Flux.reset!(simple_rnn)
  out
end

function eval_model(x)
  out = map(simple_rnn, x)[end]
  Flux.reset!(simple_rnn)
  out
end

function eval_model_mean(x)
    # l = size(x,1)
    # xx = [ x .* vcat(zeros(i), ones(l-i)) for i in 0:l-1]
    # mean(eval_model.([[xx[i]] for i in 1:size(xx,1)]))

    [mean([eval_model(x[i:end])[1] for i in 1:size(x,1)])]
end

function getPred(x,y; getMean=false, using_enc=0)
    if using_enc == 1
        if getMean
            [vencode(x)[1][1] for i in 1:size(y,1)]
        else
            [vencode(x)[1][1] for i in 1:size(y,1)]
        end
    elseif using_enc == 2
        if getMean
            [encode(x)[1][1] for i in 1:size(y,1)]
        else
            [encode(x)[1][1] for i in 1:size(y,1)]
        end
    else
        if getMean
            [eval_model_mean(x[i])[1] for i in 1:size(y,1)]
        else
            [eval_model(x[i])[1] for i in 1:size(y,1)]
        end
    end
end

function getHistPred(x)
    pred = zeros(size(x,1))
    for i in 1:size(x,1)
        pred[i] = eval_model(x[i:end])[1]
    end
    pred
end

function worstError(datA, dt, longTimeSeries, MAXLENGTH, init, add_timestep )
    er = zeros(size(datA,1))
    for i in 1:size(datA,1)
        try
            test_data, test_labels = genVs(datA[i], dt=dt, longTimeSeries=longTimeSeries, MAXLENGTH=MAXLENGTH, init=init, add_timestep=add_timestep)
            if using_enc == 1
                er[i] = √mean((test_labels .- vcat([vencode(x)[1][1] for i in 1:size(test_data,1)]...)) .^ 2)
            elseif using_enc == 2
                er[i] = √mean((test_labels .- vcat([encode(x)[1][1] for i in 1:size(test_data,1)]...)) .^ 2)
            else
                er[i] = √mean((test_labels .- vcat([eval_model(test_data[i]) for i in 1:size(test_data,1)]...)) .^ 2)
            end

        catch
            nothing
        end
    end
    idx = sortperm(er)[end]
    idx, er[idx]
end

function test_loss(test_data, test_labels, simple_rnn, loss; epoch=0)

    mean_loss=[]
    idx_epoch = 0
    for (x,y) in zip(test_data, test_labels)
        append!(mean_loss, loss(x,y))

        idx_epoch += 1
    end
    mean(mean_loss)
end

function pen(x,y; αL=2.5, αH=2, βL=30, βH=100, γL=10, γH=20, TL=85, TH=155)
    1 + αL * (1 - σ(y - (TL + βL))) * σ(x - (y+γL)) + αH * σ(y - (TH+βH)) * (1-σ(x-(y+γH)))
end

memLoss(x,y; w=5, ϵ=0.1) = σ(w*(x-(y-ϵ)))*(1-σ(w*(x-(y+ϵ))))

id(x,y) = 1.0

function custom_train!(train_data, train_labels, simple_rnn, loss, ps, opt; epoch=0)

    mean_loss=[]
    idx_epoch = 0
    for (x,y) in zip(train_data, train_labels)
        # x = [[x[i]] for i in 1:size(x,1)] #patch
        gs = gradient(()->loss(x,y),ps)
        Flux.update!(opt,ps, gs)
        append!(mean_loss, loss(x,y))

        idx_epoch += 1
    end
    mean(mean_loss)
end

function train(simple_rnn, loss; epochs=2, snap=1, printplots=true, pred=2, idx_Tr=1, idx_Te=2, MAXLENGTH=120, longTimeSeries=true, train_on_outlier=25, init=12, lr=0.0001, add_timestep=false) #lossF="MSE"using_enc=0,

    train_data, train_labels = genVs(datA[idx_Tr], dt=pred, longTimeSeries=longTimeSeries, MAXLENGTH=MAXLENGTH, init=init, add_timestep=add_timestep)
    test_data, test_labels = genVs(datA[idx_Te], dt=pred, longTimeSeries=longTimeSeries, MAXLENGTH=MAXLENGTH, init=init, add_timestep=add_timestep)

    opt = ADAM(lr, (0.9, 0.8))
    ps = Flux.params(simple_rnn)

    myPlots(train_data, train_labels, [1], [1], simple_rnn; printplots=printplots, pred=pred)
    mean_loss = []
    test_mean_loss = []
    figsList = []

    for ep = 1:epochs
        @info "Epoch $ep"
        if train_on_outlier < ep && ep % 2 == 0
        # if ep > 6 && std(mean_loss[end-5:end]) < 1.0
            idx = worstError(datA, pred, longTimeSeries, MAXLENGTH, init, add_timestep)[1]
            train_data, train_labels = genVs(datA[idx], dt=pred, longTimeSeries=longTimeSeries, MAXLENGTH=MAXLENGTH, init=init, add_timestep=add_timestep)
            test_data, test_labels = genVs(datA[idx_Te], dt=pred, longTimeSeries=longTimeSeries, MAXLENGTH=MAXLENGTH, init=init, add_timestep=add_timestep)
            @info idx
        end
        l_e = custom_train!(train_data, train_labels, simple_rnn, loss, ps, opt)
        append!(mean_loss, l_e)
        append!(test_mean_loss, test_loss(test_data, test_labels, simple_rnn, loss))
        @info l_e

        if ep % snap == 0
            f = myPlots(train_data, train_labels, mean_loss, test_mean_loss, simple_rnn; svfig=true, timestamp=true, printplots=printplots, pred=pred)
            append!(figsList, [f])
            figsList[end]
        end
    end
    mean_loss, figsList#, simple_rnn
end

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


datA = getTimeSeries(bg=true)
train_data, train_labels = genVs(datA[5], dt=3, longTimeSeries=false, init=5, add_timestep=true)

# Models
stepWeight(y; w=100, wt=3.2) = abs(w*tanh(wt*y))
stepWeight(0.21)
histogram(stepWeight.(datA[1], wt=8.2), normalize=true)

ll = selectLossFunction(w=0, λ=-10000.9)
ll = selectLossFunction(λ=0, β=0)
sqrt(ll.genc(train_data[1], train_labels[1]))
ll.gmse(train_data[1], train_labels[1])
ll.enc(train_data[1], train_labels[1])
ll.mse(train_data[1], train_labels[1])
# ll.venc(train_data[1], train_labels[1])
train_labels[1]
eval_model(train_data[1])[1]
histogram([encode(train_data[3])[1][1] for i in 1:1000])
plot([vencode(train_data[i])[1][1] for i in 1:1000])
vencode(train_data[1])

simple_rnn = Flux.RNN(1, 1, (x -> x))
simple_rnn = Chain(RNN(1, 3, (x->x)), Dense(3, 1, x->leakyrelu(x, 0.01)))
simple_rnn = Chain(RNN(1, 1, (x->x)), RNN(1, 1, (x->x)))
simple_rnn = Chain(RNN(1, 1, (x->x)), RNN(1, 1, (x->x)), RNN(1, 1, (x->x)))

simple_rnn = LSTM(1,1)
simple_rnn = GRU(1,1)
simple_rnn = GRUv3(1,1)


mean_loss, figsList = train(simple_rnn, ll.mae, epochs=50; pred=6, idx_Tr=1, idx_Te=2, MAXLENGTH=120, longTimeSeries=false, init=5, add_timestep=true, train_on_outlier=55, lr=0.0002)

mean_loss, figsList = train(simple_rnn, ll.enc, epochs=50; pred=6, idx_Tr=1, idx_Te=2, MAXLENGTH=120, longTimeSeries=false, init=10, add_timestep=true, train_on_outlier=55, using_enc=2, lr=0.0002)

mean_loss, figsList = train(simple_rnn, ll.genc, epochs=50; pred=6, idx_Tr=1, idx_Te=2, MAXLENGTH=120, longTimeSeries=false, init=10, add_timestep=true, train_on_outlier=55, using_enc=2, lr=0.0004)

mean_loss, figsList = train(simple_rnn, ll.gmse, epochs=50; pred=6, idx_Tr=1, idx_Te=2, MAXLENGTH=120, longTimeSeries=false, init=10, add_timestep=true, train_on_outlier=15, using_enc=0, lr=0.0004)


sqrt(mean(abs2, vcat([invBGriskFunc.(eval_model(train_data[i]) .* √10) for i in 1:size(train_data,1)]...) .- invBGriskFunc.(train_labels .* √10)))

using BSON: @save, @load
using Zygote


function trainAndSaveModel(simple_rnn ; path = "/Diabetes/models/15/RNN0/", idx=0, pred=6)
    mean_loss, figsList = train(simple_rnn, ll.mse, epochs=30; pred=pred, idx_Tr=1, idx_Te=2, MAXLENGTH=120, longTimeSeries=false, init=3, add_timestep=true, train_on_outlier=55, lr=0.0002)

    savePath = pwd() * path * string(idx)
    modelPath = savePath * "/model.bson"

    isdir(savePath) || mkpath(savePath)
    @save modelPath simple_rnn

    isdir(savePath * "/Plots") || mkdir(savePath * "/Plots")
    for i in 1:size(figsList,1)
        savefig(figsList[i], savePath * "/Plots/$i")
    end
end

datA = getTimeSeries(bg=false)
function trainSamples()
    for i in 1:5
        @info "Model $i"
        # global simple_rnn = Flux.RNN(1, 1, (x -> x))
        # global simple_rnn = Flux.RNN(1, 1, (x -> x))

        global simple_rnn = LSTM(1,1)
        global simple_rnn = LSTM(1,1)

        # global simple_rnn = GRU(1,1)
        # global simple_rnn = GRU(1,1)
        trainAndSaveModel(simple_rnn, path = "/Diabetes/models/15/LSTM/", idx=i, pred=3)
        @info eval_model(train_data[1])[1]
    end
end

trainSamples()

simple_rnn = Flux.RNN(1, 1, (x -> x))
@load pwd() * "/Diabetes/models/RNN0/0/model.bson" simple_rnn


###########ENDS HERE#####

function timeCorr(a,b, Δ)
    # mean([ √((a[i] - b[i+Δ])^2/a[i]^2) for i in 1:size(b,1)-Δ ])
    mean([ (a[i] - mean(a[1:size(a,1)-Δ]))*(b[i+Δ] - mean(b[Δ+1:size(b,1)]))/(std(a[1:size(a,1)-Δ])*std(b[Δ+1:size(b,1)])) for i in 1:size(b,1)-Δ ])
end

function sample(x; s=100)
    ar = hcat([ [encode(x[j])[1][1] for i in 1:s] for j in 1:size(x,1)]...)
    mean(ar, dims=1)[1,:], std(ar, dims=1)[1,:]
end

function computeErrors(x,y, f=id, er="RMSE")
    if er == "RMSE"
        e = √mean(abs2, (x .- y) .* f.(x,y))
    elseif er == "RMSRelE"
        e = √mean(abs2, (x .- y) .* f.(x,y) ./ y)
    elseif er == "CoD"
        e = 1 - mean(abs2, (x .- y) .* f.(x,y) ) / mean(abs2, (y .- mean(y)) .* f.(x,y))
    end
    return e
end


######Animation###################
g = @animate for i in 1:size(figsList,1)
        plot(figsList[i])
    end


gif(g, "./Diabetes/training.gif", fps = 5)
size(figsList,1)


####################
# using HDF5, JLD
# using BSON
# @time datA = getTimeSeries(bg=true, lim=0, idx=0)
# # datA = getTimeSeries(lim=30)
# simple_rnn = Flux.RNN(1, 1, (x -> x))
# BSON.@load pwd() * "/Diabetes/models/30/RNN/2/model.bson" simple_rnn
#
# train_data, train_labels = genVs(datA[2], dt=6, longTimeSeries=false, init=10, add_timestep=true)
#
# m, s = sample(train_data)

####
function getError(; path = "/Diabetes/models/15/RNN0/", idx=0, pred=3, MAX=10, init=10)
    train_data, train_labels = genVs(datA[1], dt=pred, longTimeSeries=false, init=init, add_timestep=true)

    # m, s = sample(train_data)

    errorDict = Dict()
    errorDict["ModelName"] = path

    # errorDict["modelSample"] = [sqrt(mean(abs2, m .- train_labels))]
    errorDict["model"] = [sqrt(mean(abs2, vcat([eval_model(train_data[i]) for i in 1:size(train_data,1)]...) .- train_labels))]
    errorDict["modelLastMeasurement"] = [sqrt(mean(abs2, vcat([train_data[i][end] for i in 1:size(train_data,1)]...) .- train_labels))]

    # errorDict["modelSampleCGM"] = [sqrt(mean(abs2, invBGriskFunc.(m .* √10) .- invBGriskFunc.(train_labels .* √10)))]
    errorDict["modelCGM"] = [sqrt(mean(abs2, vcat([invBGriskFunc.(eval_model(train_data[i]) .* √10) for i in 1:size(train_data,1)]...) .- invBGriskFunc.(train_labels .* √10)))]
    errorDict["modelLastMeasurementCGM"] = [sqrt(mean(abs2, vcat([invBGriskFunc.(train_data[i][end] .* √10) for i in 1:size(train_data,1)]...) .- invBGriskFunc.(train_labels .* √10)))]

    # errorDict["modelSampleBGrisk"] = [sqrt(mean(abs2, (m .- train_labels) .* ((train_labels .* √10) .^ 2 .* 10)))]
    errorDict["modelBGrisk"] = [sqrt(mean(abs2, (vcat([eval_model(train_data[i]) for i in 1:size(train_data,1)]...) .- train_labels) .* ((train_labels .* √10) .^ 2 .* 10)))]
    errorDict["modelLastMeasurementBGrisk"] = [sqrt(mean(abs2, (vcat([train_data[i][end] for i in 1:size(train_data,1)]...) .- train_labels) .* ((train_labels .* √10) .^ 2 .* 10)))]

    errorDict["modelBGriskCGM"] = [sqrt(mean(abs2, (vcat([invBGriskFunc.(eval_model(train_data[i]) .* √10) for i in 1:size(train_data,1)]...) .- invBGriskFunc.(train_labels .* √10)) .* ((train_labels .* √10) .^ 2 .* 10)))]

    errorDict["modelLastMeasurementBGriskCGM"] = [sqrt(mean(abs2, (vcat([invBGriskFunc.(train_data[i][end] .* √10) for i in 1:size(train_data,1)]...) .- invBGriskFunc.(train_labels .* √10)) .* ((train_labels .* √10) .^ 2 .* 10)))]

    for i in 2:MAX
        if i ∉ collect(76:86) #if i != 76
            @info i
            train_data, train_labels = genVs(datA[i], dt=pred, longTimeSeries=false, init=init, add_timestep=true)

            # m, s = sample(train_data)

            # append!(errorDict["modelSample"], [sqrt(mean(abs2, m .- train_labels))])
            append!(errorDict["model"],[sqrt(mean(abs2, vcat([eval_model(train_data[i]) for i in 1:size(train_data,1)]...) .- train_labels))])
            append!(errorDict["modelLastMeasurement"], [sqrt(mean(abs2, vcat([train_data[i][end] for i in 1:size(train_data,1)]...) .- train_labels))])

            # append!(errorDict["modelSampleCGM"], [sqrt(mean(abs2, invBGriskFunc.(m .* √10) .- invBGriskFunc.(train_labels .* √10)))])
            append!(errorDict["modelCGM"], [sqrt(mean(abs2, vcat([invBGriskFunc.(eval_model(train_data[i]) .* √10) for i in 1:size(train_data,1)]...) .- invBGriskFunc.(train_labels .* √10)))])
            append!(errorDict["modelLastMeasurementCGM"], [sqrt(mean(abs2, vcat([invBGriskFunc.(train_data[i][end] .* √10) for i in 1:size(train_data,1)]...) .- invBGriskFunc.(train_labels .* √10)))])

            # append!(errorDict["modelSampleBGrisk"], [sqrt(mean(abs2, (m .- train_labels) .* ((train_labels .* √10) .^ 2 .* 10)))])
            append!(errorDict["modelBGrisk"], [sqrt(mean(abs2, (vcat([eval_model(train_data[i]) for i in 1:size(train_data,1)]...) .- train_labels) .* ((train_labels .* √10) .^ 2 .* 10)))])
            append!(errorDict["modelLastMeasurementBGrisk"], [sqrt(mean(abs2, (vcat([train_data[i][end] for i in 1:size(train_data,1)]...) .- train_labels) .* ((train_labels .* √10) .^ 2 .* 10)))])

            append!(errorDict["modelBGriskCGM"], [sqrt(mean(abs2, (vcat([invBGriskFunc.(eval_model(train_data[i]) .* √10) for i in 1:size(train_data,1)]...) .- invBGriskFunc.(train_labels .* √10)) .* ((train_labels .* √10) .^ 2 .* 10)))])

            append!(errorDict["modelLastMeasurementBGriskCGM"], [sqrt(mean(abs2, (vcat([invBGriskFunc.(train_data[i][end] .* √10) for i in 1:size(train_data,1)]...) .- invBGriskFunc.(train_labels .* √10)) .* ((train_labels .* √10) .^ 2 .* 10)))])
        end
    end
    save(pwd() * path * "$idx/error-4.jld", "data", errorDict)
    errorDict
end

simple_rnn = Flux.RNN(1, 1, (x -> x))
simple_rnn = GRU(1,1)
simple_rnn = LSTM(1,1)
for i in 1:5
    BSON.@load pwd() * "/Diabetes/models/15/RNN/$i/model.bson" simple_rnn
    errorDict = getError(path = "/Diabetes/models/15/RNN/", pred = 3, idx = i, MAX = 139)
end


function getError(; path = "/Diabetes/models/15/RNN0/", idx=0, pred=3, MAX=10, init=10) #Only for RNN0
    train_data, train_labels = genVs(datA[1], dt=pred, longTimeSeries=false, init=init, add_timestep=true)

    # m, s = sample(train_data)

    errorDict = Dict()
    errorDict["ModelName"] = path

    # errorDict["modelSample"] = [sqrt(mean(abs2, m .- train_labels))]
    errorDict["model"] = [sqrt(mean(abs2, vcat([eval_model(train_data[i]) for i in 1:size(train_data,1)]...) .- train_labels))]
    errorDict["modelLastMeasurement"] = [sqrt(mean(abs2, vcat([train_data[i][end] for i in 1:size(train_data,1)]...) .- train_labels))]

    # errorDict["modelSampleCGM"] = [sqrt(mean(abs2, invBGriskFunc.(m .* √10) .- invBGriskFunc.(train_labels .* √10)))]
    # errorDict["modelCGM"] = [sqrt(mean(abs2, vcat([invBGriskFunc.(eval_model(train_data[i]) .* √10) for i in 1:size(train_data,1)]...) .- invBGriskFunc.(train_labels .* √10)))]
    # errorDict["modelLastMeasurementCGM"] = [sqrt(mean(abs2, vcat([invBGriskFunc.(train_data[i][end] .* √10) for i in 1:size(train_data,1)]...) .- invBGriskFunc.(train_labels .* √10)))]

    # errorDict["modelSampleBGrisk"] = [sqrt(mean(abs2, (m .- train_labels) .* ((train_labels .* √10) .^ 2 .* 10)))]
    errorDict["modelBGrisk"] = [sqrt(mean(abs2, (vcat([eval_model(train_data[i]) for i in 1:size(train_data,1)]...) .- train_labels) .* (BGriskFunc.(train_labels) .* √10) .^ 2 ))]
    errorDict["modelLastMeasurementBGrisk"] = [sqrt(mean(abs2, (vcat([train_data[i][end] for i in 1:size(train_data,1)]...) .- train_labels) .* (BGriskFunc.(train_labels) .* √10) .^ 2 ))]

    # errorDict["modelBGriskCGM"] = [sqrt(mean(abs2, (vcat([invBGriskFunc.(eval_model(train_data[i]) .* √10) for i in 1:size(train_data,1)]...) .- invBGriskFunc.(train_labels .* √10)) .* ((train_labels .* √10) .^ 2 .* 10)))]

    # errorDict["modelLastMeasurementBGriskCGM"] = [sqrt(mean(abs2, (vcat([invBGriskFunc.(train_data[i][end] .* √10) for i in 1:size(train_data,1)]...) .- invBGriskFunc.(train_labels .* √10))) .* ((train_labels .* √10) .^ 2 .* 10)))]

    for i in 2:MAX
        if i ∉ collect(76:86) #if i != 76
            @info i
            train_data, train_labels = genVs(datA[i], dt=pred, longTimeSeries=false, init=init, add_timestep=true)

            # m, s = sample(train_data)

            # append!(errorDict["modelSample"], [sqrt(mean(abs2, m .- train_labels))])
            append!(errorDict["model"],[sqrt(mean(abs2, vcat([eval_model(train_data[i]) for i in 1:size(train_data,1)]...) .- train_labels))])
            append!(errorDict["modelLastMeasurement"], [sqrt(mean(abs2, vcat([train_data[i][end] for i in 1:size(train_data,1)]...) .- train_labels))])

            # append!(errorDict["modelSampleCGM"], [sqrt(mean(abs2, invBGriskFunc.(m .* √10) .- invBGriskFunc.(train_labels .* √10)))])
            # append!(errorDict["modelCGM"], [sqrt(mean(abs2, vcat([invBGriskFunc.(eval_model(train_data[i]) .* √10) for i in 1:size(train_data,1)]...) .- invBGriskFunc.(train_labels .* √10)))])
            # append!(errorDict["modelLastMeasurementCGM"], [sqrt(mean(abs2, vcat([invBGriskFunc.(train_data[i][end] .* √10) for i in 1:size(train_data,1)]...) .- invBGriskFunc.(train_labels .* √10)))])

            # append!(errorDict["modelSampleBGrisk"], [sqrt(mean(abs2, (m .- train_labels) .* ((train_labels .* √10) .^ 2 .* 10)))])
            append!(errorDict["modelBGrisk"], [sqrt(mean(abs2, (vcat([eval_model(train_data[i]) for i in 1:size(train_data,1)]...) .- train_labels) .* (BGriskFunc.(train_labels) .* √10) .^ 2 ))])
            append!(errorDict["modelLastMeasurementBGrisk"], [sqrt(mean(abs2, (vcat([train_data[i][end] for i in 1:size(train_data,1)]...) .- train_labels) .* (BGriskFunc.(train_labels) .* √10) .^ 2 ))])

            # append!(errorDict["modelBGriskCGM"], [sqrt(mean(abs2, (vcat([invBGriskFunc.(eval_model(train_data[i]) .* √10) for i in 1:size(train_data,1)]...) .- invBGriskFunc.(train_labels .* √10)) .* ((train_labels .* √10) .^ 2 .* 10)))])

            # append!(errorDict["modelLastMeasurementBGriskCGM"], [sqrt(mean(abs2, (vcat([invBGriskFunc.(train_data[i][end] .* √10) for i in 1:size(train_data,1)]...) .- invBGriskFunc.(train_labels .* √10)) .* ((train_labels .* √10) .^ 2 .* 10)))])
        end
    end
    save(pwd() * path * "$idx/error-4.jld", "data", errorDict)
    errorDict
end

@time datA = getTimeSeries(bg=false, lim=0, idx=0)

simple_rnn = Flux.RNN(1, 1, (x -> x))
for i in 1:10
    BSON.@load pwd() * "/Diabetes/models/15/RNN0/$i/model.bson" simple_rnn
    errorDict = getError(path = "/Diabetes/models/15/RNN0/", pred = 3, idx = i, MAX = 139)
end


errorDict = load(pwd() * "/Diabetes/models/15/GRU/1/error.jld")["data"]


##
using StatsPlots, Statistics, StatsBase
using HDF5, JLD

# savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/GlucosePrediction/Figs/60minPredBGerror.png")
errorDictGRU = load(pwd() * "/Diabetes/models/15/RNN0/2/error-4.jld")["data"]

cSch = Dict()
cSch["CNN"] = RGB(1,0,1)
cSch["LSTM"] = RGB(1,1,0)
cSch["GRU"] = RGB(0,1,1)
cSch["RNN"] = RGB(0.5,1,0.5)
cSch["RNN0"] = RGB(0.5,0.5,0.9)
cSch["LM"] = RGB(0.5,0.5,0.5)
# cSch["CNN2"] = RGB(1,0.8,1)
cSch["CNN10"] = RGB(1,0.8,1)
cSch["VAE"] = RGB(1,0.8,0.8)

yPos = Dict()
yPos["15"]=[23,0.4]
yPos["30"]=[23,0.206]
yPos["60"]=[23,0.265]

###ξ
begin
    Δt = 15
# for Δt in [15, 30, 60]
    f = plot()
    for tag in ["LSTM", "GRU", "RNN", "CNN", "CNN10"]
        for j in 1:1
            try
                errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/$tag/$j/error-3.jld")["data"]
                f = violin!(["$tag-$j" for i in 1:138], errorDictGRU["model"], fillalpha=1, linewidth=0, label=:none, c=cSch[tag])
                f = boxplot!(["$tag-$j" for i in 1:138], errorDictGRU["model"], fillalpha=0.75, linewidth=2, label=:none, xrot=50, c=cSch[tag])
            catch
                nothing
            end
        end
    end
    errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/GRU/2/error-3.jld")["data"]
    f = violin!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurement"], fillalpha=1, linewidth=0, label=:none, c=cSch["LM"])
    f = boxplot!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurement"], fillalpha=0.75, linewidth=2, label=:none, c=cSch["LM"])
    f = hline!([percentile(errorDictGRU["modelLastMeasurement"], 75)], fillrange=percentile(errorDictGRU["modelLastMeasurement"], 25), fillalpha=0.3, lw=0, legend=:none, c=:black)
    f = plot!(frame=:box, size=(700,500), ylabel="ξ RMSE", xticks = :all, ann=[(yPos[string(Δt)][1],yPos[string(Δt)][2],text("Δt=$Δt", 21,:black, font))])

    # savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/GlucosePrediction/Figs/$(Δt)minPredBGerror.png")
end

###CGM
yPos = Dict()
yPos["15"]=[23,5.4]
yPos["30"]=[23,5.201]
yPos["60"]=[23,5.265]
begin
    Δt = 15
# for Δt in [15, 30, 60]
    f = plot()
    for tag in ["RNN0", "LSTM", "GRU", "RNN", "CNN", "CNN10"]# for tag in ["RNN0","GRU", "RNN", "CNN", "CNN2", "VAE"]
        for j in 1:1
            try
                if tag != "RNN0"
                    errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/$tag/$j/error-3.jld")["data"]
                    f = violin!(["$tag-$j" for i in 1:size(errorDictGRU["modelCGM"],1)], errorDictGRU["modelCGM"], fillalpha=1, linewidth=0, label=:none, c=cSch[tag])
                    f = boxplot!(["$tag-$j" for i in 1:size(errorDictGRU["modelCGM"],1)], errorDictGRU["modelCGM"], fillalpha=0.75, linewidth=2, label=:none, xrot=50, c=cSch[tag])
                else
                    j = j+5
                    errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/$tag/$j/error-3.jld")["data"]
                    f = violin!(["$tag-$j" for i in 1:138], errorDictGRU["model"], fillalpha=1, linewidth=0, label=:none, c=cSch[tag])
                    f = boxplot!(["$tag-$j" for i in 1:138], errorDictGRU["model"], fillalpha=0.75, linewidth=2, label=:none, xrot=50, c=cSch[tag])
                end
            catch
                nothing
            end
        end
    end
    errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/GRU/2/error-3.jld")["data"]
    f = violin!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurementCGM"], fillalpha=1, linewidth=0, label=:none, c=cSch["LM"])
    f = boxplot!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurementCGM"], fillalpha=0.75, linewidth=2, label=:none, c=cSch["LM"])
    f = hline!([percentile(errorDictGRU["modelLastMeasurementCGM"], 75)], fillrange=percentile(errorDictGRU["modelLastMeasurementCGM"], 25), fillalpha=0.3, lw=0, legend=:none, c=:black)

    # f = violin!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurementCGM"], fillalpha=1, linewidth=0, label=:none)
    # f = boxplot!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurementCGM"], fillalpha=0.75, linewidth=2, label=:none)
    f = plot!(frame=:box, size=(700,500), ylabel="CGM RMSE", xticks = :all, ann=[(yPos[string(Δt)][1],yPos[string(Δt)][2],text("Δt=$Δt", 21,:black, font))], ylim=(0,100))

    # savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/GlucosePrediction/Figs/$(Δt)minPredCGM.png")
end

###BGRisk
yPos = Dict()
yPos["15"]=[23,12.4]
yPos["30"]=[23,4.801]
yPos["60"]=[23,5.865]
begin
    Δt = 60
for Δt in [15, 30, 60]
    f = plot()
    for tag in ["LSTM", "GRU", "RNN", "CNN", "CNN10"] #for tag in ["CNN", "LSTM", "GRU", "RNN"]
        for j in 1:5
            try
                errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/$tag/$j/error-3.jld")["data"]
                f = violin!(["$tag-$j" for i in 1:138], errorDictGRU["modelBGrisk"], fillalpha=1, linewidth=0, label=:none, c=cSch[tag])
                f = boxplot!(["$tag-$j" for i in 1:138], errorDictGRU["modelBGrisk"], fillalpha=0.75, linewidth=2, label=:none, xrot=50, c=cSch[tag])
            catch
                nothing
            end
        end
    end
    errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/GRU/2/error-3.jld")["data"]
    f = violin!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurementBGrisk"], fillalpha=1, linewidth=0, label=:none, c=cSch["LM"])
    f = boxplot!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurementBGrisk"], fillalpha=0.75, linewidth=2, label=:none, c=cSch["LM"])
    f = plot!(frame=:box, size=(700,500), ylabel="BG weighed RMSE", xticks = :all, ann=[(yPos[string(Δt)][1],yPos[string(Δt)][2],text("Δt=$Δt", 21,:black, font))], ylim=(0,yPos[string(Δt)][2]+1))
    f = hline!([percentile(errorDictGRU["modelLastMeasurementBGrisk"], 75)], fillrange=percentile(errorDictGRU["modelLastMeasurementBGrisk"], 25), fillalpha=0.3, lw=0, legend=:none, c=:black)

    savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/GlucosePrediction/Figs/$(Δt)minPredBGWeight.png")
end


###BGRiskWeighedCGM
yPos = Dict()
yPos["15"]=[25,15.4]
yPos["30"]=[23,17.201]
yPos["60"]=[23,20.265]
begin
    Δt = 60
# for Δt in [15, 30, 60]
    f = plot()
    for tag in ["RNN0", "LSTM", "GRU", "RNN", "CNN", "CNN10"]# for tag in ["RNN0","GRU", "RNN", "CNN", "CNN2", "VAE"]
        for j in 1:5
            try
                if tag != "RNN0"
                    errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/$tag/$j/error-4.jld")["data"]
                    f = violin!(["$tag-$j" for i in 1:size(errorDictGRU["modelBGriskCGM"],1)], errorDictGRU["modelBGriskCGM"] ./ 100, fillalpha=1, linewidth=0, label=:none, c=cSch[tag])
                    f = boxplot!(["$tag-$j" for i in 1:size(errorDictGRU["modelBGriskCGM"],1)], errorDictGRU["modelBGriskCGM"] ./ 100, fillalpha=0.75, linewidth=2, label=:none, xrot=50, c=cSch[tag])
                else
                    j = j+5
                    errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/$tag/$j/error-4.jld")["data"]
                    f = violin!(["$tag-$j" for i in 1:138], errorDictGRU["modelBGrisk"] ./ 100, fillalpha=1, linewidth=0, label=:none, c=cSch[tag])
                    f = boxplot!(["$tag-$j" for i in 1:138], errorDictGRU["modelBGrisk"] ./ 100, fillalpha=0.75, linewidth=2, label=:none, xrot=50, c=cSch[tag])
                end
            catch
                nothing
            end
        end
    end
    errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/RNN/2/error-4.jld")["data"]
    f = violin!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurementBGriskCGM"] ./ 100, fillalpha=1, linewidth=0, label=:none, c=cSch["LM"])
    f = boxplot!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurementBGriskCGM"]  ./ 100, fillalpha=0.75, linewidth=2, label=:none, c=cSch["LM"])
    f = hline!([percentile(errorDictGRU["modelLastMeasurementBGriskCGM"] ./ 100, 75)], fillrange=percentile(errorDictGRU["modelLastMeasurementBGriskCGM"] ./ 100, 25), fillalpha=0.3, lw=0, legend=:none, c=:black)

    # f = violin!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurementCGM"], fillalpha=1, linewidth=0, label=:none)
    # f = boxplot!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurementCGM"], fillalpha=0.75, linewidth=2, label=:none)
    f = plot!(frame=:box, size=(700,500), ylabel="BG-score weighed CGM RMSE", xticks = :all, ann=[(yPos[string(Δt)][1],yPos[string(Δt)][2],text("Δt=$Δt", 21,:black, font))], ylim=(0,25))

    savefig(f, "/Users/javier/Library/CloudStorage/Dropbox/Aplicaciones/Overleaf/GlucosePrediction/Figs/$(Δt)minPredCGMBGWeighed.png")
    f
end


###LM different time
begin

    plot()
    for Δt in ["15", "30", "60"]
        # for j in 1:1
        #     try
                # errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/RNN/$j/error-2.jld")["data"]
        #         f = violin!(["$tag-$j" for i in 1:138], errorDictGRU["modelCGM"], fillalpha=1, linewidth=0, label=:none)
        #         f = boxplot!(["$tag-$j" for i in 1:138], errorDictGRU["modelCGM"], fillalpha=0.75, linewidth=2, label=:none, xrot=50)
        #     catch
        #         nothing
        #     end
        # end
        errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/RNN/1/error-2.jld")["data"]
        f = violin!(["LM-$Δt" for i in 1:138], errorDictGRU["modelLastMeasurementCGM"], fillalpha=1, linewidth=0, label=:none)
        f = boxplot!(["LM-$Δt" for i in 1:138], errorDictGRU["modelLastMeasurementCGM"], fillalpha=0.75, linewidth=2, label=:none)
    end

    f = plot!(frame=:box, size=(700,500), ylabel="CGM RMSE", xticks = :all)
end


###
begin
    Δt = 30
# for Δt in [15, 30, 60]
    f = plot()
    errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/GRU/5/error-2.jld")["data"]
    idx = sortperm(vcat(errorDictGRU["modelLastMeasurementCGM"][1:75], errorDictGRU["modelLastMeasurementCGM"][86:138]))
    # idx = sortperm(errorDictGRU["modelCGM"])
    f = plot!(vcat(errorDictGRU["modelLastMeasurementCGM"][1:75], errorDictGRU["modelLastMeasurementCGM"][86:138])[idx])
    for tag in ["CNN2"] #"RNN0","LSTM", "GRU", "RNN", "CNN",
        for j in 1:15
            try
                if tag != "RNN0"
                    errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/$tag/$j/error-2.jld")["data"]
                    f = plot!(errorDictGRU["modelCGM"][idx])
                else
                    nothing
                    # errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/$tag/$j/error-2.jld")["data"]
                    # f = violin!(["$tag-$j" for i in 1:138], errorDictGRU["model"], fillalpha=1, linewidth=0, label=:none, c=cSch[tag])
                    # f = boxplot!(["$tag-$j" for i in 1:138], errorDictGRU["model"], fillalpha=0.75, linewidth=2, label=:none, xrot=50, c=cSch[tag])
                end
            catch
                nothing
            end
        end
    end


    f = plot!(frame=:box, size=(700,500), ylabel="CGM RMSE", xticks = :all, ann=[(yPos[string(Δt)][1],yPos[string(Δt)][2],text("Δt=$Δt", 21,:black, font))], legend=false)

    # savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/GlucosePrediction/Figs/$(Δt)minPredCGM.png")
end



### Mean per model

begin
    Δt = 60
# for Δt in [15, 30, 60]
    # f = plot()
    meanErr = Dict()
    for tag in ["RNN0", "LSTM", "GRU", "RNN", "CNN", "CNN10"] # ["RNN0","GRU", "RNN", "CNN", "CNN2", "VAE"]
        for j in 1:5
            try
                if tag != "RNN0"
                    errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/$tag/$j/error-2.jld")["data"]
                    # f = violin!(["$tag-$j" for i in 1:size(errorDictGRU["modelCGM"],1)], errorDictGRU["modelCGM"], fillalpha=1, linewidth=0, label=:none, c=cSch[tag])
                    # f = boxplot!(["$tag-$j" for i in 1:size(errorDictGRU["modelCGM"],1)], errorDictGRU["modelCGM"], fillalpha=0.75, linewidth=2, label=:none, xrot=50, c=cSch[tag])
                    @info mean(errorDictGRU["modelCGM"])
                    meanErr["$tag-$j"] = [mean(errorDictGRU["modelCGM"]), median(errorDictGRU["modelCGM"]), maximum(errorDictGRU["modelCGM"]), minimum(errorDictGRU["modelCGM"]), std(errorDictGRU["modelCGM"])]
                else
                    errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/$tag/$j/error-2.jld")["data"]
                    # f = violin!(["$tag-$j" for i in 1:138], errorDictGRU["model"], fillalpha=1, linewidth=0, label=:none, c=cSch[tag])
                    # f = boxplot!(["$tag-$j" for i in 1:138], errorDictGRU["model"], fillalpha=0.75, linewidth=2, label=:none, xrot=50, c=cSch[tag])

                    meanErr["$tag-$j"] = [mean(errorDictGRU["model"]), median(errorDictGRU["model"]), maximum(errorDictGRU["model"]), minimum(errorDictGRU["model"]), std(errorDictGRU["model"])]
                end
            catch
                nothing
            end
        end
    end
    errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/GRU/2/error-2.jld")["data"]
    # f = violin!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurementCGM"], fillalpha=1, linewidth=0, label=:none, c=cSch["LM"])
    # f = boxplot!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurementCGM"], fillalpha=0.75, linewidth=2, label=:none, c=cSch["LM"])

    # f = violin!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurementCGM"], fillalpha=1, linewidth=0, label=:none)
    # f = boxplot!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurementCGM"], fillalpha=0.75, linewidth=2, label=:none)
    # f = plot!(frame=:box, size=(700,500), ylabel="CGM RMSE", xticks = :all, ann=[(yPos[string(Δt)][1],yPos[string(Δt)][2],text("Δt=$Δt", 21,:black, font))], ylim=(0,100))

    meanErr["LM"] = [mean(errorDictGRU["modelLastMeasurementCGM"]), median(errorDictGRU["modelLastMeasurementCGM"]), maximum(errorDictGRU["modelLastMeasurementCGM"]), minimum(errorDictGRU["modelLastMeasurementCGM"]), std(errorDictGRU["modelLastMeasurementCGM"])]

    # savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/GlucosePrediction/Figs/$(Δt)minPredCGM.png")
end
# meanErr

begin
    plot()
    for tag0 in ["RNN0", "LSTM", "GRU", "RNN", "CNN", "CNN10"] # ["RNN0","GRU", "RNN", "CNN", "CNN2", "VAE"]
        for j in 1:5
            try
                tag = tag0 * "-" * "$j"
                @info meanErr[tag][1]
                plot!([tag], [meanErr[tag][1] / meanErr["LM"][1]], yerr=meanErr[tag][1] / meanErr["LM"][1] * √((meanErr[tag][5] / meanErr[tag][1])^2 + (meanErr["LM"][5] / meanErr["LM"][1])^2), st=:scatter, frame=:box, ms=10, s=:auto, markershapes = :circle, lw=0, markerstrokewidth=1, xlabel="Model", c=cSch[tag0], ylabel="Reduced CGM RMSE", xrot=45, xticks=:all)


            catch
                nothing
            end
        end
    end
    plot!(["LM"], [meanErr["LM"][1] / meanErr["LM"][1]], yerr=meanErr["LM"][1] / meanErr["LM"][1] * √((meanErr["LM"][5] / meanErr["LM"][1])^2 + (meanErr["LM"][5] / meanErr["LM"][1])^2), st=:scatter, frame=:box, ms=10, s=:auto, markershapes = :circle, lw=0, markerstrokewidth=1, xlabel="Model", c=cSch["LM"], ylabel="Reduced CGM RMSE", xrot=45, xticks=:all)
    plot!(legend=false, margin=4mm, ann=[(20,1.7,text("Δt=60", 21,:black, font))])
    plot!(ylims=(0,2))
    savefig("/Users/javier/Dropbox/Aplicaciones/Overleaf/GlucosePrediction/Figs/60minPredCGMReduced.png")
end










################VDRD

yPos = Dict()
yPos["15"]=[3,5.4]
yPos["30"]=[3,5.201]
yPos["60"]=[3,5.265]
begin
    Δt = 60
# for Δt in [15, 30, 60]
    f = plot()
    for tag in ["RNN0", "LSTM", "GRU", "RNN", "CNN", "CNN10"]# for tag in ["RNN0","GRU", "RNN", "CNN", "CNN2", "VAE"]
        for j in 1:1
            try
                if tag != "RNN0"
                    errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/$tag/$j/error-3.jld")["data"]
                    f = violin!(["$tag" for i in 1:size(errorDictGRU["modelCGM"],1)], errorDictGRU["modelCGM"], fillalpha=1, linewidth=0, label=:none, c=cSch[tag])
                    f = boxplot!(["$tag" for i in 1:size(errorDictGRU["modelCGM"],1)], errorDictGRU["modelCGM"], fillalpha=0.75, linewidth=2, label=:none, xrot=0, c=cSch[tag])
                else
                    j = j+5
                    errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/$tag/$j/error-3.jld")["data"]
                    f = violin!(["$tag" for i in 1:138], errorDictGRU["model"], fillalpha=1, linewidth=0, label=:none, c=cSch[tag])
                    f = boxplot!(["$tag" for i in 1:138], errorDictGRU["model"], fillalpha=0.75, linewidth=2, label=:none, xrot=0, c=cSch[tag])
                end
            catch
                nothing
            end
        end
    end
    errorDictGRU = load(pwd() * "/Diabetes/models/$Δt/GRU/2/error-3.jld")["data"]
    f = violin!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurementCGM"], fillalpha=1, linewidth=0, label=:none, c=cSch["LM"])
    f = boxplot!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurementCGM"], fillalpha=0.75, linewidth=2, label=:none, c=cSch["LM"])
    f = hline!([percentile(errorDictGRU["modelLastMeasurementCGM"], 75)], fillrange=percentile(errorDictGRU["modelLastMeasurementCGM"], 25), fillalpha=0.3, lw=0, legend=:none, c=:black)

    # f = violin!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurementCGM"], fillalpha=1, linewidth=0, label=:none)
    # f = boxplot!(["LM" for i in 1:138], errorDictGRU["modelLastMeasurementCGM"], fillalpha=0.75, linewidth=2, label=:none)
    f = plot!(frame=:box, size=(700,500), ylabel="CGM RMSE (mg/dL)", xticks = :all, ann=[(yPos[string(Δt)][1],yPos[string(Δt)][2],text("Δt=$Δt", 21,:red, font))], ylim=(0,100), tickfontsize=12, yguidefontsize=20)

    savefig(f, pwd() * "/Diabetes/figs/$(Δt)minPredCGM-1.png")
    f
end
