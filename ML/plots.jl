using DataFrames, CSV, Plots, Interact, Statistics, Dates



function myPlot(y,ŷ; flag=0, label="Test")
    if flag == 0
        plot(y, lw=0.0, ms=3, markershapes = :circle, markerstrokewidth=0, label="Target", xlabel="sample", ylabel="Glucose", ylims=(-1.5,1.5))
        plot!(ŷ, lw=0.0, ms=3, markershapes = :circle, markerstrokewidth=0, label="Prediction")
    elseif flag == 2
        plot(y, lw=0.0, ms=3, markershapes = :circle, markerstrokewidth=0, label="Target", xlabel="sample", ylabel="Glucose", ylims=(-1.5,1.5))
        plot!(ŷ, lw=0.0, ms=3, markershapes = :circle, markerstrokewidth=0, label="Prediction")
        plot!(abs.(y-ŷ), lw=0.0, ms=3, markershapes = :circle, markerstrokewidth=0, label="Error")
    else
        plot(y, ŷ, lw=0.0, ms=3, markershapes = :circle, markerstrokewidth=0, xlabel="Target", ylabel="Prediction", legend=:none)
        plot!(-1:1, x->x, lw=2)
    end
end


function myPlot(y,ŷ; flag=0, label="Test")
    if flag == 0
        plot(y, lw=0.0, ms=3, markershapes = :circle, markerstrokewidth=0, label="Target", xlabel="sample", ylabel="Glucose", ylims=(-1.5,420))
        plot!(ŷ, lw=0.0, ms=3, markershapes = :circle, markerstrokewidth=0, label="Prediction")
    elseif flag == 2
        plot(y, lw=0.0, ms=3, markershapes = :circle, markerstrokewidth=0, label="Target", xlabel="sample", ylabel="Glucose", ylims=(-1.5,420))
        plot!(ŷ, lw=0.0, ms=3, markershapes = :circle, markerstrokewidth=0, label="Prediction")
        plot!(abs.(y-ŷ), lw=0.0, ms=3, markershapes = :circle, markerstrokewidth=0, label="Error")
    else
        plot(y, ŷ, lw=0.0, ms=3, markershapes = :circle, markerstrokewidth=0, xlabel="Target", ylabel="Prediction", legend=:none)
        plot!(-1:420, x->x, lw=2)
    end
end

function myPlots(train_data, train_labels, mean_loss, test_mean_loss,
        simple_rnn; idxIn=0, svfig=false, filename="plot.png", timestamp=false, modelname="Gen.bson", printplots=true, pred=2, using_enc=0)

    idxIn == 0 ? idx = rand(3:size(datA,1)) : idx = idxIn
    if size(datA[idx],1) > 20
        test_data, test_labels = genVs(datA[idx], dt=pred)
    else
        while size(datA[idx],1) < 20
            @warn "No data"
            idx = rand(2:size(datA,1))
            test_data, test_labels = genVs(datA[idx], dt=pred)
        end
    end
    p1 = myPlot(test_labels, getPred(test_data, test_labels, using_enc=using_enc), flag=1)
    p2 = myPlot(test_labels, getPred(test_data, test_labels, using_enc=using_enc), flag=0)

    p3 = myPlot(train_labels, getPred(train_data, train_labels, using_enc=using_enc), flag=1)
    p4 = myPlot(train_labels, getPred(train_data, train_labels, using_enc=using_enc), flag=0)

    p5 = plot( mean_loss, label="Train", xlabel="Epoch", ylabel="MAE", lw=2, yscale=:identity)
    p5 = plot!( test_mean_loss, label="Test", xlabel="Epoch", ylabel="MAE", lw=2, yscale=:identity)

    f = plot(p1, p2, p3, p4, p5, p5, layout = (3,2), size = (750,750))

    printplots ? (display(f); return f;) : nothing
end


function plotCNN(mean_loss, nn; data_test=false)
    f1 = plot( mean_loss, label="Train", xlabel="Epoch", ylabel="MAE", lw=2, yscale=:identity)
    f = plot(f1)
    if data_test != false
        f2 = plot(invBGriskFunc.(data_test[2][1,1+9*278:10*278] .* √10), lw=1, legend=false)
        f2 = plot!( invBGriskFunc.(vcat(nn(data_test[1][:,:,:,1+9*278:10*278])...) .* √10), lw=1)
        f2 = plot!(abs.(invBGriskFunc.(data_test[2][1,1+9*278:10*278] .* √10) .- invBGriskFunc.(vcat(nn(data_test[1][:,:,:,1+9*278:10*278])...) .* √10)), ms=4, s=:auto, markershapes = :auto, lw=0.5, markerstrokewidth=0)
        f = plot(f1,f2,layout=(2,1))
    end
    display(f)
end


function plotVAE(mean_loss, nn; data_test=false)
    f1 = plot( mean_loss, label="Train", xlabel="Epoch", ylabel="MAE", lw=2, yscale=:identity)
    f = plot(f1)
    if data_test != false
        f2 = plot(invBGriskFunc.(data_test[2][1,1+9*278:10*278] .* √10), lw=1, legend=false, c=:black)
        sample = vcat([nn(dataA_test[1][:,:,:,1+9*278:10*278])[end] for i in 1:100]...)
        sampleM = mean(sample, dims=1)
        sampleSTD = std(sample, dims=1)
        f2 = plot!( invBGriskFunc.(vcat(sampleM...) .* √10), lw=1, ribbon=sampleSTD)
        f2 = plot!(abs.(invBGriskFunc.(data_test[2][1,1+9*278:10*278] .* √10) .- invBGriskFunc.(vcat(sampleM...) .* √10)), ms=4, s=:auto, markershapes = :auto, lw=0.5, markerstrokewidth=0)
        f = plot(f1,f2,layout=(2,1))
    end
    display(f)
end
