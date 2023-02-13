using DataFrames, CSV, Plots, Interact, Statistics, Dates, JSON

# cd(dirname(pwd()))

include(pwd() * "/Diabetes/preproc/dataParsing.jl")
# include(pwd() * "/Diabetes/preproc/cleanData.jl")

PTS

pt = sort(collect(keys(PTS)))[19]
df = loadCSV(pt, parse_dates=true, linear_interpolation=true)

# Plot timeseries
function plotPtData(df, pt_id)
    plot()
    for i in 1:size(df,1)
        plot!(df[i][:,1], df[i][:,2], label="File $i")
    end
    plot!(xlabel="Date", ylabel="Glucose", frame=:box, title=pt_id, xrot=0)
end

@manipulate for pt in collect(keys(PTS))
    try
        df = loadCSV(pt)
        plotPtData(df, pt)
    catch
        @warn "Something went wrong with patient $pt"
    end
end

@manipulate for pt in collect(keys(PTS))
    try
        df1 = loadCSV(pt, linear_interpolation=false)
        df2 = loadCSV(pt, linear_interpolation=true)
        histogram(vcat([df1[i][:,2] for i in 1:size(df1,1)]...), bins=300, normalize=true, linewidth = 0, label="nonImputed")
        histogram!(vcat([df2[i][:,2] for i in 1:size(df2,1)]...), bins=300, normalize=true, linewidth = 0, fillalpha=0.6, label="linear interpolation")
        plot!(frame=:box, xlabel="Glucose", ylabel="Freq", title="Counts = $(size(vcat([df1[i][:,2] for i in 1:size(df1,1)]...),1))")
    catch
        @warn "Something went wrong with patient $pt"
    end
end
m = mean(vcat([df2[i][:,2] for i in 1:size(df2,1)]...))
s = std(vcat([df2[i][:,2] for i in 1:size(df2,1)]...))

ar = vcat([df[i][:,2] for i in 1:size(df,1)]...)
@manipulate for w in 1:200, b in 20:100
    ar2 = [mean( ar[1+i:w+i]) for i in 1:w:size(ar,1)-w]
    histogram(ar2, normalize=true, bins=b)
    m, s = mean(ar2), std(ar2)
    plot!(m-4*s:0.1:m+4*s, x->1/√(2π*s^2) * exp(-(x-m)^2/(2*s^2)))
end


# Get simple stats
function getStat()
    stats = Dict()
    errorData = []
    for pt in collect(keys(PTS))
        try
            df = loadCSV(pt, parse_dates=false)
            ar = vcat([df[i][:,2] for i in 1:size(df,1)]...)
            stats[pt] = Dict(:mean => mean(ar), :std => std(ar), :count => size(ar,1), :files => size(df,1), :min => minimum(ar), :max  => maximum(ar), :q1 => quantile(ar, .25), :q2 => quantile(ar, .5), :q3 => quantile(ar, .75))
        catch
            @warn "Something wrong with pt $pt"
            errorData = vcat(errorData, pt)
        end
    end
    stats, errorData
end

stats, errorData = getStat()

# Plot simple stats
function plotStats(metrics)
    idx = collect(keys(stats))[sortperm(collect(keys(stats)))]
    
    plot([stats[pt][:mean] for pt in idx], ribbons=[stats[pt][:std] for pt in idx], fillalpha=0.3, markershapes=:circle, label="mean ± std", lw=2, ms=5, markerstrokewidth=0, xrot=0, margin=10mm)
    
    for metric in metrics
        plot!([stats[pt][metric] for pt in idx], markershapes=:circle, lw=2, ms=5, markerstrokewidth=0, label=String(metric))
    end
    plot!(frame=:box, xlabel="Patients", ylabel="Glucose", legend=:outertopright)
end

@manipulate for m in collect(keys(stats["49551394"]))
    plotStats([m])
end

plot(sort([stats[pt][:count] for pt in collect(keys(stats))]), yscale=:log, markershapes=:circle, lw=2, ms=5, markerstrokewidth=0)

fig = plotStats([:min, :max])

mean([stats[i][:mean] for i in collect(keys(stats))])
std([stats[i][:mean] for i in collect(keys(stats))])
[stats[i][:std] for i in collect(keys(stats))]


savefig(fig, "/Users/javier/Dropbox/Aplicaciones/Overleaf/GlucosePrediction/Figs/Glucose_mean-std-min-max_vs_pt.png")

# Pt with highest Variation Coef = std/mean
argmax([stats[i][:std]/stats[i][:mean] for i in collect(keys(stats))])

pt = collect(keys(stats))[76]
df = loadCSV(pt, parse_dates=true, linear_interpolation=true)

pwd()

####
#### Separate by days
include(pwd() * "/Diabetes/statistics/patientDayStruc.jl")

# Plots data per day
pt_id = collect(keys(PTS))[1] 
df = loadCSV(pt_id, linear_interpolation=true)
test = createPerDayStruc(df, pt_id)


@manipulate for imin in 1:50, imax in 2:300
    plot()
    for i in imin:imax
        plot!(Time.(test[pt_id][i].data[:,1]), test[pt_id][i].data[:,3], markershapes=:circle, lw=2, ms=5, markerstrokewidth=0)
    end
    plot!(xlabel="Date", ylabel="Glucose", frame=:box, legend=false) #title=pt_id, xrot=0
end

using HDF5, JLD
include(pwd() * "/Diabetes/statistics/sampleAv.jl")

@time d, ptTempMean, ptTempSD = getSampleAv()


save(pwd() * "/Diabetes/data/PatientBGRiskDataPerDayAvg.jld", "data", d)
d = load(pwd() * "/Diabetes/data/PatientStandDataPerDayAvg.jld")["data"]
d = load(pwd() * "/Diabetes/data/PatientBGRiskDataPerDayAvg.jld")["data"]


@manipulate for pt in sort(collect(keys(d)))
    f1 = plot([Time(00,00,00) + Dates.Minute(m) for m in 0:5:24*60-5], d[pt][:av], ylabel="Glucose", xlabel="Time", label="Average Glucose Fluctuation" )
    f2 = plot([Time(00,00,00) + Dates.Minute(m) for m in 0:5:24*60-5], d[pt][:sd], ylabel="Glucose", xlabel="Time", label="Glucose Fluctuation Standard Deviation")
    plot(f1,f2, layout=(2,1))
end

begin
    f = plot()
    for pt in sort(collect(keys(d)))
        f = plot!([Time(00,00,00) + Dates.Minute(m) for m in 0:5:24*60-5], d[pt][:av], ylabel="Average Glucose Risk Score", xlabel="Time", legend=:none, ribbons=d[pt][:sd], fillalpha=0.015)
    end
    f = plot!(frame=:box)
end
savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/GlucosePrediction/Figs/Glucose_RiskScore_vars.png")

begin
    plot()
    for pt in sort(collect(keys(d)))
        f1 = plot!([Time(00,00,00) + Dates.Minute(m) for m in 0:5:24*60-5], d[pt][:sd], ylabel="Glucose Risk Score SD", xlabel="Time", legend=:none)
    end
    plot!()
end

###


mCovMat = [mean((d[pt1][:av] .- mean(d[pt1][:av])) .* (d[pt][:av] .- mean(d[pt][:av]))) ./ (std(d[pt1][:av]) * std(d[pt][:av])) for pt in sort(collect(keys(d))), pt1 in sort(collect(keys(d)))]

heatmap(mCovMat)

sdCovMat = [mean((d[pt1][:sd] .- mean(d[pt1][:sd])) .* (d[pt][:sd] .- mean(d[pt][:sd]))) ./ (std(d[pt1][:sd]) * std(d[pt][:sd])) for pt in sort(collect(keys(d))), pt1 in sort(collect(keys(d)))]

heatmap(sdCovMat)


using LinearAlgebra

λ, ν = eigen(mCovMat)

plot(λ, st=:scatter, legend=:none)

plot(abs.(ν[:,end]))




##
# Example
probingData = [cos(2π*1*x + b) + rand() for x in 0:0.01:1, b in 1:0.1:10]

plot(probingData[:,20])

mCovMat = [mean((probingData[:,pt1] .- mean(probingData[:,pt1])) .* (probingData[:,pt] .- mean(probingData[:,pt]))) ./ (std(probingData[:,pt1]) * std(probingData[:,pt])) for pt in 1:91, pt1 in 1:91]

heatmap(mCovMat)

λ, ν = eigen(mCovMat)

plot(λ, st=:scatter, legend=:none)

plot(ν[:,end])
##

mCovMat = [ mean( ([d[pt1][:av][i] for pt1 in sort(collect(keys(d)))] .- mean([d[pt1][:av][i] for pt1 in sort(collect(keys(d)))])) .* ([d[pt1][:av][j] for pt1 in sort(collect(keys(d)))] .- mean([d[pt1][:av][j] for pt1 in sort(collect(keys(d)))])) ) ./ (std([d[pt1][:av][i] for pt1 in sort(collect(keys(d)))]) * std([d[pt1][:av][j] for pt1 in sort(collect(keys(d)))]) + 10^(-12)) for i in 1:288, j in 1:288 ]

mCovMat = [ mean( ([d[pt1][:av][i] for pt1 in sort(collect(keys(d)))] .- mean([d[pt1][:av][i] for pt1 in sort(collect(keys(d)))])) .* ([d[pt1][:av][j] for pt1 in sort(collect(keys(d)))] .- mean([d[pt1][:av][j] for pt1 in sort(collect(keys(d)))])) ) for i in 1:288, j in 1:288 ]

hm = heatmap([Time(00,00,00) + Dates.Minute(m) for m in 0:5:24*60-5], [Time(00,00,00) + Dates.Minute(m) for m in 0:5:24*60-5], mCovMat[1:287, 1:287], c = cgrad(:lighttest, 20, categorical = false, scale = :linear), yrot=-45, topmargin = 5Plots.mm)
savefig(hm, "/Users/javier/Dropbox/Aplicaciones/Overleaf/GlucosePrediction/Figs/heatmap.png")

sum(mCovMat[1:287, 1:287] .== 0)

λ, ν = eigen(mCovMat[1:287, 1:287])

plot(λ, st=:scatter, legend=:none)

@manipulate for i in 1:size(ν,2)
    plot([Time(00,00,00) + Dates.Minute(m) for m in 0:5:24*60-10], ν[:,i], ylim=(-0.2,0.2))
end

plot([Time(00,00,00) + Dates.Minute(m) for m in 0:5:24*60-10], ν[:,end-1] .+ ν[:,end-2] .+  ν[:,end-3])
# plot([Time(00,00,00) + Dates.Minute(m) for m in 0:5:24*60-5], ν[:,end-1])

@manipulate for i in size(ν,2)-1:-1:278
    plot()
    for j in size(ν,2)-1:-1:i
        plot!([Time(00,00,00) + Dates.Minute(m) for m in 0:5:24*60-10], ν[:,j], lw=2, label="λ = $(round(λ[j], sigdigits=3))")
    end
    plot!([Time(00,00,00) + Dates.Minute(m) for m in 0:5:24*60-10], sum([ν[:,j] for j in size(ν,2)-1:-1:i]), ylim=(-0.5,0.5), lw=3, c=:black, label="Sum")
    plot!(xlabel="Time", ylabel="Principal Components", frame=:box)
end

begin
    i = 281
    f = plot()
    for j in size(ν,2)-1:-1:i
        f = plot!([Time(00,00,00) + Dates.Minute(m) for m in 0:5:24*60-10], ν[:,j], lw=2, label="λ = $(round(λ[j], sigdigits=3))")
    end
    f = plot!([Time(00,00,00) + Dates.Minute(m) for m in 0:5:24*60-10], sum([ν[:,j] for j in size(ν,2)-1:-1:i]), ylim=(-0.5,0.5), lw=3, c=:black, label="Sum")
    f = plot!(xlabel="Time", ylabel="Principal Components", frame=:box)
end

savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/GlucosePrediction/Figs/PrincComp.png")


###

######################Working with BGRisk
# Plots data per day
pt_id = collect(keys(PTS))[3] 
df = loadCSV(pt_id, linear_interpolation=true)
test = createPerDayStruc(df, pt_id)

f = histogram(vcat([test[pt_id][i].BGrisk / √10 for i in 1:size(test[pt_id],1)]...), bins=70, normalize=true, linewidth = 0, xlabel="Glucose Risk Score", ylabel="PDF", title="Counts = $(size(vcat([test[pt_id][i].BGrisk for i in 1:size(test[pt_id],1)]...),1))", size=(700,500), frame=:box, legend=:none)

PATHToHist = pwd() * "/Diabetes/figs/"
savefig(f, PATHToHist * "hist_BGRiskScore$(pt_id).png")

include(pwd() * "/Diabetes/ML/parseTimeSeries.jl")

genVs(collect(1:15))




# Imputed percentage

PTS

# pt = sort(collect(keys(PTS)))[1]
# pt = collect(keys(PTS))[2]
# pt = collect(keys(PTS))[109]
# df = loadCSV(pt, parse_dates=false, linear_interpolation=false)

prct = zeros(size(collect(PTS),1), 2)
for (i,pt) in enumerate(collect(keys(PTS)))
    df = loadCSV(pt, parse_dates=false, linear_interpolation=false)
    oorVal = 0
    for j in 1:size(df,1)
        oorVal = oorVal + sum(df[j][:,2] .< 39) + sum(df[j][:,2] .> 400)
    end
    prct[i,1] = oorVal / sum([size(df[h],1) for h in 1:size(df,1)])
    prct[i,2] = sum([size(df[h],1) for h in 1:size(df,1)])
    
end

prct

f = plot(sort(prct[:,1]) .* 100, st=:scatter, frame=:box, ms=5, s=:auto, legend=false,
    lw=0.3, markerstrokewidth=0, xlabel="Participants ranked by imputed data %", ylabel="Imputed data %")

# plot!(prct[sortperm(prct[:,1]),2] ./ maximum(prct[:,2]))

savefig(f, "/Users/javier/Library/CloudStorage/Dropbox/Aplicaciones/Overleaf/GlucosePrediction/Figs/ImputedPerc.png")



consecArr = zeros(size(collect(PTS),1), 4)
for (j,pt) in enumerate(collect(keys(PTS)))
    df = loadCSV(pt, parse_dates=false, linear_interpolation=false)
    ar = vcat([df[i][:,2] for i in 1:size(df,1)]...)
    l = sign.(convert(Array{Int}, (ar .< 39) .+ (ar .> 400)))
    lstring = join(string.(l))
    arString = split(lstring, "0")
    arString[length.(arString) .> 0]

    consecutiveImp = length.(arString[length.(arString) .> 0])
    if size(consecutiveImp,1) > 0
        if size(consecutiveImp,1) > 1
            consecArr[j,:] = [mean(consecutiveImp) std(consecutiveImp) maximum(consecutiveImp) minimum(consecutiveImp)]
        else
            consecArr[j,:] = [mean(consecutiveImp) 0 maximum(consecutiveImp) minimum(consecutiveImp)]
        end
    else
        consecArr[j,:] = [0 0 0 0]
    end
end


f = plot(consecArr[sortperm(prct[:,1]),1], yerrors=consecArr[sortperm(prct[:,1]),2], st=:scatter, frame=:box, ms=5, 
    s=:auto, legend=false,lw=0.3, markerstrokewidth=0, xlabel="Participants ranked by imputed data %", 
    ylabel="Mean number of consecutive \n imputed values")

savefig(f, "/Users/javier/Library/CloudStorage/Dropbox/Aplicaciones/Overleaf/GlucosePrediction/Figs/ImputedConsec.png")

f = plot(prct[sortperm(prct[:,1]),2] .* 100, st=:scatter, frame=:box, ms=5, s=:auto, legend=false,
    lw=0.3, markerstrokewidth=0, xlabel="Participants ranked by imputed data %", ylabel="Number of measurements", yscale=:log10)

savefig(f, "/Users/javier/Library/CloudStorage/Dropbox/Aplicaciones/Overleaf/GlucosePrediction/Figs/NumOfMeas.png")