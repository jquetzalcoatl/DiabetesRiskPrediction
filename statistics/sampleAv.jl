using DataFrames, CSV, Statistics, Dates
include(pwd() * "/Diabetes/statistics/patientDayStruc.jl")
include(pwd() * "/Diabetes/preproc/dataParsing.jl")

# Compute time average over day samples
function temporalMean(pt_id)
    df = loadCSV(pt_id)
    test = createPerDayStruc(df, pt_id)

    CGM_m = zeros(288)
    for (i,m) in enumerate(0:5:24*60-5)
        Δt = Time(00,00,00) + Dates.Minute(m)
        # @info Δt, i
        c=0
        for d in 1:size(test[pt_id],1)
            bb = (Time.(test[pt_id][d].data[:,1]) .< Δt + Dates.Minute(5)) .* (Time.(test[pt_id][d].data[:,1]) .>= Δt)
            
            # if size(test[pt_id][d].data[bb,2],1) >= 1
            if size(test[pt_id][d].stand_data[bb],1) >= 1
                # CGM_m[i] = CGM_m[i] + mean(test[pt_id][d].data[bb,2])
                # CGM_m[i] = CGM_m[i] + mean(test[pt_id][d].stand_data[bb])
                CGM_m[i] = CGM_m[i] + mean(test[pt_id][d].BGrisk[bb])
                c = c + 1
            end

        end
        CGM_m[i] = c == 0 ? CGM_m[i] : CGM_m[i]/c
    end
    CGM_m
end

function temporalSD(pt_id, CGM_m)
    df = loadCSV(pt_id)
    test = createPerDayStruc(df, pt_id)

    CGM_std = zeros(288)
    for (i,m) in enumerate(0:5:24*60-5)
        Δt = Time(00,00,00) + Dates.Minute(m)
        # @info Δt, i
        c=0
        for d in 1:size(test[pt_id],1)
            bb = (Time.(test[pt_id][d].data[:,1]) .< Δt + Dates.Minute(5)) .* (Time.(test[pt_id][d].data[:,1]) .>= Δt)
            
            # if size(test[pt_id][d].data[bb,2],1) >= 1
            if size(test[pt_id][d].stand_data[bb],1) >= 1
                # CGM_m[i] = CGM_m[i] + mean(test["13029224"][d].data[bb,2])
                # CGM_std[i] = CGM_std[i] + (mean(test[pt_id][d].data[bb,2]) - CGM_m[i])^2
                # CGM_std[i] = CGM_std[i] + (mean(test[pt_id][d].stand_data[bb]) - CGM_m[i])^2
                CGM_std[i] = CGM_std[i] + (mean(test[pt_id][d].BGrisk[bb]) - CGM_m[i])^2
                c = c + 1
            end

        end
        CGM_std[i] = c == 0 ? √(CGM_std[i]) : √(CGM_std[i]/c)
        # CGM_std[i] = √(CGM_std[i])
    end
    CGM_std
end

function getSampleAv()
    d = Dict()
    ptTempMean = zeros(288, size(collect(keys(PTS)),1))
    for (i,id) in enumerate(collect(keys(PTS))[1:2])
        try
            ptTempMean[:,i] = temporalMean(id)
            d[id] = Dict(:av => ptTempMean[:,i])
        catch
            @warn "issue with $id"
            d[id] = Dict(:av => zeros(288))
        end
    end
    ptTempSD = zeros(288, size(collect(keys(PTS)),1))
    for (i,id) in enumerate(collect(keys(PTS))[1:2])
        try
            ptTempSD[:,i] = temporalSD(id, ptTempMean[:,i])
            d[id][:sd] = ptTempSD[:,i]
        catch
            @warn "issue with $id"
            d[id][:sd] = zeros(288)
        end
    end
    d, ptTempMean, ptTempSD
end
