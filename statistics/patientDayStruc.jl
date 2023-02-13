mutable struct CGM
    date
    data
    stand_data
    count
    BGrisk
end

α = 1.084
β = 5.381
γ = 1.509
# BGriskFunc(x) = 10 * (γ * ((log(x))^α - β) )^2
BGriskFunc(x) = γ * ((log(x))^α - β)

function createPerDayStruc(df, pt_id)
    dataPerdate = Dict()
    dataPerdate[pt_id] = []

    for k in 1:size(df,1)
        removeDaysList = []
        alldates = unique(Dates.yearmonthday.(df[k][1:end,1]))

        ar = Vector{CGM}(undef, size(alldates,1))

        for (i,date) in enumerate(alldates)
            tmpdata = df[k][Dates.yearmonthday.(df[k][:,1]) .== [date for i in 1:size(df[k],1)],:]
            tmpdata[!, "BGrisk"] = BGriskFunc.(tmpdata[:,2])
            stand_data = isnan(std(tmpdata[:,2])) ? (tmpdata[:,2] .- mean(tmpdata[:,2])) : ( (tmpdata[:,2] .- mean(tmpdata[:,2])) ./ std(tmpdata[:,2]) )
            # stand_data = isnan(std(tmpdata[:,2])) ? (tmpdata[:,2] .- tmpdata[1,2]) : ( (tmpdata[:,2] .- tmpdata[1,2]) ./ std(tmpdata[:,2]) )
            ar[i] = CGM(Date(date[1], date[2], date[3]), tmpdata, stand_data, size(tmpdata,1), BGriskFunc.(tmpdata[:,2]))
            if size(unique(tmpdata[:,2]),1) == 1
                append!(removeDaysList,i)
            end
        end
        deleteat!(ar, removeDaysList)
        dataPerdate[pt_id] = vcat(dataPerdate[pt_id],ar)
    end
    dataPerdate
end
