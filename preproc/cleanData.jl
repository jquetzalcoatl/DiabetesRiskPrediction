# using DataFrames, CSV, Plots, Interact, Statistics, Dates, JSON

# include(pwd() * "/Diabetes/preproc/dataParsing.jl")
# This module is for data cleaning. Imputes values outside the range 39-400.
# This is done by doing a linear interpolation between the prior and the next
# measurment of a value outside the permited range.
# If the next measurement is also outside the bounds, the function calls itself
# and iterates.

LOWBOUND = 39
HIGHBOUND = 400

function imputeValue(df, i, i_prior, str=:min)
    if str == :min
        if df[i+1,2] >= LOWBOUND
            df[i,2] = floor((df[i+1,2] + df[i_prior,2])/2)
        else
            imputeValue(df, i+1, i_prior)
            df[i,2] = floor((df[i+1,2] + df[i_prior,2])/2)
        end
    else
        if df[i+1,2] <= HIGHBOUND
            df[i,2] = floor((df[i+1,2] + df[i_prior,2])/2)
        else
            imputeValue(df, i+1, i_prior, :max)
            df[i,2] = floor((df[i+1,2] + df[i_prior,2])/2)
        end
    end
end

function imputeDF(df)
    if df[1,2] < LOWBOUND || df[1,2] > HIGHBOUND
        deleteat!(df , collect(1:findall(x->x >= LOWBOUND && x <= HIGHBOUND,df[:,2] )[1]-1))
    end

    if df[end,2] < LOWBOUND || df[end,2] > HIGHBOUND
        deleteat!(df , collect(findall(x->x >= LOWBOUND && x <= HIGHBOUND,df[:,2] )[end]+1:size(df,1)))
    end

    l = findall(x -> x < LOWBOUND, df[:,2])
    for i in l
        imputeValue(df, i, i-1)
    end
    @assert size(findall(x -> x < LOWBOUND, df[:,2]),1) == 0

    l = findall(x -> x > HIGHBOUND, df[:,2])
    for i in l
        imputeValue(df, i, i-1, :max)
    end
    @assert size(findall(x -> x > HIGHBOUND, df[:,2]),1) == 0
end

function imputeAr(df)
    for i in 1:size(df,1)
        imputeDF(df[i])
    end
end
