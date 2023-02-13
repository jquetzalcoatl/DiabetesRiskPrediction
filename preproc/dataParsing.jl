using DataFrames, CSV, Dates
# using Plots, Interact, Statistics, JSON

include(pwd() * "/Diabetes/preproc/patientToPathDict.jl")
include(pwd() * "/Diabetes/preproc/cleanData.jl")
datF = dateformat"y-m-dTH:M:S"
datF2 = dateformat"e u d H:M:S y"
datF3 = dateformat"y-m-d H:M:S"
datF4 = dateformat"m/d/y H:M:S"

function parseDate(str)
    try
        return DateTime(str[1:19], datF)
    catch
        try
            return DateTime(str[1:19], datF3)
        catch
            try
                return DateTime(str[1:19], datF4)
            catch
                try
                    return DateTime(split(str, " EST")[1] * split(str, " EST")[2], datF2)
                catch
                    try
                        return DateTime(split(str, " EDT")[1] * split(str, " EDT")[2], datF2)
                    catch
                        try
                            return DateTime(split(str, " CEST")[1] * split(str, " CEST")[2], datF2)
                        catch
                            try
                                return DateTime(split(str, " MDT")[1] * split(str, " MDT")[2], datF2)
                            catch
                                try
                                    return DateTime(split(str, " GMT")[1] * split(str, " GMT")[2][end-4:end], datF2)
                                catch
                                    try
                                        return DateTime(split(str, " CDT")[1] * split(str, " CDT")[2], datF2)
                                    catch
                                        try
                                            return DateTime(split(str, " PDT")[1] * split(str, " PDT")[2], datF2)
                                        catch
                                            try
                                                return DateTime(split(str, " PST")[1] * split(str, " PST")[2], datF2)
                                            catch
                                                try
                                                    return DateTime(split(str, " MST")[1] * split(str, " MST")[2], datF2)
                                                catch
                                                    try
                                                        return DateTime(split(str, " CET")[1] * split(str, " CET")[2], datF2)
                                                    catch
                                                        return DateTime(split(str, " AST")[1] * split(str, " AST")[2], datF2)
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

# Load dataframe
function loadCSV(pt_id; parse_dates=true, linear_interpolation=true)
    @info pt_id
    nmax = size(PTS[pt_id],1)
    df = Array{DataFrame}(undef, nmax)
    for n in 1:nmax
        if !isempty(readdir(PTS[pt_id][n], join=true))
            df[n] = CSV.read(readdir(PTS[pt_id][n], join=true)[end], DataFrame, header=["TimeStamp"])
            df[n] = df[n][df[n][:,1] .!= "null", :]

            if typeof(df[n][1,2]) == String7
                df[n][findall(x->x == " ", df[n][:,2]),2] .= " null"
                df[n] = df[n][df[n][:,2] .!= " null", :]
                df[n][!,"Glucose"] = parse.(Int, df[n][:,2])
            else
                df[n][!, "Glucose"] = df[n][:,2]
            end
            df[n] = df[n][:,[1,3]]
            
            if parse_dates
                df[n].TimeStamp = parseDate.(df[n][:,1])
            end
            unique!(df[n], :TimeStamp)
            sort!(df[n],1)
        else
            @warn "Nothing in $(PTS[pt_id][n]) $n "
            df[n] = DataFrame()
        end
    end
    deleteat!(df,findall(x->isempty(x)==true,df))
    if linear_interpolation
        imputeAr(df)
    end
    df
end
