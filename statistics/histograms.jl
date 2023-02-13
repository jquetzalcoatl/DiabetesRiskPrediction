using DataFrames, CSV, Plots, Interact, Statistics, Dates, JSON

include(pwd() * "/Diabetes/preproc/dataParsing.jl")
PATH = pwd() * "/Diabetes/figs/"

isdir(PATH) || mkpath(PATH)
for pt in sort(collect(keys(PTS)))
    try
        df1 = loadCSV(pt, linear_interpolation=false)
        df2 = loadCSV(pt, linear_interpolation=true)
        f = histogram(vcat([df1[i][:,2] for i in 1:size(df1,1)]...), bins=300, normalize=true, linewidth = 0, label="nonImputed")
        f = histogram!(vcat([df2[i][:,2] for i in 1:size(df2,1)]...), bins=300, normalize=true, linewidth = 0, fillalpha=0.6, label="linear interpolation")
        f = plot!(frame=:box, xlabel="Glucose", ylabel="PDF", title="Counts = $(size(vcat([df1[i][:,2] for i in 1:size(df1,1)]...),1))", size=(700,500))
        savefig(f, PATH * "hist_$(pt).png")
    catch
        @error "Something went wrong with patient $pt"
    end
end

