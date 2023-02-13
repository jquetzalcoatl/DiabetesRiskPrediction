# This script generates a dictionary where keys are patients and values
# are absolute paths to CSV entry data.
PATH = "/Volumes/ExtDisk/OpenAPS/"

dirs = readdir(PATH, join=false)

PTS = Dict()
for pt in dirs
    try
        local d = readdir(PATH * pt * "/direct-sharing-31", join=true)
        dSub = d[contains.(d, "entries")]
        PTS[pt] = dSub[contains.(dSub, "json_csv")]
    catch
        @warn "Something wrong with the paths for patient $pt"
    end
end
