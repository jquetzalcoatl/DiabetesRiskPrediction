# If you downloaded the directories one by one, you might need to run this.

# Set PATH to directories
PATH = "/Volumes/ExtDisk/OpenAPS/"

# Get dir names
dirs = readdir(PATH)

# There are some hidden dirs which start with ._ What follows gets rid of them.
if !isempty(dirs[contains.(dirs, "._")])
    rm.(PATH .* dirs[contains.(dirs, "._")])
end

# Get dir names again.
dirs = readdir(PATH)


pts = Dict()
for d in dirs[contains.(dirs, "direct-sharing")]
    try
        patient = split(readdir(PATH * d)[1], "_")[2]
        pts[patient] = d
    catch
        nothing
    end
end

if !isempty(pts)
    for tmp in keys(pts)
        tmpVal = pts[tmp]
        @info tmp, tmpVal
        mkpath(PATH * tmp * "/direct-sharing-31")
        for FileOrDir in readdir(PATH * tmpVal)
            mv(PATH * tmpVal * "/" * FileOrDir, PATH * tmp * "/direct-sharing-31/" * FileOrDir)
        end
        rm(PATH * tmpVal)
    end
end

for dir in dirs
    try
        newDirs = readdir(PATH * dir * "/direct-sharing-31/")
        if !isempty(newDirs[contains.(newDirs, "._")])
            rm.(PATH * dir * "/direct-sharing-31/" .* newDirs[contains.(newDirs, "._")])
        end
    catch
        @warn "$dir does not contain direct-sharing-31 directory"
    end
end
