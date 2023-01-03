#using DataFrames, CSV, Plots, Interact, Statistics, Dates

function genVs(x; init=12, dt=1, MAXLENGTH=100, longTimeSeries=false, add_timestep=false)
    if add_timestep
        initialPoint = init*dt
    else
        initialPoint = init
    end
    if size(x,1) > initialPoint
        nothing
    else
        @warn "Not enough points..."
        return
    end
    # if init < 2
    #     init = 2
    #     @warn "init should be at least equal to 2. Setting init = 2"
    # end
    # c = sign((size(x,1)-initialPoint)/dt - floor((size(x,1)-initialPoint)/dt))

    # v = Vector{Array{Float32, 1}}(undef, Int(floor((size(x,1)-initialPoint)/dt) + c))
    # u = Vector{Float32}(undef, Int(floor((size(x,1)-initialPoint)/dt) + c))
    v = Vector{Vector{Array{Float32, 1}}}(undef, size(x,1)-initialPoint)
    u = Vector{Float32}(undef, size(x,1)-initialPoint)
    # for (j,i) in enumerate(reverse(size(x,1):-dt:initialPoint + 1))
    for (j,i) in enumerate(reverse(size(x,1):-1:initialPoint + 1))
        longTimeSeries ? (i > MAXLENGTH ? r = MAXLENGTH : r = i - 1) : (r = add_timestep ? initialPoint : initialPoint ) #(r = add_timestep ? rand(5*dt:initialPoint) : initialPoint )
        # v[j] = x[i-r:dt:i-dt]
        # v[j] = reverse(x[i-dt:-dt:i-r])
        if add_timestep
            xx = reverse(x[i-dt:-dt:i-r])
        else
            xx = reverse(x[i-dt:-1:i-r])
        end
        v[j] = [[xx[i]] for i in 1:size(xx,1)]
        u[j] = x[i]
    end
    # if i != size(x,1)
    #     v[end] = x[i-1 - rand(5:init):end-1]
    #     u[end] = x[end]
    # end
    v, u
end


function gen4CNN(x; init=12, dt=1, MAXLENGTH=100)

    initialPoint = init
    if size(x,1) >= initialPoint+dt
        nothing
    else
        @warn "Not enough points..."
        return
    end

    # v = Vector{Vector{Array{Float32, 1}}}(undef, size(x,1)-initialPoint-dt+1)
    v = Vector{Array{Float32, 1}}(undef, size(x,1)-initialPoint-dt+1)
    u = Vector{Float32}(undef, size(x,1)-initialPoint-dt+1)

    for (j,i) in enumerate(reverse(size(x,1)-dt:-1:initialPoint))
        r = initialPoint

        xx = reverse(x[i:-1:i-r+1])
        # @info xx
        # v[j] = [[xx[i]] for i in 1:size(xx,1)]
        v[j] = xx
        u[j] = x[i+dt]
    end
    v, u
end
