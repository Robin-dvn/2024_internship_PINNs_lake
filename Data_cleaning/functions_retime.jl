# # `retime` function

#TODO: check this function. It is not very elegant.
"""
    reasonable_impute(X::AbstractVector{T}, max_length)
Impute `missing` with an impute_method from [Impute.jl](https://github.com/invenia/Impute.jl) only if the `missing` sequence is of length ≤ than `max_length` (and not starting nor finishing the vector)
"""
function reasonable_impute(X::AbstractVector{T}, max_length; impute_method=Interpolate()) where {T<:Union{Real,Missing}}
    seq_type, seq_len = rle(oneOrmissing.(X))
    agg_type = Union{Real,Missing}[]
    for i in eachindex(seq_type)
        if seq_type[i] === missing && (i == firstindex(seq_type) || seq_len[i] > max_length || i == lastindex(seq_type))
            push!(agg_type, missing)
        else
            push!(agg_type, 1)
        end
    end
    agg_len = Int[]
    s = 0
    for i in eachindex(seq_type)
        if agg_type[i] === missing
            append!(agg_len, seq_len[i])
            s = 0 #? useless
        else
            s += seq_len[i]
            if i == lastindex(seq_len)
                append!(agg_len, s)
            elseif agg_type[i+1] === missing
                append!(agg_len, s)
                s = 0
            end
        end
    end
    intervals = [range(r...) for r in zip(cumsum(agg_len) - agg_len .+ 1, cumsum(agg_len))]
    @views vcat([!(all(ismissing, X[iv])) ? impute(X[iv], impute_method) : X[iv] for iv in intervals]...)
end

reasonable_impute(X::AbstractVector{<:Real}, max_length; impute_method=Interpolate()) = X
reasonable_impute(X::AbstractVector{<:Missing}, max_length; impute_method=Interpolate()) = X

oneOrmissing(x::Missing) = missing
oneOrmissing(x) = 1

fORmissing(f, x::AbstractVector{<:Missing}) = missing
fORmissing(f, x::AbstractVector) = all(ismissing.(x)) ? missing : f(filter(!ismissing, x))

#TODO: In Matlab values are interpolated (how does it handle aggregated + interpolation??)
#TODO: write test
#TODO: all could be in one @chain with rename and @transform?
"""
    retime(df_origin::DataFrame, new_Δt; impute_method = Interpolate(), start = :round)
`retime` function similar to the [Matlab `retime`](https://fr.mathworks.com/help/matlab/ref/timetable.retime.html). 
It creates a regular timestamp at every `Δt`. Multiple data point in `[t, t+Δt]` are aggregated with `agg_method` e.g. `mean`, `last`. 
It also impute missing values with `impute_method` when the duration of `missing` values is smaller or equal to `max_Δt` (in which case it leaves the `missing`).
"""
function retime(df_origin::DataFrame, Δt; impute_method=Interpolate(), start=:round, agg_method=mean, max_Δt=10 * Δt)

    res = retime_agg(df_origin, Δt; start=start, agg_method=agg_method)

    max_length = max_Δt ÷ Δt
    for col in names(res, Not(:date))
        @transform!(res, $col = reasonable_impute($col, max_length; impute_method=impute_method))
    end
    return res
end

function retime_agg(df_origin::DataFrame, Δt; start=:round, agg_method=mean)
    if start == :round || start == :floor
        newTime = floor(df_origin.date[begin], Δt):Δt:ceil(df_origin.date[end], Δt)
    elseif start == :unchanged
        newTime = df_origin.date[begin]:Δt:ceil(df_origin.date[end], Δt)
    elseif start isa Date
        newTime = start:Δt:ceil(df_origin.date[end], Δt)
    end
    df = outerjoin(DataFrame(:date => newTime), df_origin, on=:date)
    return DataFramesMeta.@chain df begin
        @transform(:time_cat = (:date .- newTime[1]) .÷ Δt)
        groupby(:time_cat)
        combine(names(df, Not(:date)) .=> x -> fORmissing(agg_method, x), renamecols=false)
        @transform(:date = :time_cat * Δt .+ newTime[1])
        select(:date, Not(:date, :time_cat))
    end
end

"""
    reasonableInterpolation(X, t::AbstractVector, method, max_Δt; kwargs...)
Interpolate continuously data (support `missing` values)  with an interpolation methods from [DataInterpolations.jl](https://github.com/SciML/DataInterpolations.jl) only if the time gap is smaller than `max_Δt`. It returns a vector of all cuts where interpolation was reasonable.
"""
function reasonableInterpolation(X::AbstractVector, t::AbstractVector, method, max_Δt; kwargs...)
    @assert length(X) == length(t)
    idx_missings_X = (!ismissing).(X)
    t_wo_missing = view(t, idx_missings_X) # inspired by https://discourse.julialang.org/t/how-to-perform-an-argmax-argmin-on-a-subset-of-a-vector/70569
    cuts = first(parentindices(t_wo_missing))[findall(diff(t_wo_missing) .> max_Δt)]
    N_cuts = length(cuts)
    if N_cuts == 0
        return [method(X, t; kwargs...)]
    else
        append!(cuts, length(t))
        idx_ranges =
            map(enumerate(cuts)) do (i, idx_cut)
                idx_start = i == 1 ? 1 : (cuts[i-1] + 1)
                range(idx_start, idx_cut)
            end
        return [method(X[idxs], t[idxs]; kwargs...) for idxs in idx_ranges]
    end
end

"""
    reasonableInterpolation(X, t::AbstractVector, method, max_Δt; kwargs...)
Interpolate continuously data (support `missing` values)  with an interpolation methods from [DataInterpolations.jl](https://github.com/SciML/DataInterpolations.jl) only if the time gap is smaller than `max_Δt`. It returns a vector of all cuts where interpolation was reasonable.
"""
function reasonableInterpolation(X::AbstractVector{<:Real}, t::AbstractVector{<:Real}, method, max_Δt; kwargs...)
    @assert length(X) == length(t)
    cuts = findall(diff(t) .> max_Δt)
    N_cuts = length(cuts)
    if N_cuts == 0
        return [method(X, t; kwargs...)]
    else
        append!(cuts, length(t))
        idx_ranges =
            map(enumerate(cuts)) do (i, idx_cut)
                idx_start = i == 1 ? 1 : (cuts[i-1] + 1)
                range(idx_start, idx_cut)
            end
        return [method(X[idxs], t[idxs]; kwargs...) for idxs in idx_ranges]
    end
end
using Combinatorics

"""
    Median of Mean estimator of all (overlaping or not) subset of cardinality `J` of the array `X`.
"""
function MoM_full(X::AbstractArray, J::Integer)
    if length(X) ≤ J
        return mean(X)
    else
        C = multiset_combinations(X, J)
        return median([mean(c) for c in C])
    end
    # itsample(C, 15; replace = false, ordered = false)
end

# X = rand(4)
# J = 2

# MoM_full(X, 2)
# length(X) % J 

# iter = multiset_combinations(X, J)
# itsample(iter, 2; replace=false, ordered=false)
# # MoM_full(X, k)
# # collect(multiset_combinations(X, 2))

# # eltype(multiset_combinations(X, 2))

# # MoM_full(X, 3)
# # mean(X)

# X = 1:6
# k = 2
# iter = multiset_combinations(X, k)
# itsample(iter, 2; replace = false, ordered = false)

# unique(itsample(iter, 14, ordered = false)) # ask for more element than the 15 in the iter
# unique(itsample(iter, 14, algRSWRSKIP, ordered = false))
# X = 1:6
# k = 2
# iter = multiset_combinations(X, k)

# rs = ReservoirSample(Vector{Int}, 20)

# for x in iter
#     update!(rs, x)
# end

# value(rs) # this contains the sample values collected