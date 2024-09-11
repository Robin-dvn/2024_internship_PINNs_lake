#  # Packages

using Markdown

using Statistics
using CSV, DataFrames, Dates, DataFramesMeta
using JLD2
# using DateFormats
using OrderedCollections
using DataInterpolations, RegularizationTools

cd(@__DIR__)

if isfile("df_clean.csv")
    df_clean = CSV.read("df_clean.csv", DataFrame)  
else
    md"""
    For datacleaning
    """
    using DataInterpolations, RegularizationTools
    function dateformplot(date)
        d = Date(yearmonth(date)...)
        return string(monthabbr(d), "-", year(d) - 2000)
    end

    md"""
    # Importing data
    """

    path_aux_func = "C:/Users/metivier/Dropbox/PC (2)/Documents/Simulations/NeuralNet/time_serie_neural"
    #include(joinpath(path_aux_func, "function_data.jl"))
    include("functions_retime.jl")

    if gethostname() == "mistea-jojoba"
        file_path = "C:/Users/metivier/Dropbox/PC (2)/Documents/GitLab/lake_data_mistea/data_lake_creteil"
    elseif gethostname() == "mistea-torreya"
        file_path = "/home/mathieu/docs/data"
    elseif gethostname() == "mistea-sophora"
        file_path = "C:/Users/davenner/Documents/PINNs/PGNN-PINN-Lakes/PINNs/Apprentissage/DifferentialEquations/data"
    end

    file_lake = "2022_2024_Lake_ESP2/20221015_20240402_ESPWat.csv"
    file_meteo = "2022_2024_Lake_ESP2/20221015_20240402_ESPMet.csv"

    df_lake_full = CSV.read(joinpath(file_path, file_lake), DataFrame, dateformat="d/m/y HH:MM")
    df_meteo_full = CSV.read(joinpath(file_path, file_meteo), DataFrame, dateformat="d/m/y HH:MM", delim=";")

    md"""
    Unbiased Cyano and Chl to only have ≥0 values
    - Cyano: +0.19 
    - Chl: +0.26
    """

    @transform!(df_lake_full, :Cyano = :Cyano .- minimum(skipmissing(:Cyano)))
    @transform!(df_lake_full, :Chl = :Chl .- minimum(skipmissing(:Chl)))


    md"""
    Interpolation and imputing missing
    """
    Δdate = Dates.Hour(1)
    date_start_full = df_lake_full.date[1]
    t_lake_full = (df_lake_full.date - date_start_full) ./ Δdate
    inters_lake_temp_surf = reasonableInterpolation(df_lake_full.Tw_05, t_lake_full, QuadraticInterpolation, Hour(24) / Δdate)

    md"""
    We select only the third piece of the date
    """

    piece = 3
    date_start = df_lake_full[findfirst(inters_lake_temp_surf[piece].t[begin] .== t_lake_full), :date]
    df_lake = @subset(df_lake_full, :date .≥ date_start)
    t_lake = (df_lake.date - date_start) ./ Δdate
 

    date_month = DateTime(2023, 1):Month(1):DateTime(yearmonth(last(df_meteo_full.date))...)
    t_month = (date_month - date_start) ./ Δdate
    md"""
    Agg & Regulirize
    """

    derivative_order = 1
    length_block_MoM = 2
    λ_reg = 1.5
    # # Subsampling otherwise we get outofmemory()
    ## Do once
    # df_lake_agg = retime_agg(df_lake, diff(df_lake.date)[1]*5, start = :unchanged)
    df_lake_agg_MoM = retime_agg(df_lake, diff(df_lake.date)[1] * 5, start=:unchanged, agg_method=x -> MoM_full(x, length_block_MoM))
    t_lake_agg = (df_lake_agg_MoM.date - date_start) ./ Δdate

    tt = (dropmissing(df_lake_agg_MoM).date - date_start)./Δdate
    var_bio = map([:Cyano, :O2]) do col
        file = "reg_d_$(derivative_order)_lamb_1p5_agg_MoM_$(col).jld2"
        if isfile(file)
            @info "Loading $col"
            reg = load(file)["reg"]
        else
            @time "regularization $(col)" reg = RegularizationSmooth(df_lake_agg_MoM[:, col], t_lake_agg, derivative_order; λ=λ_reg, alg=:fixed)
            jldsave(file; Δdate, start_date = date_start, reg)
        end
        return reg
    end
    md"""
    We put in the aggregated DataFrame the regulirized quantities Cyano, O2, Chl. The temperatures are interpolated at the few missing sites.
    The output is a regulirly spaced DataFrame with no missing and where most outliers have be done
    """

    temperatures = [BSplineInterpolation(df_lake_agg_MoM[:,c], t_lake_agg, 2, :ArcLen, :Average) for c in names(df_lake, r"^Tw")]

    df_clean = OrderedDict([:date, :t, :Cyano, :O2, Symbol.(names(df_lake, r"^Tw"))...] .=>  [df_lake_agg_MoM.date, t_lake_agg, [r.(t_lake_agg) for r in var_bio]..., [TT(t_lake_agg) for TT in temperatures]...]) |> DataFrame
    CSV.write("df_clean.csv", df_clean)
    # CSV.write("df_clean_std.csv", df_std)
end

df_std = @chain df_clean begin
    @transform(:Cyano = :Cyano/std(:Cyano))
    @transform(:O2 = :O2/std(:O2))
    # @transform(:Chl = :Chl/std(:Chl))
    @transform(:Tw_05 = :Tw_05/std(:Tw_05)) 
    @transform(:Tw_15 = :Tw_15/std(:Tw_05))
    @transform(:Tw_25 = :Tw_25/std(:Tw_05))
    @transform(:Tw_35 = :Tw_35/std(:Tw_05))
    @transform(:Tw_45 = :Tw_45/std(:Tw_05))
end