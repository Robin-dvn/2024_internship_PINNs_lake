using Markdown

using Statistics
using CSV, DataFrames, Dates, DataFramesMeta
using JLD2
# using DateFormats
using OrderedCollections
using DataInterpolations, RegularizationTools

cd(@__DIR__)

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
include("functions_retime.jl")

if gethostname() == "mistea-jojoba"
    file_path = "C:/Users/metivier/Dropbox/PC (2)/Documents/GitLab/lake_data_mistea/data_lake_creteil"
elseif gethostname() == "mistea-torreya"
    file_path = "/home/mathieu/docs/data"
elseif gethostname() == "mistea-sophora"
    file_path = "C:/Users/davenner/Documents/PINNs/PGNN-PINN-Lakes/PINNs/Apprentissage/DifferentialEquations/data"
end
using DataFramesMeta
using StatsPlots
using Dates

file_lake = "2023_2024_Lake_ESP2/20240723_ESPWat_with_corrected_sensor_data.csv"
df_lake_full = CSV.read(joinpath(file_path, file_lake), DataFrame)
start_date = DateTime("2023-01-04T00:00:00")
end_date = DateTime("2024-04-01T00:00:00")
filtered_new_df = @where(df_lake_full, :date .>= start_date .&& :date .<= end_date)
plt = Plots.plot()
@df filtered_new_df plot!(:date, :Chl, label = "new smooth")

md"""
Interpolation and imputing missing
"""

Δdate = Dates.Hour(1)
date_start_full = filtered_new_df.date[1]
t_lake_full = (filtered_new_df.date - date_start_full) ./ Δdate
inters_lake_temp_surf = reasonableInterpolation(filtered_new_df.Chl, t_lake_full, QuadraticInterpolation, Hour(24) / Δdate)
for inter in inters_lake_temp_surf
    println(inter.t[begin])
end

md"""
We select only the third piece of the date
"""

piece =1
date_start = filtered_new_df[findfirst(inters_lake_temp_surf[piece].t[begin] .== t_lake_full), :date]
df_lake = @subset(filtered_new_df, :date .≥ date_start)
t_lake = (filtered_new_df.date - date_start) ./ Δdate
df_lake = filtered_new_df


date_month = DateTime(2023, 1):Month(1):DateTime(yearmonth(last(filtered_new_df.date))...)
t_month = (date_month - date_start) ./ Δdate
md"""
Agg & Regulirize
"""

derivative_order = 1
length_block_MoM = 2
λ_reg = 1.5
df_lake_agg_MoM = retime_agg(df_lake, diff(df_lake.date)[1] * 5, start=:unchanged, agg_method=x -> MoM_full(x, length_block_MoM))
t_lake_agg = (df_lake_agg_MoM.date - date_start) ./ Δdate

md"""
We put in the aggregated DataFrame the regulirized quantities Cyano, O2, Chl. The chls are interpolated at the few missing sites.
The output is a regulirly spaced DataFrame with no missing and where most outliers have be done
"""
var_bio = map([:O2]) do col
    file = "../data/reg_d_1_lamb_1p5_agg_MoM_$(col).jld2"
    if isfile(file)
       @info "Loading $col"
       reg = load(file)["reg"]
    else
        @time "regularization $(col)" reg = RegularizationSmooth(df_lake_agg_MoM[:, col], t_lake_agg, derivative_order; λ=λ_reg, alg=:fixed)
        jldsave(file; Δdate, start_date = date_start, reg)
    end
    return reg
end
chl = BSplineInterpolation(df_lake_agg_MoM[:,:Chl], t_lake_agg, 2, :ArcLen, :Average)

temperatures = [BSplineInterpolation(df_lake_agg_MoM[:,c], t_lake_agg, 2, :ArcLen, :Average) for c in names(df_lake, r"^Tw")]

df_clean = OrderedDict([:date, :t, :Chl, :O2, Symbol.(names(df_lake, r"^Tw"))...] .=>  [df_lake_agg_MoM.date, t_lake_agg,chl.(t_lake_agg), [r.(t_lake_agg) for r in var_bio]..., [TT(t_lake_agg) for TT in temperatures]...]) |> DataFrame

CSV.write("df_clean_chl.csv", df_clean)



@df df_clean Plots.plot(:date,:Chl)
