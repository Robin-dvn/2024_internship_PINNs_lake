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


file_meteo = "2022_2024_Lake_ESP2/20221015_20240402_ESPMet.csv"

df_meteo_full = CSV.read(joinpath(file_path, file_meteo), DataFrame, dateformat="d/m/y HH:MM", delim=";")


md"""
Interpolation and imputing missing
"""
Δdate = Dates.Hour(1)
date_start_full = df_meteo_full.date[1]
t_meteo_full = (df_meteo_full.date - date_start_full) ./ Δdate
inters_lake_temp_surf = reasonableInterpolation(df_meteo_full.AirTemp, t_meteo_full, QuadraticInterpolation, Hour(24) / Δdate)
for inter in inters_lake_temp_surf
    println(inter.t[begin])
end

md"""
We select only the third piece of the date
"""

piece = 3
date_start = df_meteo_full[findfirst(inters_lake_temp_surf[piece].t[begin] .== t_meteo_full), :date]
df_meteo = @subset(df_meteo_full, :date .≥ date_start)
t_meteo = (df_meteo.date - date_start) ./ Δdate
df_meteo = df_meteo[1:3:end,:]


date_month = DateTime(2023, 1):Month(1):DateTime(yearmonth(last(df_meteo_full.date))...)
t_month = (date_month - date_start) ./ Δdate
md"""
Agg & Regulirize
"""

derivative_order = 1
length_block_MoM = 2
λ_reg = 1.5
df_lake_agg_MoM = retime_agg(df_meteo, diff(df_meteo.date)[1] * 5, start=:unchanged, agg_method=x -> MoM_full(x, length_block_MoM))
t_lake_agg = (df_lake_agg_MoM.date - date_start) ./ Δdate

md"""
We put in the aggregated DataFrame the regulirized quantities Cyano, O2, Chl. The temperatures are interpolated at the few missing sites.
The output is a regulirly spaced DataFrame with no missing and where most outliers have be done
"""

temperature = BSplineInterpolation(df_lake_agg_MoM[:,:AirTemp], t_lake_agg, 2, :ArcLen, :Average)

df_meteo_clean = OrderedDict([:date, :t, :AirTemp] .=>  [df_lake_agg_MoM.date, t_lake_agg, temperature]) |> DataFrame
CSV.write("df_meteo_clean.csv", df_meteo_clean)

