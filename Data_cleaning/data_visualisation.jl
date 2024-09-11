using CSV
using DataFrames
using Plots
using Dates
file_lake = "2022_2024_Lake_ESP2/20221015_20240402_ESPWat.csv"
file_meteo = "2022_2024_Lake_ESP2/20221015_20240402_ESPMet.csv"

if gethostname() == "mistea-jojoba"
    file_path = "C:/Users/metivier/Dropbox/PC (2)/Documents/GitLab/lake_data_mistea/data_lake_creteil"
elseif gethostname() == "mistea-torreya"
    file_path = "/home/mathieu/docs/data"
elseif gethostname() == "mistea-sophora"
    file_path = "C:/Users/davenner/Documents/PINNs/PGNN-PINN-Lakes/PINNs/Apprentissage/DifferentialEquations/data"
end



df_lake_full = CSV.read(joinpath(file_path, file_lake), DataFrame, dateformat="d/m/y HH:MM")
df_meteo_full = CSV.read(joinpath(file_path, file_meteo), DataFrame, dateformat="d/m/y HH:MM", delim=";")

Δdate = Dates.Hour(1)
date_start_full = df_lake_full.date[1]
t_lake_full = (df_lake_full.date - date_start_full) ./ Δdate

X = df_lake_full.Cyano
t = t_lake_full
max_Δt =Hour(24) / Δdate
@assert length(X) == length(t)
idx_missings_X = (!ismissing).(X)
t_wo_missing = view(t, idx_missings_X) # inspired by https://discourse.julialang.org/t/how-to-perform-an-argmax-argmin-on-a-subset-of-a-vector/70569
cuts = first(parentindices(t_wo_missing))[findall(diff(t_wo_missing) .> max_Δt)]
N_cuts = length(cuts)

plotlyjs()
using Plots.Measures
plot(df_lake_full[!,:date],[df_lake_full[!,:Chl],df_lake_full[!,:Cyano]],color=[:pink :green])
gr()
pltgen = plot([df_lake_full[!,:Tw_05],df_lake_full[!,:Tw_15],df_lake_full[!,:Tw_25],df_lake_full[!,:Tw_35],df_lake_full[!,:Tw_45]],label= ["Temp 0.5m" "Temp 1.5m" "Temp 2.5m" "Temp 3.5m" "Temp 4.5m"],title = "Températures",xlabel = "Temps",ylabel="Température en C°")
pltzoom = plot([df_lake_full[!,:Tw_05],df_lake_full[!,:Tw_15],df_lake_full[!,:Tw_25],df_lake_full[!,:Tw_35],df_lake_full[!,:Tw_45]],label= ["Temp 0.5m" "Temp 1.5m" "Temp 2.5m" "Temp 3.5m" "Temp 4.5m"],title = "Zoom sur trou",xlabel = "Temps",ylabel="Température en C°",xlim=(3000,6000),ylim=(3,10))
l = @layout [a{0.7w} b]
plotfin = plot(pltgen,pltzoom,layout=l,margin=6mm,size =(1000,400),subplot_ratio=[7/3, 1])

# rapportpath = "C:/Users/davenner/Documents/PINNs/PGNN-PINN-Lakes/PINNs/Apprentissage/Rapport"
# savefig(plotfin,rapportpath*"/temp_trou.png")

md"""
Visualisation Chlorophille. L'optention des fichiers a été faite grace au notebook de david
"""
using Statistics
using CSV, DataFrames, Dates, DataFramesMeta

file_lake_old = filepath*"/2022_2024_Lake_ESP2/20221015_20240402_ESPWat.csv"
file_lake = "/20240723_ESPWat_with_corrected_sensor_data.csv"
dir = @__DIR__
df_lake_full = CSV.read(dir*"/"*file_lake, DataFrame) 
df_lake_full_old = CSV.read(file_lake_old, DataFrame, dateformat="dd/mm/yyyy HH:MM", delim=";", comment = "#")


gr()
using StatsPlots
using Dates
begin

    start_date = DateTime("2023-01-04T00:00:00")
    end_date = DateTime("2024-04-01T00:00:00")
    filtered_new_df = @where(df_lake_full, :date .>= start_date .&& :date .<= end_date)
    filtered_old_df = @where(df_lake_full_old, :date .>= start_date .&& :date .<= end_date)

    @df filtered_old_df plot(:date, :Chl, label = "old data",title ="Données Chlorophylle old/new",xlabel = "Date",ylabel= "Chlorophylle")
    # scatter!(DateTime.(profile_date), [@combine(@subset(df, 0.4 .≤ :depth .≤ 0.6), :mean = mean(:totChl)).mean[1] for df in df_p]
    # , c=:black, s = :dot, label = "profile (Chl tot)")
    @df filtered_df plot!(:date, :Chl, label = "new smooth")
    # @df df_lake_full plot!(:date, :Chl_sensor_corrected, label = "new smooth + corrected?")
end
# savefig(rapportpath*"/chloro_preposttrait.png")