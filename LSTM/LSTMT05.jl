md"""
# LSTM simple pour la prédiction de la température de l'eau à 0.5 m avec eau et air comme données
## Création du modèle LUX
"""

#%%
using Lux
using Random
#%%


#%%
cd(@__DIR__)
include("LSTM_utils.jl")
Random.seed!(1234)
rng = Random.default_rng()
forecaster = T05Forecaster(2,128,1)
ps,st = Lux.setup(rng,forecaster)





#%%
md"""
## Création du dataloader avec les datasets
### importation et visualisation des données
"""
using CSV,DataFrames
using PlotlyJS
using Plots


dir = @__DIR__
df_clean = CSV.read("../data/cleandata/df_clean.csv", DataFrame) 
df_meteo_clean = CSV.read("../data/cleandata/df_meteo_clean.csv", DataFrame) 

T05 = df_clean[!,:Tw_05]
T15 = df_clean[!,:Tw_15]
T25 = df_clean[!,:Tw_25]
T35 = df_clean[!,:Tw_35]
T45 = df_clean[!,:Tw_45]

Tair = df_meteo_clean[!,:AirTemp]

plt = Plots.plot(T05, title = "Données entrainement",label = ["T05m"])
vline!(plt,[7000+31],label = "valsplit")
vline!(plt,[7000+31+500],label = "testsplit")
Plots.plot(Tair)






md"""
### création du dataset/dataloader
"""
nbinstants = 30
trainspan = (1,7000)
valspan = (7000,7500)
testspan = (7500,size(T05)[1])
in_shift_series = [Tair]
in_unshift_series = [T05]
pred_series = [T05]
traindataloader,valdataloader,testdatalaoder,trainplotdataloader,means,stds = getdataloader(trainspan,valspan,testspan,in_unshift_series,in_shift_series,pred_series;nbinstants = nbinstants,batchsize = 128)

md"""
## Loss
On fait une MSE basiquque
"""
function MSE(ŷ,y)
    return sum(sum(abs2,ŷ.-y))/size(y)[2]
end


function loss_function(ps,st,x,y)
    ŷ, st_ = forecaster(x,ps,st)   
    loss = MSE(ŷ,y)


    return loss, st_, (; ŷ = ŷ)
end


md"""
## Entrainement
"""
using Zygote
using Optimisers

adam = Optimisers.Adam(0.001)

clip = Optimisers.ClipNorm(1,2)
opt = Optimisers.OptimiserChain(clip,adam)


Random.seed!(1234)
ps,st = Lux.setup(rng,forecaster)
opt_state = Optimisers.setup(opt,ps)
loss_function = loss_function
nb_epochs =3
history = nothing


history,ps,st,optstate = train(nb_epochs,opt_state,ps,st,loss_function,traindataloader,valdataloader,testdatalaoder,1;history = history,forecaster = forecaster,batchsize=128)


md"""
## Visualisation
"""
plotlyjs()
dataloaders = (trainplotdataloader,valdataloader,testdatalaoder)
tspans = (trainspan,valspan,testspan)
plots = plotTrainValTest(dataloaders,tspans,["Temp 0.5m"],pred_series,1,30,19,forecaster,ps,st)
for plot in plots
display(plot)
end
Plots.plot([history["train_loss"]],yaxis = :log10,label = "train_loss" )
Plots.plot!([history["val_loss"]], yaxis = :log10,label = "val_loss",title = "Losses")
using JLD2
dict = 0

# @load "C:/Users/davenner/Documents/PINNs/PGNN-PINN-Lakes/PINNs/Apprentissage/DifferentialEquations/LSTM/2in_256hid_2out_T05_Tair.jdl2" dict


Plots.plot([history["train_loss"]], labels = ["train_loss"],yaxis = :log10)

md"""
## Prédiction sur 200 échantillons
"""


yt05,y05 = predict(7500,1100,in_unshift_series,in_shift_series,pred_series,19,nbinstants,means,stds,1,ps,st,forecaster;remise = true)

md"""
### Visualisation de la prédiction
"""
plotlyjs()


import  Plots.Measures
plt05 = Plots.plot(df_clean.date[7500:8599],[yt05,y05],margin = 10mm,title = "Prédiction Température 0.5m",xlabel = "Date",ylabel="Température en C°",label = ["Temp 0.5m" "Pred Temp 0.5m"])
Plots.vline!(plt,[7000+31],label = "valsplit",color = :red)
Plots.vline!(plt,[7000+31+500],label = "testsplit",color =:black)

abs35 = RMSEpred(yt35,y35)
rel35 = RMSErelativepred(yt35,y35)

function RMSEpred(ŷ,y)
    return sqrt(sum(abs2,ŷ.-y)/length(y))
end

function RMSErelativepred(ŷ,y)
    sumabs = abs.(ŷ.-y)
    absy = abs.(y)
    res = mean(sumabs./absy)
    return res
end


md"""
## Sauvegarde des paramètres
"""
using HDF5
filename = "testSec_T05fut_Tairfut_T45pres.h5"
tuple = (inner=128,ps = ps)
saveTuple(filename,tuple)
md"""
## Load parameters
"""
in, ps = loadTuple( "T05_bonne_pred.h5")

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

