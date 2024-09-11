
using Lux
using Plots
using CSV,DataFrames
using Statistics
using MLUtils
using Plots.PlotMeasures
include("timeseries_dataset.jl")

struct T05Forecaster{L, D} <: Lux.AbstractExplicitContainerLayer{(:lstm_cell, :dense_layer)}
    lstm_cell::L
    dense_layer::D
end



"""
    T05Forecaster(in_dims, hidd_dims, out_dims)

Crée une instance de `T05Forecaster` avec une cellule LSTM et une couche dense, initialisées selon les dimensions spécifiées.

# Arguments

- `in_dims`: Dimension de l'entrée de la cellule LSTM.
- `hidd_dims`: Dimension cachée de la cellule LSTM et de la couche dense.
- `out_dims`: Dimension de sortie de la couche dense.

# Retour

Renvoie une instance de `T05Forecaster` initialisée avec une cellule LSTM (`Lux.LSTMCell`) prenant les dimensions d'entrée et cachée spécifiées, et une couche dense (`Dense`) avec les dimensions cachée et de sortie spécifiées, utilisant la fonction d'activation ReLU (`Lux.relu`).

# Exemple

Voici comment vous pouvez créer un `T05Forecaster` avec des dimensions spécifiques :

```julia
using Lux

in_dims = 10
hidd_dims = 20
out_dims = 5

forecaster = T05Forecaster(in_dims, hidd_dims, out_dims)
```
"""
function T05Forecaster(in_dims,hidd_dims,out_dims)
    return T05Forecaster(Lux.LSTMCell(in_dims => hidd_dims), Dense(hidd_dims,out_dims,Lux.relu))
end


"""
    (F::T05Forecaster)(x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where {T}

Applique le `T05Forecaster` à une séquence d'entrée `x`, en utilisant les paramètres `ps` et les états `st`.

# Arguments

- `F::T05Forecaster`: Une instance de `T05Forecaster`.
- `x::AbstractArray{T, 3}`: Une séquence d'entrée de type `AbstractArray` à trois dimensions.
    La dimension 1 est l'axe temporelle. Le `T` représente le type des éléments dans l'array.
- `ps::NamedTuple`: Un tuple nommé contenant les paramètres pour `lstm_cell` et `dense_layer`.
- `st::NamedTuple`: Un tuple nommé contenant les états pour `lstm_cell` et `dense_layer`.

# Retour

- `vec(y)`: La sortie de la couche dense après transformation par `T05Forecaster`, aplatie en un vecteur.
- `st`: Un tuple nommé contenant les états mis à jour pour `lstm_cell` et `dense_layer`.

# Description       

Cette fonction applique un modèle `T05Forecaster` à une séquence d'entrée `x` en utilisant les paramètres et les états fournis. Le processus est le suivant :

1. La séquence est d'abord passée par la cellule LSTM (`lstm_cell`). La première étape initialise l'état caché.
2. La séquence est traitée élément par élément en utilisant `eachslice` pour obtenir les éléments sans les copier, et `Iterators.peel` pour séparer le premier élément pour l'initialisation de la LSTM.
3. Le premier élément de la séquence initialise la cellule LSTM, générant une sortie `y` et un état `carry`.
4. Les éléments restants de la séquence sont passés à la cellule LSTM avec l'état `carry` mis à jour à chaque étape.
5. Après avoir traité toute la séquence, la sortie `y` est passée à la couche dense (`dense_layer`) pour la transformation finale.
6. Les états mis à jour pour `lstm_cell` et `dense_layer` sont combinés dans un tuple nommé `st`.

"""
function (F::T05Forecaster)(
    x::Any, ps::NamedTuple, st::NamedTuple) where {T}
    # First we will have to run the sequence through the LSTM Cell
    # The first call to LSTM Cell will create the initial hidden state
    # See that the parameters and states are automatically populated into a field called
    # `lstm_cell` We use `eachslice` to get the elements in the sequence without copying,
    # and `Iterators.peel` to split out the first element for LSTM initialization.
    x_init, x_rest = Iterators.peel(Lux._eachslice(x, Val(2)))#TODO tropuver la bonne dimension en fonction du dataset
    # @show any(ismissing, x_init)
    (y, carry), st_lstm = F.lstm_cell(x_init, ps.lstm_cell, st.lstm_cell)

    # Now that we have the hidden state and memory in `carry` we will pass the input and
    # `carry` jointly
    for x in x_rest
        # @show any(ismissing, x)
        (y, carry), st_lstm = F.lstm_cell((x, carry), ps.lstm_cell, st_lstm)
    end
    # After running through the sequence we will pass the output through the classifier
    y, st_dense_layer = F.dense_layer(y, ps.dense_layer, st.dense_layer)
    # Finally remember to create the updated state
    st = merge(st, (dense_layer=st_dense_layer, lstm_cell=st_lstm))
    
    return y, st
end


"""
    normalize_series(data)

Normalise une série de données et renvoie la série normalisé ainsi que la moyenne et la std
# Arguments
- `data::Vector{T}` : Série de données à normaliser, de type `Vector`.

# Retour
- `normalized_data::Vector{T}` : La série de données normalisée.
- `mean_val::T` : La moyenne des données.
- `std_val::T` : L'écart-type des données (si non nul).
"""
function normalize_series(data)
    mean_val = mean(data)
    std_val = std(data)

    if std_val == 0
        return data .- mean_val  ,meanval# Avoid division by zero
    else
        return (data .- mean_val) ./ std_val,mean_val,std_val
    end
end
"""
    normalize_seriesMeanStd(data, meanv, stdv)

Normalise une série de données en utilisant une moyenne et un écart-type donnés.
# Arguments
- `data::Vector{T}` : Série de données à normaliser.
- `meanv::T` : Moyenne à utiliser pour la normalisation.
- `stdv::T` : Écart-type à utiliser pour la normalisation.

# Retour
- `normalized_data::Vector{T}` : La série de données normalisée.
"""


function normalize_seriesMeanStd(data,meanv,stdv)
    mean_val =meanv
    std_val = stdv

    if std_val == 0
        return data .- mean_val  # Avoid division by zero
    else
        return (data .- mean_val) ./ std_val
    end
end

"""
    getdataloader(trainspan, valspan, testspan, unshift_series, shift_series, pred_series; batchsize=128, shiftfutur=19, nbinstants=30)

Prépare les séries temporelles pour l'entraînement, la validation et le test. Applique le décalage temporel, normalise les données et crée des DataLoader pour chaque phase.

# Arguments
- `trainspan::Tuple` : Intervalle des données d'entraînement.
- `valspan::Tuple` : Intervalle des données de validation.
- `testspan::Tuple` : Intervalle des données de test.
- `unshift_series::Matrix` : Séries non décalées.
- `shift_series::Matrix` : Séries décalées.
- `pred_series::Matrix` : Séries cibles à prédire.
- `batchsize::Int` : Taille des lots pour les DataLoader (par défaut 128).
- `shiftfutur::Int` : Nombre d'instants de décalage à appliquer (par défaut 19).
- `nbinstants::Int` : Nombre d'instants temporels à inclure dans chaque échantillon (par défaut 30).

# Retour
- `traindataloader`, `valdataloader`, `testdataloader`, `trainplotdataloader` : Les DataLoader pour l'entraînement, la validation, et le test.
- `means`, `stds` : Les moyennes et écarts-types utilisés pour normaliser les séries.
"""


function getdataloader(trainspan,valspan,testspan,unshift_series,shift_series,pred_series;batchsize = 128,shiftfutur = 19,nbinstants=30)
    # gestion des dimensions si l'entrée est un vecteur avec une série temporelle 
    # C'est mal codé, ca doit être plus robuste
    if length(unshift_series) == 1
        unshift_series = [unshift_series[1];]
    end
    if length(shift_series) == 1
        shift_series = [shift_series[1];]
    end
    if length(pred_series) == 1
        pred_series = [pred_series[1];]
    end

    trainstart,trainend = trainspan 
    valstart,valend = valspan
    teststart,testend = testspan

    ## gestion des indices de shiftage pour avoir la meme taille ( série déclaé dans le future ...)
    testend = testend - shiftfutur
    shiftserie(ser) = ser[1+shiftfutur:end]
    unshiftserie(ser) = ser[1:end-shiftfutur]
    shift_series = mapslices((ser) -> shiftserie(ser),shift_series,dims=1)
    unshift_series = mapslices((ser) -> unshiftserie(ser),unshift_series,dims=1)
    pred_series = mapslices(unshiftserie,pred_series,dims=1)
    @show (size(shift_series),size(unshift_series),size(pred_series))

    ## normalisation du dataset
    gettrainmean(ser) = mean(ser[trainstart:trainend])
    gettrainstd(ser) = std(ser[trainstart:trainend])
    means =  [mapslices(gettrainmean,shift_series,dims=1),mapslices(gettrainmean,unshift_series,dims=1)]
    stds =  [mapslices(gettrainstd,shift_series,dims=1),mapslices(gettrainstd,unshift_series,dims=1)]
    
    if length(unshift_series) == 1
        unshift_series = [unshift_series[1] ;;]
    end
    if length(shift_series) == 1
        shift_series = [shift_series[1] ;;]
    end


    shift_series = [shift_series[:, i] for i in 1:size(shift_series, 2)]
    unshift_series = [unshift_series[:, i] for i in 1:size(unshift_series, 2)]
    norm_shift_series = []
    for (i,serie) in enumerate(shift_series)
        push!(norm_shift_series,normalize_seriesMeanStd(serie,means[1][i],stds[1][i]))
      
    end
    print("avant")
    norm_unshift_series = []
    for (i,serie) in enumerate(unshift_series) 

        push!(norm_unshift_series,normalize_seriesMeanStd(serie,means[2][i],stds[2][i]))

    end

    ## Création des dataloaders

    #norm_shift_series = map((d,m,v) -> normalize_seriesMeanStd(d,m,v),[shift_series],means[1],stds[1])
    #norm_unshift_series = map((d,m,v) -> normalize_seriesMeanStd(d,m,v),[unshift_series],means[2],stds[2])
    @show (size(norm_shift_series),size(norm_unshift_series),size(pred_series))
    time_series = [norm_shift_series... norm_unshift_series... [pred_series]...]
    @show size(time_series)
    traindataloader = timeseries_dataset_from_table(time_series[trainstart:trainend,:],nbinstants;batch_size= batchsize,shuffle = true)
    trainplotdataloader = timeseries_dataset_from_table(time_series[trainstart:trainend,:],nbinstants;batch_size= batchsize,shuffle = false)
    valdataloader = timeseries_dataset_from_table(time_series[valstart:valend,:],nbinstants;batch_size= batchsize)
    testdataloader = timeseries_dataset_from_table(time_series[teststart:size(time_series)[1],:],nbinstants;batch_size= batchsize)
    return traindataloader,valdataloader,testdataloader,trainplotdataloader,means,stds

end
"""
    train(nbepochs, opt_state, ps, st, loss, traindataloader, valdataloader, testdataloader, output_dim; smoothing_factor=0.99, verbose=1, history=nothing, forecaster=nothing, batchsize=58)

Entraîne un modèle sur les données fournies, en ajustant les paramètres avec descente de gradient.

# Arguments
- `nbepochs::Int` : Nombre d'époques d'entraînement.
- `opt_state::OptimState` : État de l'optimiseur.
- `ps::Parameters` : Paramètres du modèle.
- `st::State` : État du modèle.
- `loss::Function` : Fonction de perte à minimiser.
- `traindataloader::DataLoader` : DataLoader pour les données d'entraînement.
- `valdataloader::DataLoader` : DataLoader pour les données de validation.
- `testdataloader::DataLoader` : DataLoader pour les données de test.
- `output_dim::Int` : Dimension de la sortie.
- `smoothing_factor::Float` : Facteur de lissage pour la perte (par défaut 0.99).
- `verbose::Int` : Niveau de verbosité (par défaut 1).
- `history::Dict` : Dictionnaire pour stocker l'historique de la perte (par défaut `nothing`).
- `forecaster::Function` : Modèle prédictif à utiliser pour la validation. Si un modèle est fournit on évaluera la prédiction en ajoutant la nouvelle prédiction à la série temporelle. SI rien n'est mis la validation se fait avec les valeurs réèlles de la série temporelle

# Retour
- `history::Dict` : Historique des pertes.
- `ps::Parameters` : Paramètres du modèle mis à jour.
- `st::State` : État du modèle mis à jour.
- `opt_state::OptimState` : État de l'optimiseur mis à jour.
"""


function train(nbepochs,opt_state,ps,st,loss,traindataloader,valdataloader,testdataloader,output_dim;smoothing_factor=0.99,verbose = 1,history = nothing,forecaster = nothing,batchsize=58)
    if isnothing(history)
        history = Dict()
        history["train_loss"] = Array{Float32}(undef, 0)
        history["current_loss"] = Array{Float32}(undef, 0)
        history["val_loss"] = Array{Float32}(undef,0)
    end
    
    for epoch in 1:nbepochs
        last_loss = 0 ## for smoothed loss
        ## train on data
        batch = 0
        for x_y in traindataloader
            y = x_y[end - output_dim+1:end,end,:]
            x = x_y[1:end - output_dim,1:end-1,:]
            ##  reshaping pour compatibilité de la loss
            if output_dim == 1
                y = reduce(hcat,y)
            end

            (current_loss,st,pred),pb_f = Zygote.pullback(loss,ps,st,x,y)
            smoothed_loss = smoothing_factor * current_loss + (1.0 - smoothing_factor) * last_loss
            last_loss = smoothed_loss
            push!(history["current_loss"],current_loss)
            println("Loss :$last_loss")
          
            #Backprop and Gradient Descent
            gd = pb_f((one(current_loss),nothing,nothing,nothing))[1]
            # if batch == 1

            #     println(gd[:lstm_cell]) 
            # end  
            opt_state,ps = Optimisers.update(opt_state,ps,gd)
            batch +=1
        end
        push!(history["train_loss"],last_loss)
        println("Loss Value after $epoch epochs: $last_loss")
        ## Validation
        loss_computed = false
        last_val_loss =0
        val_losses = []
        st_ = Lux.testmode(st)
        for x_y in valdataloader
            y = x_y[end - output_dim+1:end,end,:]
            x = x_y[1:end - output_dim,1:end-1,:]
            ##  reshaping pour compatibilité de la loss
            if output_dim == 1
                y = reduce(hcat,y)
            end
            current_val_loss = 0
            loss_computed = false
            #calcul de la loss de prédiction si le batch est complet.
            #C'est à dire la loss pour une prédiction avec remise sur une un nombre d'échantillon correspondant  à la taille du batch
            #Si aucun forecaster n'est fournit, ka validation se fait sur la loss de la prédiction avec les vraies données (sans remise)
            if forecaster !=nothing
                if size(x)[3] == batchsize
                    current_val_loss = predictforloss(x,y,batchsize,ps,st,forecaster,MSE)
                    loss_computed = true
                end
            else
                (current_val_loss,st_,pred) = loss(ps,st_,x,y)
                
                loss_computed = true
            end
            if loss_computed
                smoothed_val_loss = smoothing_factor * current_val_loss + (1.0 - smoothing_factor) * last_val_loss
                last_val_loss = smoothed_val_loss
                push!(val_losses,current_val_loss)
            end
            
        end
        mena_val_loss = mean(val_losses)
        push!(history["val_loss"],mean(val_losses))
        println("Mean val Loss Value after $epoch epochs: $mena_val_loss")
            
    end
    return history,ps,st,opt_state
end
"""
    plotTrainValTest(dataloaders, tspans, labels, pred_series, output_dim, nbinstants, shiftfutur, forecaster, ps, st)

Trace les prédictions du modèle sur les données d'entraînement, de validation et de test avec l'erreur quadratique moyenne (RMSE) pour chaque série.

# Arguments
- `dataloaders::Tuple` : DataLoader pour l'entraînement, la validation et le test.
- `tspans::Tuple` : Intervalles des données pour chaque phase.
- `labels::Vector` : Étiquettes pour les séries.
- `pred_series::Vector` : Séries de prédictions cibles.
- `output_dim::Int` : Dimension de la sortie.
- `nbinstants::Int` : Nombre d'instants temporels à considérer.
- `shiftfutur::Int` : Nombre d'instants de décalage.
- `forecaster::Function` : Modèle utilisé pour faire des prédictions.
- `ps::Parameters` : Paramètres du modèle.
- `st::State` : État du modèle.

# Retour
- `plots::Vector` : Liste des graphiques montrant les prédictions pour l'entraînement, la validation, et le test.
"""


function plotTrainValTest(dataloaders,tspans,labels,pred_series,output_dim,nbinstants,shiftfutur,forecaster,ps,st)
    traindataloader,valdataloader,testdatalaoder = dataloaders
    trainspan,valspan,testspan = tspans

    if output_dim==1
        pred_series = reshape(pred_series[1],size(pred_series[1])[1],1)
    end

    ypred = []
    yvalpred = []
    ytestpred = []

    for x_y in traindataloader

        xplot = x_y[1:end-output_dim,1:end-1,:]
        push!(ypred,forecaster(xplot,ps,st)[1])
    end

    # plttrain = Plots.plot([serie[trainspan[1]+nbinstants:trainspan[2]] reduce(hcat,ypred)'])
    for x_y in valdataloader

        xplot = x_y[1:end-output_dim,1:end-1,:]
        push!(yvalpred,forecaster(xplot,ps,st)[1])
    end
    i = 1
    for x_y in testdatalaoder

        xplot = x_y[1:end-output_dim,1:end-1,:]
        res  = forecaster(xplot,ps,st)[1]
 
        if i ==1
            print(xplot[:,end,2])
            print(res[1])
        end
        i=2
        push!(ytestpred,forecaster(xplot,ps,st)[1])
    end
    print(size(reduce(hcat,ypred)')) 

    plots = []



    for i in 1:size(pred_series)[2]
        serie = pred_series[:,i]
        
        trpltserie = serie[trainspan[1]+nbinstants:trainspan[2]]
        rmse = round(sqrt(MSE(reshape(trpltserie,1,size(trpltserie)[1]),reduce(hcat,ypred)'[:,i]')),digits=2)
        plttrain = Plots.plot([reshape(trpltserie,size(trpltserie)[1],1) reduce(hcat,ypred)'[:,i]],title = "Train RMSE = $rmse")
        valpltserie = serie[valspan[1]+nbinstants:valspan[2]]
        rmse =round(sqrt(MSE(reshape(valpltserie,1,size(valpltserie)[1]),reduce(hcat,yvalpred)'[:,i]')),digits = 2) 
        pltval = Plots.plot([reshape(valpltserie,size(valpltserie)[1],1) reduce(hcat,yvalpred)'[:,i]],title = "Valisation RMSE = $rmse")
        testpltserie = serie[testspan[1]+nbinstants:size(serie)[1]-shiftfutur]
        rmse = round(sqrt(MSE(reshape(testpltserie,1,size(testpltserie)[1]),reduce(hcat,ytestpred)'[:,i]')),digits=2)
        plttest = Plots.plot([reshape(testpltserie,size(testpltserie)[1],1) reduce(hcat,ytestpred)'[:,i]],title = "Test RMSE = $rmse")
        layout = @layout [a; [b c]]
        push!(plots,Plots.plot(plttrain,pltval,plttest,layout = layout,legend = :none))
    end

    return plots
end

"""
    predict(start::Int, nbpoints::Int, in_unshift_series::Array, in_shift_series::Array, outputseries::Array, shiftfutur::Int, nbinstants::Int, means::Array, stds::Array, nboutput::Int, ps, st, forecaster; remise=true)

Prédit des valeurs futures basées sur les séries temporelles d'entrée et le modèle entraîné.L'argument permet d'indiquer si la prédiction de l'instant t sera utiliser pour prédire l'instant t+1. Si ce n'est pas le cas, les valeurs réèlles de la série temporelle seront ajouté

# Arguments:
- `start`: Index de départ pour la prédiction.
- `nbpoints`: Nombre de points à prédire.
- `in_unshift_series`, `in_shift_series`, `outputseries`: Données des séries d'entrée et de sortie.
- `shiftfutur`: Nombre de décalages futurs.
- `nbinstants`: Nombre d'instants temporels.
- `means`, `stds`: Moyennes et écarts-types pour la normalisation.
- `nboutput`: Nombre de valeurs de sortie à prédire.
- `ps`: Paramètres du modèle.
- `st`: État du modèle.
- `forecaster`: Fonction de prévision pour les valeurs futures.
- `remise`: Si les valeurs prédites doivent être réintroduites dans les entrées (par défaut : `true`).

# Retour:
- Tuple des valeurs vraies et prédites.
"""


function predict(start,nbpoints,in_unshift_series,in_shift_series,outputseries,shiftfutur,nbinstants,means,stds,nboutput,ps,st,forecaster;remise = true)
    
    if length(in_unshift_series) == 1
        in_unshift_series = [in_unshift_series[1] ;;]
    end
    if length(in_shift_series) == 1
        in_shift_series = [in_shift_series[1] ;;]
    end
    if length(outputseries) == 1
        outputseries = [outputseries[1] ;;]

    end
    outputseries = [outputseries[:, i] for i in 1:size(outputseries, 2)]
    in_shift_series = [in_shift_series[:, i] for i in 1:size(in_shift_series, 2)]
    in_unshift_series = [in_unshift_series[:, i] for i in 1:size(in_unshift_series, 2)]
    # création des vecteurs target
    ytrues = []
    for serie in outputseries
        push!(ytrues,serie[start+nbinstants:start+nbpoints+nbinstants])
    end
    #création des séries normalisées et du premier inpu
    x = []
       
    norm_shift_series = []
    for (i,serie) in enumerate(in_shift_series)
        push!(norm_shift_series,normalize_seriesMeanStd(serie',means[1][i],stds[1][i]))
        push!(x,norm_shift_series[i][start+shiftfutur:start+shiftfutur+nbinstants-1])
    end

     norm_unshift_series = []
    for (i,serie) in enumerate(in_unshift_series) 

        push!(norm_unshift_series,normalize_seriesMeanStd(serie,means[2][i],stds[2][i]))
        push!(x,norm_unshift_series[i][start:start+nbinstants-1])
    end

    x = reduce(hcat,x)'

    x = reshape(x, size(x)[1],nbinstants,1 )
    #print(x[7,:,1])
    #x = permutedims(xt, (2, 1, 3))


    ypreds = []
    for i in 1:nbpoints
        new_y, st = forecaster(x,ps,st)
        push!(ypreds,new_y[1])
        
       
        
        for (j,new) in enumerate(new_y)
            new_y[j]= normalize_seriesMeanStd(new,means[2][j],stds[2][j])
        end
        xtnew = []
        for shi_ser in norm_shift_series
            push!(xtnew,shi_ser[start+nbinstants-1+shiftfutur+i])
        end
        
        for new in new_y
            if remise
                 push!(xtnew,new)
            else
                push!(xtnew,norm_unshift_series[1][start+nbinstants-1+i])
            end
        end
        xtnew = reduce(vcat,xtnew)
        xtnew = reshape(xtnew,size(x)[1],1,1)
        if i == 2
            print(new_y[1])
            print(x[:,end,1])
        end

        x = cat(x[:,2:end,:],xtnew,dims = 2)
        
    end
    return (ytrues[1][1:end-1],ypreds)


end
"""
    predictforloss(x::Array, y::Array, batchsize::Int, ps, st, forecaster, loss)

Fait la prédiction pour la calculer une loss de validation utilisé dans le train. la prédiction à t est utilisée oour prédire à t+1.

# Arguments:
- `x`: Données d'entrée.
- `y`: Valeurs de sortie vraies.
- `batchsize`: Nombre de lots.
- `ps`: Paramètres du modèle.
- `st`: État du modèle.
- `forecaster`: Fonction de prévision.
- `loss`: Fonction de perte.

# Retour:
- Valeur de la perte calculée pour la prédiction.
"""



function predictforloss(x,y,batchsize,ps,st,forecaster,loss)
    xinit = x[:,:,1]
    ypred = []
    xboucle = xinit

    xboucle = reshape(xboucle,size(x)[1],size(x)[2],1)


    for i in 2:batchsize
        ŷ= forecaster(xboucle,ps,st)
        push!(ypred,ŷ[1][1])
        xnew=[]
        for j in 1:size(x)[1]-1
            push!(xnew,x[j,end,i])
        end
        push!(xnew,ŷ[1])
        xnew = reduce(vcat,xnew)
        xnew = reshape(xnew,size(x)[1],1,1)
        xboucle = cat(x[:,2:end,1],xnew,dims=2)
        

        
    end
    ŷ = vcat(ypred)
    ŷ = Matrix(reshape(ŷ,1,length(ŷ)))
    ŷ =  convert(Matrix{Float64}, ŷ)
    #ŷ =  Float64.(ŷ)
    y = y[:,1:end-1]
    l = loss(ŷ,y)
    println("Loss Val instant: $l")
    return l

end

"""
    recursiveSave(file_or_group, namedtuple::NamedTuple)

Permet d'enregistrer un Tuple dans un groupe HDF5 de manière récrsive afin de créer l'arborescence.

# Arguments
- `file_or_group`: Un fichier ou groupe HDF5 dans lequel les données doivent être enregistrées.
- `namedtuple`: Un `NamedTuple` dont les valeurs seront sauvegardées dans le groupe.

# Retour
- Renvoie l'objet `file_or_group` après l'enregistrement.
"""


function recursiveSave(file_or_group,namedtuple)
    for (key,value) in pairs(namedtuple)

        

        if isa(value,NamedTuple)
            g = create_group(file_or_group,string(key))
            recursiveSave(g,value)
        else
            file_or_group[string(key)] = value

        end
    end
  return file_or_group
end

"""
    namedtuplefromfile(group)

Reconstitue un `NamedTuple` à partir d'un groupe HDF5. Si un sous-groupe est trouvé, la fonction est appelée récursivement pour reconstruire l'arborescence des données.

# Arguments
- `group`: Le groupe HDF5 ou fichier à lire.

# Retour
- Renvoie un `NamedTuple` reconstruit à partir du fichier ou du groupe HDF5.
"""
function namedtuplefromfile(group)
    dict = Dict()
    for name in keys(group)
        obj = group[name]
        if isa(obj,HDF5.Group)
            dict[name]=namedtuplefromfile(obj)
        elseif isa(obj,HDF5.Dataset)
            dict[name] = read(obj)
        end
    end
    return NamedTuple((Symbol(key),value) for (key,value) in dict)

end

"""
    saveTuple(path, namedtuple)

Enregistre un `NamedTuple` dans un fichier HDF5 au chemin spécifié.

# Arguments
- `path`: Le chemin du fichier HDF5.
- `namedtuple`: Le `NamedTuple` à sauvegarder.

# Retour
- Aucun, les données sont directement sauvegardées dans le fichier.
"""


function saveTuple(path,namedtuple)
    h5open(path,"w") do file
        recursiveSave(file,namedtuple)
    end
    
end
"""
    loadTuple(path)

Charge un `NamedTuple` à partir d'un fichier HDF5.

# Arguments
- `path`: Le chemin du fichier HDF5 à lire.

# Retour
- Renvoie un `NamedTuple` reconstruit à partir des données du fichier HDF5.
"""


function loadTuple(path)
    file = h5open(path,"r") 
    tuple = namedtuplefromfile(file)
    close(file)
    return tuple
    
end
    



