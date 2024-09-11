using CSV,DataFrames
using Plots
using JLD2

"""
    find_intervals(arr::Array{T}, min_length::Int, strat::Bool=true) -> Array{Tuple{Int, Int}}

Prend en entrée un tableau de 1 et 0 qui correspond à la stratification ou non.Demande aussi l'état du lac. Renvoie une liste de tuple des indices de début et de fin de chaque période temporelle correspondante à l'état renseigné.
# Arguments
- `arr::Array{T}` : Tableau d'entrée qu'on veut découper
- `min_length::Int` : Longueur minimum de l'intervalle.
- `strat::Bool` :  Si on cherche les intervalles de temps statrifiés du lac ou pas.

# Return
- `Array{Tuple{Int, Int}}` : Liste des intervalles sous forme de tuples (début, fin).

"""
function find_intervals(arr, min_length,strat = true)
    intervals = []
    count = 0
    start_idx = 0
    if strat
        strat = 1
    else
        strat = 0
    end
    
    for (i, val) in enumerate(arr)
        if val == strat
            if count == 0
                start_idx = i
            end
            count += 1
        else
            if count >= min_length
                push!(intervals, (start_idx, start_idx + count - 1))
            end
            count = 0
        end
    end
    
    # Check the last interval
    if count >= min_length
        push!(intervals, (start_idx, start_idx + count - 1))
    end
    
    return intervals
end

"""
    getPartitionSpans(spans::Array{Tuple{Int, Int}}, length::Int) -> Array{Tuple{Int, Int}}

Partitionne la période d'état constant(stratifié ou mélangé) en sous-intervales de période défini. On entrainera un PINN sur chaqun de ces sous intervalles. La période définie est donc la période surlaquelle on fera la modélisation

# Arguments
- `spans::Array{Tuple{Int, Int}}` : Liste des intervalles à découper.
- `length::Int` : Longueur des sous-intervalles.

# Retour
- `Array{Tuple{Int, Int}}` : Liste des sous-intervalles partitionnés sous forme Tuple{Int, Int}
"""
function getPartitionSpans(spans,length)
    partition_spans = []
    for span in spans
        start,stop = span
        indice = start
        while indice+length <= stop
            push!(partition_spans,(indice,indice + length))
            indice += length
        end
    end
    return partition_spans
end

"""
    getStratMixedSpans(length::Int) -> Tuple{Array{Tuple{Int, Int}}, Array{Tuple{Int, Int}}}

Se sert des fonctions précédentes pour renvoyer deux tableaux. Un pour chauqe état rempli des tuples d'indices de chaque sous intervalles correspondant à l'état

# Arguments
- `length::Int` : Longueur des intervalles.

# Retour
- `Tuple{Array{Tuple{Int, Int}}, Array{Tuple{Int, Int}}}` : Intervalles stratifiés et mélangés (indices)
"""
function getStratMixedSpans(length)
    dir = @__DIR__
    df_clean = CSV.read(dir*"/df_clean.csv", DataFrame) 


    T05 = df_clean[!,:Tw_05]
    T15 = df_clean[!,:Tw_15]
    T25 = df_clean[!,:Tw_25]
    T35 = df_clean[!,:Tw_35]
    T45 = df_clean[!,:Tw_45]

    stratif = abs.(T05 .- T45) .> 3 |> x -> Int.(x)

    strat_intervals = find_intervals(stratif, length,true)
    mixed_intervals = find_intervals(stratif, length,false)
    strat_partition_spans = getPartitionSpans(strat_intervals,length)
    mixed_partition_spans = getPartitionSpans(mixed_intervals,length)
    return strat_partition_spans,mixed_partition_spans
end

using Base.Iterators
"""
    getStratMixedLosses(series::Array{Array{T}}, profondeurs::Array{T}, strat_partition_spans::Array{Tuple{Int, Int}}, mixed_partition_spans::Array{Tuple{Int, Int}}) -> Tuple{Array{Function}, Array{Function}}

Calcule les fonctions de pertes pour chaque intervalle stratifié et mélangé. On entraine un PINN sur chaque intervalles donc on a besoin de faire chauqe étape sur chaqun des intervalles.

# Arguments
- `series_temp::Array{Array{T}}` : Tableau des séries temporelles des différentes températures
- `profondeurs::Array{T}` : Indices des profondeurs correspondantes.
- `strat_partition_spans::Array{Tuple{Int, Int}}` : Intervalles stratifiés (indices)
- `mixed_partition_spans::Array{Tuple{Int, Int}}` : Intervalles mélangés (indices)

# Retour
- `Tuple{Array{Function}, Array{Function}}` : Fonctions de perte pour les intervalles stratifiés et mélangés.
"""
function getStratMixedLosses(series_temp,profondeurs,strat_partition_spans,mixed_partition_spans)
    strat_losses = []
    mixed_losses = []
    ##Pour la stratification
    for stratspan in strat_partition_spans
        start,stop = stratspan
        nbtemps = (stop-start)+1
        temps = collect(range(0, stop=1, length=nbtemps))

        function lossfunction(phi,θ,p)
            loss = 0
            for (serie,x) in zip(series_temp,profondeurs)
                pred = [phi([x,t],θ)[1] for t in temps ]
                diff = pred .- serie[start:stop]
                loss+=sum(diff.^2)/length(diff)
                
            end
            return loss
            
        end
        push!(strat_losses,lossfunction)
    end
    ## Pour l'état mélangé
    for mixedspan in mixed_partition_spans
        start,stop = mixedspan
        nbtemps = (stop-start)+1
        temps = collect(range(0, stop=1, length=nbtemps))

        function lossfunction(phi,θ,p)
            loss = 0
            for (serie,x) in zip(series_temp,profondeurs)
                pred = [phi([x,t],θ)[1] for t in temps ]
                diff = pred .- serie[start:stop]
                loss+=sum(diff.^2)/length(diff)
            end
            return loss
        end

        lf = lossfunction
        push!(mixed_losses,lf)
    end

    return (strat_losses,mixed_losses)
    
end

using OptimizationOptimisers,Optim, OptimizationOptimJL,Optimization

"""
    trainPINN(discr::PhysicsInformedNN, pinrep::PINNRepresentation, lossfunction::Function, callback::Function; nbiteradam::Int=100, nbiterlbgs::Int=10, adamlambda::Float64=0.001) -> Tuple{Any, Any}

Entraîne un réseau de neurones par régression de physique informée (PINN). On alterne entre deux algorithmes d'optimisation pour converger. ADAM LBFGS. Le nombres d'itération en argument n'est pas le nombre réelle d'itération de l'algorithme. Ce n'est pas propre mais c'est la solution que j'ai trouvé et je n'ai pas le temps de recoder les différentes fonctions.

# Arguments
- `discr::NeuralPDE::PhysicsInformedNN` : Discrétisation du problème
- `pinrep::PINNRepresentation` : Représentation du PINN.
- `lossfunction::Function` : Fonction de perte.
- `callback::Function` : Fonction de rappel pour l'optimisation.
- `nbiteradam::Int=100` : Nombres d'itérations utilisé dans l'algorithme. Pas le nombre réelle d'itération
- `nbiterlbgs::Int=10` : Nombres d'itérations pour LBFGS. Pas le nombre d'itérations utilisé dans l'algorithme. 
- `adamlambda::Float64=0.001` : Taux d'apprentissage pour Adam (optionnel).

# Retour
- `Tuple{Any, Any}` : Réseau de neurones entraîné et paramètres optimisés.
"""
function trainPINN(discr,pinrep,decomposedloss,callback;nbiteradam = 100, nbiterlbgs = 10,adamlambda =0.001)
    optimFunc = OptimizationFunction(decomposedloss,Optimization.AutoZygote())
    optimprob = OptimizationProblem(optimFunc,pinrep.flat_init_params)
    optadam = OptimizationOptimisers.Adam(adamlambda)
    optlbfgs = Optim.LBFGS()

    maxiters = nbiteradam
    maxiters2 = nbiterlbgs
    ## Convergence vers la moyenne
    res1 = Optimization.solve(optimprob,optadam;maxiters=maxiters,callback = callback)
    prob2 = remake(optimprob;u0 = res1.u)
    ## Sortie due la moyenne
    res2 = Optimization.solve(prob2,optlbfgs;maxiters=maxiters2,callback = callback)
    prob3 = remake(optimprob;u0 = res2.u)
    ## Fit aux données
    res3 = Optimization.solve(prob3,optadam;maxiters=maxiters*15,callback = callback)
    prob4 = remake(optimprob;u0 = res3.u)
    res4 = Optimization.solve(prob4,optadam;maxiters=maxiters*5,callback = callback)
    
    phi = discr.phi
    θ = res4.u
    return (phi,θ)
end


"""
    getRealSeries(Tseries::Array{Array{T}}, spans::Array{Tuple{Int, Int}}) -> Array{Array{T}}

Extrait les séries temporelles réelles à partir des tuples d'indices donnés.

# Arguments
- `Tseries::Array{Array{T}}` : Séries de température.
- `spans::Array{Tuple{Int, Int}}` : Intervalles d'indices.

# Retour
- `Array{Array{T}}` : Séries temporelles réelles extraites.
"""
function getRealSeries(Tseries,spans)

    real_series = []
    for span in spans
        start,stop = span
        series = []
        for serie in Tseries
            push!(series,serie[start:stop])
        end

        push!(real_series,series)
    end
    
    return real_series
end

"""
    plotPINN(network::Tuple{Any, Any}, realseries::Array{Array{T}}, datespan::Array{T}, testloss::Float64, lp::Float64,pltloss::Plots; disp::Bool=true, save::Bool=false, path::String="outpout", anim::Bool=false)

Trace les prédictions du réseau de neurones par rapport aux séries temporelles réelles et génère éventuellement une animation du profil de température au cours du temps

# Arguments
- `network::Tuple{Any, Any}` : Réseau de neurones et paramètres.
- `realseries::Array{Array{T}}` : Séries temporelles réelles.
- `datespan::Array{T}` : Période de temps pour les données.
- `testloss::Float64` : Loos de test sur les températures pas utilisés.
- `lp::Float64` : lambdaphys.
- `disp::Bool=true` : Afficher le graphique (optionnel).
- `save::Bool=false` : Enregistrer le graphique (optionnel).
- `path::String="outpout"` : Chemin pour enregistrer le graphique (optionnel).
- `anim::Bool=false` : Générer une animation (optionnel).
"""
function plotPINN(network,realseries,datespan,testloss,lp,pltloss;disp =true,save=false,path = "outpout",anim = false, indice_prof = 80)


    phi,θ = network
    plt = plot()
    xplot = [0.5,1.5,2.5,3.5,4.5]
    labelsReal = ["Real 0.5" "Real 1.5" "Real 2.5" "Real 3.5" "Real 4.5"]
    labels = ["Estim 0.5" "Estim 1.5" "Estim 2.5" "Estim 3.5" "Estim 4.5"]

    tplot = 0:0.01:1


    colors_base = [colorant"red"  colorant"blue"  colorant"green"  colorant"orange"  colorant"purple"]
    

    treal = collect(range(0, stop=1, length=length(realseries[1])))
    nbjours = length(treal)/19
    plot!(plt,datespan,realseries,label = labelsReal,color=colors_base,title = "Test loss: $testloss Lambda = $lp")
    
    for (i,xp) in enumerate(xplot)
        temp = [phi([xp,t],θ)[1] for t in tplot]
        plot!(plt,datespan,temp,label = labels[i],color=colors_base[i],linestyle=:dash)

    end
    
    if disp
        display(plt)
    end
   
    if save
        savefig(plt,path*"//plot.png")
        savefig(pltloss,path*"//plotloss.png")
    end

    testloss = round(testloss;digits =2)
    if anim
        x_values = LinRange(0.5, 4.5, 101)
        t_values = LinRange(0.,1., 101)
        x_array = collect(x_values)
        t_array = collect(t_values)
        solution = [phi([x, t],θ)[1] for x in x_array, t in t_array]
        # Création d'une animation
        anim = @animate for i in 1:size(solution)[2]
            day  = getDay(i,length(t_values),datespan)
            plot(solution[:, i],x_array,yflip = true,xlim = (4.,26.),title = "Date: $day, Test loss: $testloss",label = "Profil de T")
        end
        # gif(anim,"profil.gif", fps=23)
        gif(anim, path*"//profile.gif", fps=23)
    end
    
    x_values = LinRange(0.5,4.5,101)
    x_array = collect(x_values)
    solution  = [phi([x,i/100],θ)[1] for x in x_array, i in 1:100]
    for i in 1:100
        points_scat  =  map(r -> r[i],realseries)
        day = getDay(indice_prof,101,datespan)
        plot(solution[:,i],x_array,yflip = true,xlim = (4.,26.),title = "Date: $day",label = "Profil T",xlabel = "Température en C°",ylabel = "profondeur")
        scatter!((reduce(hcat,points_scat),[0.5 1.5 2.5 3.5 4.5]),label = ["T 0.5" "T 1.5" "T 2.5" "T 3.5" "T 4.5"])
        savefig(path*"/profil_et_points$i.png")

    end
   

end

"""
    getTestLoss(network::Tuple{Any, Any}, testseries::Array{Array{T}}, profondeurs::Array{T}) -> Float64

Calcule la perte de test pour un réseau de neurones donné et des séries temporelles de test.

# Arguments
- `network::Tuple{Any, Any}` : Réseau de neurones et paramètres.
- `testseries::Array{Array{T}}` : Séries temporelles de test. Températures non utilisées pour l'entrainement
- `profondeurs::Array{T}` : Profondeurs correspondantes.(indices)

# Retour
- `Float64` : Perte de test calculée.
"""
function getTestLoss(network,testseries,profondeurs)
    temps = collect(range(0, stop=1, length=length(testseries[1])))
    phi,θ = network
    loss = 0
    for (serie,x) in zip(testseries,profondeurs)
        pred = [phi([x,t],θ)[1] for t in temps ]
        diff = pred .- serie
        loss+=sum(diff.^2)/length(diff)
    end
    return loss
end
"""
    getDateSpans(indices_spans::Array{Tuple{Int, Int}}, dates::Array{Date}) -> Array{Array{Date}}

Convertit les intervalles d'indices en intervalles de dates.

# Arguments
- `indices_spans::Array{Tuple{Int, Int}}` : Intervalles d'indices.
- `dates::Array{Date}` : Tableau de dates.

# Retour
- `Array{Array{Date}}` : Liste des intervalles de dates.
"""
function getDateSpans(indices_spans,dates)
    datespans = []
    for span in indices_spans
        start,stop = span
        push!(datespans,dates[start:stop])
    end
    return datespans
end

using Dates
"""
    getDay(indice::Int, indicespanlen::Int, dates::Array{Date}) -> Date

Calcule la date correspondante à un indice donné dans une plage d'indices.

# Arguments
- `indice::Int` : Indice dans la plage.
- `indicespanlen::Int` : Longueur de la plage d'indices.
- `dates::Array{Date}` : Tableau de dates.

# Retour
- `Date` : Date correspondante à l'indice.
"""
function getDay(indice,indicespanlen,dates)
    avancement = indice/indicespanlen
    indicedate = Int32(floor(avancement * length(dates)))
    return Date(dates[indicedate])
end

using JSON
"""
    createConfigFile(nb_inner::Int, nb_layer::Int, activation::String, nbpoint::Int, iteradam::Int, indicetemp::Int, iterlbfgs::Int, path::String)

Crée un fichier de configuration JSON avec les paramètres spécifiés.

# Arguments
- `nb_inner::Int` : Nombre de neuronnes par couche;
- `nb_layer::Int` : Nombre de couches.
- `activation::String` : Fonction d'activation.
- `nbpoint::Int` : Nombre de points de données.
- `indicetemp::Int` : Indice de la température.
- `iteradam::Int` : Nombre d'itérations pour Adam.
- `iterlbfgs::Int` : Nombre d'itérations pour LBFGS.
- `path::String` : Chemin du fichier de configuration.
"""
function createConfigFile(nb_inner,
                          nb_layer,
                          activation,
                          nbpoint,
                          indicetemp,
                          iteradam,
                          iterlbfgs,path)

    config = (Reseau = (nb_inner = nb_inner,nb_layer = nb_layer,activation = activation),Donnes = (nbpoints = nbpoint,indice_températures = indicetemp),Optimiseur = (nb_iter_adam = iteradam,nb_iter_lbfs = iterlbfgs) )
    json_str = JSON.json(config)
    open(path, "w") do file
        write(file, json_str)
    end
    
end

using ModelingToolkit
import ModelingToolkit: Interval
"""
    getSys() -> PDESystem

Définit et renvoie un système d'équations aux dérivées partielles pour la diffusion de la chaleur.

# Retour
- `PDESystem` : Système d'équations aux dérivées partielles.
"""

function getSys()


    @parameters x,t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2


    xmin = 0.5
    xmax = 4.5
    tmin = 0
    tmax = 1

    eq = Dt(u(x,t)) ~ 9*Dxx(u(x,t)) ## ici 9 correspond au coefficient de diffusion de la chaleur dans l'eau

t
    ## bcs = [
    ##     u(xmin,t) ~ T053ji(t),
    ##     u(xmax,t) ~ T453ji(t),
    ## ]
    bcs = [u(0,0) ~ 1]
    domains = [
        x ∈ Interval(xmin,xmax),
        t ∈ Interval(tmin,tmax)
    ]

    return @named pdesys = PDESystem(eq,bcs,domains,[x,t],[u(x,t)])

end
"""
    getPersonalLossFunction(pinnrep::PINNRepresentation, poids::Tuple{Float64, Float64, Float64}) -> Function

Crée une fonction de perte personnalisée pondérée pour le réseau de neurones. Décompose les trois loss différentes. Permet de récupérer les valeurs des différentes loss pour pouvoir les plots après. 

# Arguments
- `pinnrep::Any` : Représentation du PINN.
- `poids::Tuple{Float64, Float64, Float64}` : Poids pour les différentes pertes (lp, lb, ld).

# Retour
- `Function` : Fonction de perte personnalisée.
"""
function getPersonalLossFunction(pinnrep,poids)
    lp,lb,ld = poids
    function real_loss(θ,p)
        ## récupération des différentes loss
        bc_losses = pinnrep.loss_functions.bc_loss_functions
        pde_losses =pinnrep.loss_functions.pde_loss_functions
        addloss = pinnrep.loss_functions.additional_loss_function
        full_loss = pinnrep.loss_functions.full_loss_function(θ,p)
        
        return full_loss,[bc_loss(θ) for bc_loss in bc_losses],[pde_loss(θ) for pde_loss in pde_losses],full_loss.-(lb.*[bc_loss(θ) for bc_loss in bc_losses]).-(lp.*[pde_loss(θ) for pde_loss in pde_losses])
    end
    return real_loss
end
"""
    recursiveSave(file_or_group, namedtuple)

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
    


