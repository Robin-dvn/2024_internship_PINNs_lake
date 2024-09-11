using Pkg
#Pkg.activate("/PINN/")
using Distributed # permet de faire du calcul en parallèle
addprocs(3) # ajoute des "coeur" de calcul
#everywhere permet d'exécuter le code sur l'ensemble des coeurs
@everywhere using Dates
@everywhere using JLD2
@everywhere using Statistics
@everywhere using Lux,NeuralPDE
@everywhere using Random
@everywhere using ComponentArrays
@everywhere using Distributed
@everywhere using HDF5
@everywhere include("PINN_utils.jl")


"""
    show_progress(completed::Int, total::Int)

Affiche une barre de progression indiquant l'avancement de l'entraînement.

# Arguments
- `completed::Int`: Le nombre d'étapes déjà réalisées.
- `total::Int`: Le nombre total d'étapes à accomplir.

# Exemple
show_progress(50, 100)

Affichera :
Progress: [=========================                              ] 50.0%

Cette fonction est utile pour suivre la progression de longues boucles ou d'opérations nécessitant un suivi visuel.
"""
@everywhere function show_progress(completed, total)
    percentage = completed / total
    completed_width = Int(round(percentage * 100))
    remaining_width = 100 - completed_width

    bar = "[" * "="^completed_width * " "^(remaining_width) * "]"
    println("Progress: $bar $((percentage*100))%")
end

"""
    main(nom_exp::String, lambdas::Vector{Float64}, indices_profondeurs::Vector{Int}, 
         nb_res_st::Int, nb_res_mx::Int, nbiter_adam::Int, nbiter_lbfgs::Int)

Point d'entrée principal pour l'entraînement de réseaux de neurones basés sur des données physiques.
Cette fonction gère le processus d'entraînement pour plusieurs valeurs de régularisation (`lambda`), sur différentes profondeurs de températures, avec deux stratégies d'entraînement (stratification et couches mixtes).

# Arguments
- `nom_exp::String`: Nom de l'expérience. Utilisé pour créer des dossiers de sortie.
- `lambdas::Vector{Float64}`: Liste des valeurs du 'λ' physique associé à la fonction de coût résiduelle.
- `indices_profondeurs::Vector{Int}`: Indices des températures à utiliser pour l'entraînement.
- `nb_res_st::Int`: Nombre d'intervales à modéliser pour l'entraînement stratifié.
- `nb_res_mx::Int`: Nombre d'intervales à modéliser pour l'entraînement avec couches mixtes.
- `nbiter_adam::Int`: Nombre d'itérations utilisé dans l'algorithme d'entraînement. Attention. Le nombre réel d'itération n'est aps celui-ci. VOir fonction 'trainPINN'
- `nbiter_lbfgs::Int`: Nombre d'itérations utilisé dans l'algorithme d'entraînement.Attention. Le nombre réel d'itération n'est aps celui-ci. VOir fonction 'trainPINN'

# Description
- Charge les données de température depuis un fichier CSV.
- sépare les données en des intervales de 5 jours, classé par état (stratifié ou mélangé)
- Entraîne un PINN pour chacun des intervales .
- Gère les sauvegardes des résultats et des paramètres de chaque modèle dans des dossiers spécifiques.

# Exemple
main("Exp_1", [0.1, 0.01], [1, 3, 5], 3, 2, 1000, 500)

Entraîne des modèles pour l'expérience nommée "Exp_1" avec les régularisations `λ = 0.1` et `λ = 0.01`, pour les indices de profondeur `1`, `3` et `5`, avec 3 résolutions stratifiées, 2 résolutions mixtes, et fait une alternance ADAM et LBFGS. 
# Notes
Cette fonction utilise un backend graphique pour visualiser les pertes et génère des fichiers de sortie sous forme de graphiques et de paramètres optimisés.
"""
function main(nom_exp,lambdas,indices_profondeurs,nb_res_st,nb_res_mx,nbiter_adam,nbiter_lbfgs)
    # backend d'affichage
    gr()
    # indices des températures qui seront utilisés pour le test
    indices_a_exclure = setdiff(1:5,indices_profondeurs)
    # importation des données et création des légendes
    dir = @__DIR__
    df_clean = CSV.read(dir*"/df_clean.csv", DataFrame)
    
    T05 = df_clean[!,:Tw_05]
    T15 = df_clean[!,:Tw_15]
    T25 = df_clean[!,:Tw_25]
    T35 = df_clean[!,:Tw_35]
    T45 = df_clean[!,:Tw_45]
    
    temps_strings = ["T05" "T15" "T25" "T35" "T45" ]
    temps_strings_train = temps_strings[indices_profondeurs]
    temps_strings_test = temps_strings[indices_a_exclure]
    tempstringpath = ""
    for temp in temps_strings_train
        tempstringpath *= "_$temp"
    end
    templosstitle = ""
    for temp in temps_strings_test
        templosstitle *= " $temp"
    end
    temperatures = [T05,T15,T25,T35,T45]
    temperatures = temperatures[indices_profondeurs]
    profondeurs = [0.5 1.5 2.5 3.5 4.5]
    profondeurstrain = profondeurs[indices_profondeurs]
    dates = df_clean[!,:date]

    # création du dossier d'expérience
    dir = @__DIR__
    date = now()
    datetime_str = Dates.format(date, "yyyy-mm-dd_HH_MM_SS")
    dirpath = dir*"//$(nom_exp)_($tempstringpath)_"*datetime_str
    mkdir(dirpath)

    # création du réseau de neurones et enregistrement de la configuration de l'expérience
    inner = 25
    nbpoints_exp = 100
    nbiteradam = nbiter_adam
    nbiterlbfgs = nbiter_lbfgs
    
    
    configpath = dirpath*"//config.json"
    createConfigFile(inner,4,"tanh",nbpoints_exp,indices_profondeurs,nbiteradam,nbiterlbfgs,configpath)

    chain = Lux.Chain(Dense(2, inner, Lux.tanh),
                        Dense(inner, inner, Lux.tanh),
                        Dense(inner, inner, Lux.tanh),
                        Dense(inner, inner, Lux.tanh),
                        Dense(inner, 1))
    
    ps = Lux.setup(Random.default_rng(), chain)[1]
    
    pdesys = getSys() # création du sytème de PDE de Modeling toolkit 
    
    # Distribution des calculs sur les coeur et @sync permet d'attendre que chaque coeur finisse son calcul avant que le processus principal ne se termine
    @sync @distributed for i in 1:length(lambdas)
        lambda = lambdas[i]
        ## création des dossier
        lambdapath = dirpath*"//lambdaphys_$lambda"
        mkdir(lambdapath)
        stratpath = lambdapath*"//strat"
        mixedpath = lambdapath*"//mixed"
        mkdir(stratpath)
        mkdir(mixedpath)

        ## génération des différent intervales de modélisation
        st, mx = getStratMixedSpans(100)
        st_dates,mx_dates = getDateSpans(st,dates),getDateSpans(mx,dates)

        ## génération des discrétisation, des représentation et des loos personalisé
        stratloss, mixedloss = getStratMixedLosses(temperatures,profondeurstrain,st,mx)
        stratdiscrs = []
        mixeddiscrs = []
        stpinreps = []
        mxpinreps = []
        stdecomposedlosses = []
        mxdecomposedlosses = []

        for (stloss,mxloss) in zip(stratloss,mixedloss)

            naloss = NonAdaptiveLoss(;pde_loss_weights = lambda, bc_loss_weights = 0., additional_loss_weights = 1.)
            # graloss = GradientScaleAdaptiveLoss(100;
            #             weight_change_inertia = 0.9,
            #             pde_loss_weights = lambda,
            #             bc_loss_weights = 0.,
            #             additional_loss_weights = 1.0)
            strategy = QuasiRandomTraining(1000;bcs_points = 1) #inutile puisque pas de poids sur les loss²
            st_dis = PhysicsInformedNN(chain,strategy;init_params = ps,adaptive_loss=naloss,additional_loss=stloss)
            mx_dis = PhysicsInformedNN(chain,strategy;init_params = ps,adaptive_loss=naloss,additional_loss=mxloss)
            stpinnrep = symbolic_discretize(pdesys,st_dis)
            strealloss = getPersonalLossFunction(stpinnrep,[lambda,0.,1.])
            mxpinnrep = symbolic_discretize(pdesys,mx_dis)
            mxrealloss = getPersonalLossFunction(mxpinnrep,[lambda,0.,1.])
            push!(stratdiscrs,st_dis)
            push!(mixeddiscrs,mx_dis)
            push!(stpinreps,stpinnrep)
            push!(mxpinreps,mxpinnrep)
            push!(stdecomposedlosses,strealloss)
            push!(mxdecomposedlosses,mxrealloss)
        end
        st_realseries = getRealSeries([T05,T15,T25,T35,T45],st)
        mx_realseries = getRealSeries([T05,T15,T25,T35,T45],mx)

        

        ## entrainement stratification
 

        for i in 1:nb_res_st
            dis,pinrep,decomposedloss,realserie,datespan  = stratdiscrs[i],stpinreps[i],stdecomposedlosses[i],st_realseries[i],st_dates[i]
            realserieloss = realserie[indices_a_exclure]
            profondeur_test = profondeurs[indices_a_exclure]
            startdate,stopdate = datespan[1], datespan[end]
            datestr = "//"*Dates.format(startdate, "yyyy-mm-dd")*"---"*Dates.format(stopdate, "yyyy-mm-dd")
            partitionpath = stratpath*datestr 
            mkdir(partitionpath)

  
            totalLoss = []
            physloss = []
            dataloss = []
            bcloss = []
            iteration = 1
            callback = function callback(p,l,args...)
                bc_loss_arr,pde_loss_arr,add_loss_arr = args
                push!(totalLoss,l)
                push!(physloss,lambda*pde_loss_arr[1])
                push!(bcloss,0*bc_loss_arr[1])
                push!(dataloss,1.0*add_loss_arr[1])
                
                iteration  = iteration +1
                if iteration % 500 == 0
                    print("Strat lambda: $lambda , n°$(floor(i))  ")
                    show_progress(iteration,21 * nbiter_adam + 2*nbiter_lbfgs)
                end
                
                return false    
            end
            time = 0
            phi = nothing
            θ = nothing
            if lambda == 0.
                network,time =@timed trainPINN(dis,pinrep,decomposedloss,callback;nbiteradam = 1, nbiterlbgs = 100,adamlambda = 0.01)
                phi,θ = network
            else
                network,time =@timed trainPINN(dis,pinrep,decomposedloss,callback;nbiteradam = nbiteradam, nbiterlbgs = nbiterlbfgs,adamlambda = 0.01)
                phi,θ = network
            end
            ## sauvegarde des paramètres et gestion du cas ou il y a les 5 températures pour l'entraînement
            if length(indices_profondeurs) == 5
                pltloss = plot([totalLoss,physloss,bcloss,dataloss],label = ["total" "phy" "bc" "dataloss"])
                plotPINN((phi,θ),realserie,datespan,0,lambda,pltloss;anim = true,save=true,path = partitionpath)
                params = (θ = θ,testloss = -1,totalLoss = totalLoss, dataloss = dataloss,physloss = physloss,time = time)

            else
                testloss = getTestLoss((phi,θ),realserieloss,profondeur_test)
                pltloss = plot([totalLoss,physloss,bcloss,dataloss],label = ["total" "phy" "bc" "dataloss"])
                pltloss = plot([totalLoss,physloss,bcloss,dataloss],label = ["total" "phy" "bc" "dataloss"])
                plotPINN((phi,θ),realserie,datespan,testloss,lambda,pltloss;anim = true,save=true,path = partitionpath)
                params = (testloss = testloss,totalLoss = totalLoss.|>Float64, dataloss = dataloss.|>Float64,physloss = physloss.|>Float64,time = time)

            end
            
            saveTuple(partitionpath*"/params.h5",params)
            GC.gc()



        end

        ## entrainement mixed layers
      

        for i in 1:nb_res_mx 
            dis,pinrep,decomposedloss,realserie,datespan  = mixeddiscrs[i],mxpinreps[i],mxdecomposedlosses[i],mx_realseries[i],mx_dates[i]
            realserieloss = realserie[indices_a_exclure]
            profondeur_test = profondeurs[indices_a_exclure]
            startdate,stopdate = datespan[1], datespan[end]
            datestr = "//"*Dates.format(startdate, "yyyy-mm-dd")*"---"*Dates.format(stopdate, "yyyy-mm-dd")
            partitionpath = mixedpath*datestr 
            mkdir(partitionpath)
            
            totalLoss = []
            physloss = []
            dataloss = []
            bcloss = []
            iteration = 1
            callback = function callback(p,l,args...)
                bc_loss_arr,pde_loss_arr,add_loss_arr = args
                push!(totalLoss,l)
                push!(physloss,lambda*pde_loss_arr[1])
                push!(bcloss,0.0*bc_loss_arr[1])
                push!(dataloss,1.0*add_loss_arr[1])
                # println("Current loss is : $l")
                
                iteration  = iteration +1
                if iteration % 500 == 0
                    print("Mixed lambda: $lambda , n°$(floor(i))  ")
                    show_progress(iteration,21 * nbiter_adam + nbiter_lbfgs)
                end
                return false    
            end
            time = 0
            phi = nothing 
            θ = nothing
            if lambda == 0.
                network,time =@timed trainPINN(dis,pinrep,decomposedloss,callback;nbiteradam = 1, nbiterlbgs = 100,adamlambda = 0.01)
                phi,θ = network
            else
                network,time =@timed trainPINN(dis,pinrep,decomposedloss,callback;nbiteradam = nbiteradam, nbiterlbgs = nbiterlbfgs,adamlambda = 0.01)
                phi,θ = network
            end

            if length(indices_profondeurs) == 5
                pltloss = plot([totalLoss,physloss,bcloss,dataloss],label = ["total" "phy" "bc" "dataloss"])
                plotPINN((phi,θ),realserie,datespan,0,lambda,pltloss;anim = true,save=true,path = partitionpath)
                params = (θ = θ,testloss = -1,totalLoss = totalLoss, dataloss = dataloss,physloss = physloss,time = time)
            else
                testloss = getTestLoss((phi,θ),realserieloss,profondeur_test)
                pltloss = plot([totalLoss,physloss,bcloss,dataloss],label = ["total" "phy" "bc" "dataloss"])
                plotPINN((phi,θ),realserie,datespan,testloss,lambda,pltloss;anim = true,save=true,path = partitionpath)
                params = (testloss = testloss,totalLoss = totalLoss.|>Float64, dataloss = dataloss.|>Float64,physloss = physloss.|>Float64,time = time)            end
            saveTuple(partitionpath*"/params.h5",params)
            GC.gc()
        end


    end


    

end

## définition des paramètres du problème et résolution
nom_exp = "test_pourloss"
lambdas = [0.005,0.0000005]
#lambdas = [0.0001,0.0005,0.5,0.1,0.005,0.001,0.05,0.01]
indices_profondeurs = [1,2,3,5]
nb_res_st = 1
nb_res_mx= 1
nbiter_adam = 300
nbiter_lbfgs = 100

if nprocs()>4
    main(nom_exp,lambdas,indices_profondeurs,nb_res_st,nb_res_mx,nbiter_adam,nbiter_lbfgs)
end

