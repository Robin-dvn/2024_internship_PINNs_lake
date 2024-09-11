"""
    timeseries_dataset_from_table(data, targets, sequence_length;
                                  sequence_stride=1,
                                  sampling_rate=1,
                                  batch_size=128,
                                  shuffle=false,
                                  seed=nothing,
                                  start_index=nothing,
                                  end_index=nothing)

Creates a dataset of sliding windows over a timeseries provided as an array.

This function takes in a sequence of data-points gathered at equal intervals, along with time series parameters such as length of the sequences/windows, spacing between two sequence/windows, etc., to produce batches of timeseries inputs and targets.

# Arguments
- `data`: An array containing consecutive data points (timesteps). Axis 1 is expected to be the time dimension.
- `targets`: Targets corresponding to timesteps in `data`. `targets[i]` should be the target corresponding to the window that starts at index `i`.
- `sequence_length`: Length of the output sequences (in number of timesteps).
- `sequence_stride`: Period between successive output sequences. For stride `s`, output samples would start at index `data[i]`, `data[i + s]`, `data[i + 2 * s]`, etc.
- `sampling_rate`: Period between successive individual timesteps within sequences. For rate `r`, timesteps `data[i], data[i + r], ... data[i + sequence_length]` are used for creating a sample sequence.
- `batch_size`: Number of timeseries samples in each batch (except maybe the last one). If `nothing`, the data will not be batched (the dataset will yield individual samples).
- `shuffle`: Whether to shuffle output samples, or instead draw them in chronological order.
- `seed`: Optional random seed for shuffling.
- `start_index`: Optional; data points earlier (exclusive) than `start_index` will not be used in the output sequences. This is useful to reserve part of the data for test or validation.
- `end_index`: Optional; data points later (exclusive) than `end_index` will not be used in the output sequences. This is useful to reserve part of the data for test or validation.

Returns a dataset instance. If `targets` was passed, the dataset yields tuple `(batch_of_sequences, batch_of_targets)`. If not, the dataset yields only `batch_of_sequences`.
"""
function timeseries_dataset_from_table(
    data::AbstractVecOrMat,
    targets::AbstractVecOrMat,
    sequence_length::Integer;
    sequence_stride=1,
    sampling_rate=1,
    batch_size=1,
    shuffle=false,
    # rng=nothing,
    start_index=1,
    end_index=size(data, 1)
)

    N_row = size(data, 1)

    start_index < 1 && throw(ArgumentError("`start_index` must be 1 or greater. Received: start_index=$start_index"))

    start_index > N_row && throw(ArgumentError("`start_index` must be lower than the length of the data. Received: start_index=$start_index, for data of length $(N_row)"))

    end_index ≤ start_index && throw(ArgumentError("`end_index` must be higher than `start_index`. Received: start_index=$start_index, and end_index=$end_index"))

    end_index > N_row && throw(ArgumentError("`end_index` must be lower than the length of the data. Received: end_index=$end_index, for data of length $(N_row)"))

    end_index < 1 && throw(ArgumentError("`end_index` must be higher than 0. Received: end_index=$end_index"))

    # Validate strides
    sampling_rate ≤ 0 && throw(ArgumentError("`sampling_rate` must be higher than 0. Received: sampling_rate=$sampling_rate"))
    
    sampling_rate > N_row && throw(ArgumentError("`sampling_rate` must be lower than the length of the data. Received: sampling_rate=$sampling_rate, for data of length $(N_row)"))
    
    sequence_stride ≤ 0 && throw(ArgumentError("`sequence_stride` must be higher than 0. Received: sequence_stride=$sequence_stride"))

    sequence_stride > N_row && throw(ArgumentError("`sequence_stride` must be lower than the length of the data. Received: sequence_stride=$sequence_stride, for data of length $(N_row)"))

    # Generate start positions
    num_seqs = end_index - start_index - (sequence_length - 1) * sampling_rate

    start_positions = start_index:sequence_stride:num_seqs

    # if shuffle
    #     Random.shuffle!(rng, start_positions)
    # end

    # For each initial window position, generates indices of the window elements
    indices = [(i, i + (sequence_length * sampling_rate-1), sampling_rate) for i in start_positions]

    # Create the dataset of sequences
    dataset = sequences_from_indices(@view(data[start_index:end_index, :]), indices)
    

    # If targets are provided, create dataset for targets as well
    target_indices = start_positions
   
    target_ds = cat3d([@view(targets[start_index:end_index, :])[i, :] for i in target_indices])
    # target_ds = sequences_from_indices(@view(targets[start_index:end_index, :]), target_indices)
     

    return DataLoader((dataset, target_ds); batchsize=batch_size, shuffle = shuffle)
end

function timeseries_dataset_from_table(
    data::AbstractVecOrMat,
    sequence_length::Integer;
    sequence_stride=1,
    sampling_rate=1,
    batch_size=1,
    shuffle=false,
    # rng=nothing,
    start_index=1,
    end_index=size(data, 1)
)

    N_row = size(data, 1)

    start_index < 1 && throw(ArgumentError("`start_index` must be 1 or greater. Received: start_index=$start_index"))

    start_index > N_row && throw(ArgumentError("`start_index` must be lower than the length of the data. Received: start_index=$start_index, for data of length $(N_row)"))

    end_index ≤ start_index && throw(ArgumentError("`end_index` must be higher than `start_index`. Received: start_index=$start_index, and end_index=$end_index"))

    end_index > N_row && throw(ArgumentError("`end_index` must be lower than the length of the data. Received: end_index=$end_index, for data of length $(N_row)"))

    end_index < 1 && throw(ArgumentError("`end_index` must be higher than 0. Received: end_index=$end_index"))

    # Validate strides
    sampling_rate ≤ 0 && throw(ArgumentError("`sampling_rate` must be higher than 0. Received: sampling_rate=$sampling_rate"))
    
    sampling_rate > N_row && throw(ArgumentError("`sampling_rate` must be lower than the length of the data. Received: sampling_rate=$sampling_rate, for data of length $(N_row)"))
    
    sequence_stride ≤ 0 && throw(ArgumentError("`sequence_stride` must be higher than 0. Received: sequence_stride=$sequence_stride"))

    sequence_stride > N_row && throw(ArgumentError("`sequence_stride` must be lower than the length of the data. Received: sequence_stride=$sequence_stride, for data of length $(N_row)"))

    # Generate start positions
    num_seqs = end_index - start_index - (sequence_length - 1) * sampling_rate

    start_positions = start_index:sequence_stride:num_seqs

    # For each initial window position, generates indices of the window elements
    indices = [(i, i + sequence_length * sampling_rate, sampling_rate) for i in start_positions]
    @show size(data[start_index:end_index, :])
    @show num_seqs
    
    # Create the dataset of sequences
    dataset = sequences_from_indices(@view(data[start_index:end_index, :]), indices)

    return DataLoader(dataset; batchsize=batch_size, shuffle = shuffle)
end



"""
    sequences_from_indices(array, indices, start_index, end_index)

Generates sequences from given indices and cast them in a (F)Lux format.

# Arguments
- `array`: The array from which sequences will be generated.
- `indices`: A collection of tuples representing start, end, and step of each sequence.

Returns an array of sequences.
"""
sequences_from_indices(array::AbstractMatrix, indices) =            permutedims(cat3d([array[i[1]:i[3]:i[2], :] for i in indices]), (2, 1, 3))
# sequences_from_indices(array::AbstractVector, indices) = [array[i[1]:i[3]:i[2]] for i in indices]

sequences_from_indices(array::AbstractMatrix, indices::StepRange) = permutedims(cat3d([array[i:1:i, :] for i in indices]), (2, 1, 3))

# https://discourse.julialang.org/t/cat-allocates-too-much-memory/50443/7
cat3d(A) = reshape(reduce(hcat, A), size(A[1])..., :)
cat3d_perm(A) = permutedims(cat3d(A), (1, 3, 2))
# A = [data[i[1]:i[3]:i[2], :] for i in indices];
# @btime cat3d($A)
# @btime reduce((x,y)->cat(x,y, dims=3), $A) 