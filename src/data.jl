"""
    minibatch_next_train!(nlp::LuxNLPModel)

Advances the internal DataLoader to the next batch. 
If the epoch is complete, it automatically restarts the iterator.
"""
function minibatch_next_train!(nlp::LuxNLPModel)
    # Attempt to pull the next batch
    iter = iterate(nlp.data_loader, nlp.iter_state)
    
    if iter === nothing
        # Epoch finished. Restart the iterator from the beginning.
        iter = iterate(nlp.data_loader)
    end
    
    # Mutate the struct to hold the new batch and new state
    nlp.current_batch, nlp.iter_state = iter
    
    return nothing
end