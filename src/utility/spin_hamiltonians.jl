


"""
    heisenberg xxz chain
"""
function heisenberg_chain(L::Int; J::Real=1., Jzz::Real=J, hz::Real=0.)
    sp, sm, z = QuantumCircuits._get_op.(["+", "-", "Z"])
    terms = []
    # one site terms
    for i in 1:L
        push!(terms, QubitsTerm(i=>z, coeff=hz))
    end
    # nearest-neighbour interactions
    for i in 1:L-1
        t = QubitsTerm(i=>sp, i+1=>sm, coeff=2*J)
        push!(terms, t)
        push!(terms, t')
        push!(terms, QubitsTerm(i=>z, i+1=>z, coeff=Jzz))
    end
    return simplify(QubitsOperator(terms...))
end
heisenberg_1d(L::Int; kwargs...) = heisenberg_chain(L; kwargs...)

function heisenberg_2d(m::Int, n::Int=m; J::Real=1., Jzz::Real=J, hz::Real=0.)
    sp, sm, z = QuantumCircuits._get_op.(["+", "-", "Z"])
    terms = []
    for i in 1:m*n
        push!(terms, QubitsTerm(i=>z, coeff=hz))
    end
    index = LinearIndices((m, n))
    for i in 1:m
        for j in 1:(n-1)
            t = QubitsTerm(index[i, j]=>sp, index[i, j+1]=>sm, coeff=2*J)
            push!(terms, t)
            push!(terms, t')
            push!(terms, QubitsTerm(index[i, j]=>z, index[i, j+1]=>z, coeff=Jzz) )
        end
    end
    for i in 1:(m-1) 
        for j in 1:n
            t = QubitsTerm(index[i, j]=>sp, index[i+1, j]=>sm, coeff=2*J)
            push!(terms, t)
            push!(terms, t')
            push!(terms, QubitsTerm(index[i, j]=>z, index[i+1, j]=>z, coeff=Jzz) )
        end
    end
    return simplify(QubitsOperator(terms...))
end
heisenberg_2d(shapes::Tuple{Int, Int}; kwargs...) = heisenberg_2d(shapes[1], shapes[2]; kwargs...)


function ising_chain(L::Int; J::Real=1., hz::Real=1.)
    x, z = QuantumCircuits._get_op.(["X", "Z"])
    terms = []
    for i in 1:L
    	push!(terms, QubitsTerm(i=>z, coeff=hz))
    end
    for i in 1:L-1
    	push!(terms, QubitsTerm(i=>x, i+1=>x, coeff=J))
    end
    return simplify(QubitsOperator(terms...))
end
ising_1d(L::Int; kwargs...) = ising_chain(L; kwargs...)

function ising_2d(m::Int, n::Int=m; J::Real=1., hz::Real=1.)
    x, z = QuantumCircuits._get_op.(["X", "Z"])
    terms = []
    for i in 1:m*n
        push!(terms, QubitsTerm(i=>z, coeff=hz))
    end
    index = LinearIndices((m, n))
    for i in 1:m
        for j in 1:(n-1)
            push!(terms, QubitsTerm(index[i, j]=>x, index[i, j+1]=>x, coeff=J) )
        end
    end
    for i in 1:(m-1) 
        for j in 1:n
            push!(terms, QubitsTerm(index[i, j]=>x, index[i+1, j]=>x, coeff=J) )
        end
    end
    return simplify(QubitsOperator(terms...))
end
ising_2d(shapes::Tuple{Int, Int}; kwargs...) = ising_2d(shapes[1], shapes[2]; kwargs...)
