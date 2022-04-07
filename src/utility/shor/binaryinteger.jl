

struct BinaryInteger
    value::Int
    svalue::Vector{Int}
end

get_value(x::BinaryInteger) = x.value
get_svalue(x::BinaryInteger) = x.svalue
num_bits(x::BinaryInteger) = length(get_svalue(x))

Base.length(x::BinaryInteger) = length(get_svalue(x))
Base.getindex(x::BinaryInteger, i::Int) = getindex(get_svalue(x), i)
Base.iterate(x::BinaryInteger) = iterate(get_svalue(x))
Base.iterate(x::BinaryInteger, state) = iterate(get_svalue(x), state)
Base.IndexStyle(::Type{<:BinaryInteger}) = IndexLinear()
Base.firstindex(x::BinaryInteger) = firstindex(get_svalue(x))
Base.lastindex(x::BinaryInteger) = lastindex(get_svalue(x))

function BinaryInteger(v::Int, Nbits::Int)
    svalue = digits(v, base=2, pad=Nbits)
    (length(svalue) == Nbits) || error("$Nbits is not enough for integer $v.")
    return BinaryInteger(v, reverse(svalue) )
end

function modular_inverse(a::BinaryInteger, N::BinaryInteger)
	(num_bits(a) == num_bits(N)) || error("nbits mismatch.")
	r = invmod(get_value(a), get_value(N))
	return BinaryInteger(r, num_bits(a))
end

function modular_multiply(a::Integer, b::Integer, N::BinaryInteger)
	r = (a * b) % get_value(N)
	return BinaryInteger(r, num_bits(N))
end

function modular_multiply(a::BinaryInteger, b::Integer, N::BinaryInteger)
	(num_bits(a) == num_bits(N)) || error("nbits mismatch.")
	return modular_multiply(get_value(a), b, N)
end

function modular_multiply(a::BinaryInteger, b::BinaryInteger, N::BinaryInteger)
	((num_bits(a) == num_bits(b)) && (num_bits(a) == num_bits(N))) || error("nbits mismatch.")
	return modular_multiply(get_value(a), get_value(b), N)
end

function modular_pow(bs::Integer, exponent::Integer, modulus::Integer)
	(modulus==1) && return 0
	if exponent < 0
		bs = modular_inverse(bs, modulus)
		exponent = -exponent
	end
	result=1
	bs = bs % modulus
	while exponent > 0
		if (exponent % 2 == 1)
			result = (result*bs) % modulus
		end
		exponent = exponent >> 1
		bs = (bs*bs) % modulus
	end
	return result
end
