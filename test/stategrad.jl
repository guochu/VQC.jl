push!(LOAD_PATH, "../src")

using VQC: qstate, qrandn, simple_gradient, distance, check_gradient, probabilities
using VQC: get_coef_sizes_1d, variational_circuit_1d
using LinearAlgebra: dot

using Zygote

"""
	state gradient with distance loss function
"""
function state_grad_dot_real(L::Int, depth::Int)
	target_state = qrandn(Complex{Float64}, L)
	initial_state = qstate(Complex{Float64}, L)
	x0 = randn(get_coef_sizes_1d(L, depth))
	circuit =  variational_circuit_1d(L, depth, x0)

	loss(x) = real(dot(target_state, circuit * x))
	return check_gradient(loss, initial_state)
end

"""
	state gradient with distance loss function
"""
function state_grad_dot_imag(L::Int, depth::Int)
	target_state = qrandn(Complex{Float64}, L)
	initial_state = qstate(Complex{Float64}, L)
	x0 = randn(get_coef_sizes_1d(L, depth))
	circuit =  variational_circuit_1d(L, depth, x0)

	loss(x) = imag(dot(target_state, circuit * x))
	return check_gradient(loss, initial_state)
end

"""
	state gradient with distance loss function
"""
function state_grad_dot_abs(L::Int, depth::Int)
	target_state = qrandn(Complex{Float64}, L)
	initial_state = qstate(Complex{Float64}, L)
	x0 = randn(get_coef_sizes_1d(L, depth))
	circuit =  variational_circuit_1d(L, depth, x0)

	loss(x) = abs(dot(target_state, circuit * x))
	return check_gradient(loss, initial_state)
end

"""
	state gradient with distance loss function
"""
function state_grad_distance(L::Int, depth::Int)
	target_state = qrandn(Complex{Float64}, L)
	initial_state = qstate(Complex{Float64}, L)
	x0 = randn(get_coef_sizes_1d(L, depth))
	circuit =  variational_circuit_1d(L, depth, x0)

	loss(x) = distance(target_state, circuit * x)
	return check_gradient(loss, initial_state)
end

"""
	state gradient with distance loss function
"""
function state_grad_probabilities(L::Int, depth::Int)
	target_state = qrandn(Complex{Float64}, L)
	initial_state = qstate(Complex{Float64}, L)
	x0 = randn(get_coef_sizes_1d(L, depth))
	circuit =  variational_circuit_1d(L, depth, x0)

	loss(x) = sum(probabilities(circuit * x))
	return check_gradient(loss, initial_state)
end

@testset "gradient of quantum state with loss function real(dot(a, circuit*x))" begin
	for L in 2:5
		for depth in 0:5
		    @test state_grad_dot_real(L, depth)
		end	    
	end
end

@testset "gradient of quantum state with loss function imag(dot(a, circuit*x))" begin
	for L in 2:5
		for depth in 0:5
		    @test state_grad_dot_imag(L, depth)
		end	    
	end
end

@testset "gradient of quantum state with loss function abs(dot(a, circuit*x))" begin
	for L in 2:5
		for depth in 0:5
		    @test state_grad_dot_abs(L, depth)
		end	    
	end
end

@testset "gradient of quantum state with loss function distance(a, circuit*x)" begin
	for L in 2:5
		for depth in 0:5
		    @test state_grad_distance(L, depth)
		end	    
	end
end

@testset "gradient of quantum state with loss function sum(probabilities(circuit*x))" begin
	for L in 2:5
		for depth in 0:5
		    @test state_grad_probabilities(L, depth)
		end	    
	end
end
