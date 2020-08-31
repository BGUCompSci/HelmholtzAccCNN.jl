include("flux_components.jl");

# Multigrid Helmholtz Shifted Laplacian Methods

function get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)
    shifted_laplacian_matrix = kappa .* kappa .* omega .* (omega .- im .* gamma .- im .* omega .* alpha)
    helmholtz_matrix = kappa .* kappa .* omega .* (omega .- im .* gamma)

    return shifted_laplacian_matrix, helmholtz_matrix
end

function jacobi_helmholtz_method!(n, h, x, b, matrix; max_iter=1, w=0.8)
    for i in 1:max_iter
        y = helmholtz_chain!(reshape(x, n-1, n-1, 1, 1), matrix; h=h)
        residual = b - y[:,:,1,1]
        d = 4.0 * ((-1.0 / h^2) .+ matrix)
        x = x + w * (1.0 ./ d) .* residual
    end
    return x
end

function compute_gmres_alpha!(residual, helmholtz_matrix, n, h)
    helmholtz_residual = helmholtz_chain!(reshape(residual, n-1, n-1, 1, 1), helmholtz_matrix; h=h)[:,:,1,1]
    residual_vector = reshape(residual,length(residual),1)
    helmholtz_residual_vector = reshape(helmholtz_residual,length(helmholtz_residual),1)
    alpha = (residual_vector' * helmholtz_residual_vector) / (helmholtz_residual_vector' * helmholtz_residual_vector)
    return alpha
end

function v_cycle_helmholtz!(n, h, x, b, kappa, omega, gamma; u = 1, v1_iter = 1, v2_iter = 10, use_gmres = 0, alpha= 0.5, log = 0, level = nothing)

    shifted_laplacian_matrix, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=alpha)

    # Relax on Ax = b v1_iter times with initial guess x
    x = jacobi_helmholtz_method!(n, h, x, b, shifted_laplacian_matrix; max_iter=v1_iter)

    if( n % 2 == 0 && n > 4 && (level == nothing || level > 0))

        # Compute residual on fine grid
        x_matrix = reshape(x, n-1, n-1, 1, 1)
        result_helmholtz = helmholtz_chain!(x_matrix, helmholtz_matrix; h=h)
        residual_fine = b - result_helmholtz[:,:,1,1]

        # Compute residual, kappa and gamma on coarse grid
        residual_coarse = down(reshape(residual_fine, n-1, n-1, 1, 1))[:,:,1,1]
        kappa_coarse = down(reshape(kappa, n-1, n-1, 1, 1))[:,:,1,1]
        gamma_coarse = down(reshape(gamma, n-1, n-1, 1, 1))[:,:,1,1]

        # Recursive operation of the method on the coarse grid
        n_coarse = size(residual_coarse,1)+1
        x_coarse = zeros(n_coarse-1, n_coarse-1)
    	for i = 1:u
            x_coarse, helmholtz_matrix_coarse = v_cycle_helmholtz!(n_coarse, h*2, x_coarse, residual_coarse, kappa_coarse, omega, gamma_coarse;
                                                                    u=u, v1_iter=v1_iter, v2_iter=v2_iter, log=log, level = (level == nothing ? nothing : (level-1)))
        end
        x_coarse_matrix = reshape(x_coarse, n_coarse-1, n_coarse-1, 1, 1)

        # Correct
        fine_error = up(x_coarse_matrix)[:,:,1,1]
        gmres_alpha = 1.0
        if use_gmres == 1
            gmres_alpha = compute_gmres_alpha!(fine_error, helmholtz_matrix, n, h)
            if log == 1
                println("n = $(n), gmres_alpha = $(gmres_alpha)")
            end
        end
        x = x + gmres_alpha .* fine_error
        residual_vcycle = b - reshape(helmholtz_chain!(reshape(x, n-1, n-1, 1, 1), helmholtz_matrix; h=h), n-1, n-1)

        if log == 1
            r1 = residual_fine
            r2 = b - reshape(helmholtz_chain!(reshape(x, n-1, n-1, 1, 1), helmholtz_matrix; h=h), n-1, n-1)
            println("n = $(n), norm of x = $(norm(x)), norm of fine_error = $(norm(fine_error)), residual before vcycle =$(norm(r1)), residual after vcycle =$(norm(r2))")
        end
    else
        # Coarsest grid
        x = jacobi_helmholtz_method!(n, h, x, b, shifted_laplacian_matrix; max_iter=v2_iter)

    end

    # Relax on Ax = b v1_iter times with initial guess x
    x = jacobi_helmholtz_method!(n, h, x, b, shifted_laplacian_matrix; max_iter=v1_iter)

    return x, helmholtz_matrix
end

function absorbing_layer!(gamma::Array,pad,ABLamp;NeumannAtFirstDim=false)

    n=size(gamma)

    #FROM ERAN ABL:

    b_bwd1 = ((pad[1]:-1:1).^2)./pad[1]^2;
	b_bwd2 = ((pad[2]:-1:1).^2)./pad[2]^2;

	b_fwd1 = ((1:pad[1]).^2)./pad[1]^2;
	b_fwd2 = ((1:pad[2]).^2)./pad[2]^2;
	I1 = (n[1] - pad[1] + 1):n[1];
	I2 = (n[2] - pad[2] + 1):n[2];

	if NeumannAtFirstDim==false
		gamma[:,1:pad[2]] += ones(n[1],1)*b_bwd2'.*ABLamp;
		gamma[1:pad[1],1:pad[2]] -= b_bwd1*b_bwd2'.*ABLamp;
		gamma[I1,1:pad[2]] -= b_fwd1*b_bwd2'.*ABLamp;
	end

	gamma[:,I2] +=  (ones(n[1],1)*b_fwd2').*ABLamp;
	gamma[1:pad[1],:] += (b_bwd1*ones(1,n[2])).*ABLamp;
	gamma[I1,:] += (b_fwd1*ones(1,n[2])).*ABLamp;
	gamma[1:pad[1],I2] -= (b_bwd1*b_fwd2').*ABLamp;
	gamma[I1,I2] -= (b_fwd1*b_fwd2').*ABLamp;

    return gamma
end
