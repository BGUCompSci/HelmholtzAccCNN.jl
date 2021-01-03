include("../flux_components.jl");

# Multigrid Laplacian Methods

function jacobi_laplacian_method!(n, h, x, b; max_iter=1, w=0.8)
    for i in 1:max_iter
        result_laplacian = laplacian_conv!(reshape(x, n-1, n-1, 1, 1); h=h)
        residual = b - reshape(result_laplacian, length(result_laplacian), 1)
        d = 4.0 / h^2
        x = x + w * (1.0/d) * residual
    end
    return x
end

function v_cycle_laplacian!(n, h, x, b; u = 1, v1_iter = 1, v2_iter = 10, log = 0)

    # Relax on Ax = b v1_iter times with initial guess x
    x = jacobi_laplacian_method!(n, h, x, b; max_iter=v1_iter)

    if( n%2 == 0 && n > 4 )

        # Compute residual on fine grid
        result_laplacian = laplacian_conv!(reshape(x, n-1, n-1, 1, 1); h=h)
        residual_fine = b - reshape(result_laplacian, length(result_laplacian), 1)

        # Compute residual on coarse grid
        residual_coarse_matrix = down(reshape(residual_fine, n-1, n-1, 1, 1))
        residual_coarse = reshape(residual_coarse_matrix, length(residual_coarse_matrix), 1)

        # Recursive operation of the method on the coarse grid
        n_coarse = size(residual_coarse_matrix,1)+1
        x_coarse = zeros((n_coarse-1)^2)
        for i = 1:u
            x_coarse = v_cycle_laplacian!(n_coarse, h*2, x_coarse, residual_coarse; u=u, v1_iter=v1_iter, v2_iter=v2_iter, log=log)
        end

        # Correct
        fine_error_matrix = up(reshape(x_coarse, n_coarse-1, n_coarse-1, 1, 1))
        fine_error = reshape(fine_error_matrix, length(fine_error_matrix), 1)
        x = x + fine_error

        if log == 1
            r1 = residual_fine
            r2 = b - reshape(laplacian_conv!(reshape(x, n-1, n-1, 1, 1); h=h), length(b), 1)
            println("n = $(n), norm of x = $(norm(x)), norm of fine_error = $(norm(fine_error)), residual before vcycle =$(norm(r1)), residual after vcycle =$(norm(r2))")
        end

    else
        # Coarsest grid
        x = jacobi_laplacian_method!(n, h, x, b; max_iter=v2_iter)

    end

    # Relax on Ax = b v1_iter times with initial guess x
    x = jacobi_laplacian_method!(n, h, x, b; max_iter=v1_iter)

    return x
end
