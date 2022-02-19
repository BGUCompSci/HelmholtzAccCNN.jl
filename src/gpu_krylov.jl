import Base.isempty
using LinearAlgebra
using Printf

mutable struct FGMRESmem
	V    			#::Array
	Z				#::Array
end

function getFGMRESmem(n::Int,flexible::Bool,T::Type,k::Int,nrhs::Int=1)
	if flexible
		return FGMRESmem(zeros(T,n,k*nrhs)|>pu,zeros(T,n,k*nrhs)|>pu);
	else
		return FGMRESmem(zeros(T,n,k*nrhs)|>pu,zeros(T,0)|>pu);
	end
end

function getEmptyFGMRESmem()
	return FGMRESmem(zeros(0),zeros(0));
end

function isempty(mem::FGMRESmem)
	return size(mem.V,1)==0;
end

function checkMemorySize(mem::FGMRESmem,n::Int,k::Int,TYPE::Type,flexible::Bool,nrhs::Int=1)
	if isempty(mem)
		# warn("Allocating memory in FGMRES")
		mem = getFGMRESmem(n,flexible,TYPE,k,nrhs);
		return mem
	else
		if size(mem.V,2)!= k*nrhs
			error("FGMRES: size of Krylov subspace is different than inner*nrhs");
		end
	end
end

"""
x,flag,err,iter,resvec = fgmres(A,b,restrt,tol=1e-2,maxIter=100,M=1,x=[],out=0)
(flexible) Generalized Minimal residual ( (F)GMRES(m) ) method with restarts applied to A*x = b.
This is the "right preconditioning" version of GMRES: A*M^{-1}Mx = b. See gmres.jl for "left preconditioning".
Input:
A       - function computing A*x
b       - right hand side vector
restrt  - number of iterations between restarts
tol     - error tolerance
maxIter - maximum number of iterations
M       - preconditioner, function computing M\\x
x       - starting guess
out     - flag for output (0 : only errors, 1 : final status, 2: error at each iteration)
Output:
x       - approximate solution
flag    - exit flag (  0 : desired tolerance achieved,
	-1 : maxIter reached without converging
	-9 : right hand side was zero)
	err     - norm of relative residual, i.e., norm(A*x-b)/norm(b)
	iter    - number of iterations
	resvec  - norm of relative residual at each iteration

	preconditioner M(r) must return a copy of a vector. Cannot reuse memory of r.
	"""
function krylov_cpu_flexible_gmres(A::Function,b,restrt::Int; tol::Real=1e-2,maxIter::Int=100,M::Function=t->copy(t),x,out::Int=0,storeInterm::Bool=false,flexible::Bool=false,mem::FGMRESmem =  getEmptyFGMRESmem())
    t0 = now()
	x,flag,resvec1,iter,resvec2 = KrylovMethods.fgmres(A,b|>cpu,restrt;tol=tol,maxIter=maxIter,M=M,x=x|>cpu,out=out,flexible=flexible) # initialization

	t1 = now()
	@info "$(Dates.format(now(), "HH:MM:SS.sss")) - KrylovMethods.fgmres cpu took: $(t1 - t0)"
	return x,flag,resvec1,iter,resvec2
end

function cpu_flexible_gmres(A::Function,b::Vector,restrt::Int; tol::Real=1e-2,maxIter::Int=100,M::Function=t->copy(t),x::Vector=[],out::Int=0,storeInterm::Bool=false,flexible::Bool=false,mem::FGMRESmem =  getEmptyFGMRESmem())
	# initialization

	t0 = t1 = now()

	n  = length(b)
	TYPE = eltype(b)
	mem = checkMemorySize(mem,n,restrt,TYPE,flexible);

	if norm(b)==0.0
	  return zeros(eltype(b),n),-9,0.0,0,[0.0];
	end

	if Base.isempty(x)
		x = zeros(eltype(b),n)
	  	r = copy(b);
	elseif norm(x) < eps(real(TYPE))
	  	r = copy(b);
	else
	  	r = copy(b);
	  	r.-=A(x);
	end

	if eltype(b) <: Complex
		x = complex(x)
	 end

	if storeInterm
	  	X = zeros(eltype(b),n,maxIter)	# allocate space for intermediates
	end

	betta =  r_type(norm(r))
	rnorm0 =  r_type(norm(b))

	if rnorm0 == 0.0; rnorm0 = 1.0; end

	err = norm( r )/rnorm0
	if err < tol; return x, err; end

	constOne = one(TYPE);
	constZero = zero(TYPE);

	restrt = min(restrt,n-1)
	Z = mem.Z;
	V = mem.V;
	H = zeros(TYPE,restrt+1,restrt)
	xi = zeros(TYPE,restrt+1);
	t = zeros(TYPE,restrt);

	resvec = zeros(restrt*maxIter)
	times = zeros(restrt*maxIter)
	if out==2
	  	println(@sprintf("=== fgmres ===\n%4s\t%7s\n","iter","relres"))
	end
	flag = -1

	counter = 0
	w = zeros(TYPE,0);
	iter = 0
	while iter < maxIter
		iter+=1;
		xi[1] = betta;
		BLAS.scal!(n,(1/betta)*constOne,r,1); # w = w./betta
		H[:] .= 0.0;
		t[:] .= 0.0;

		if out==2;; print(@sprintf("%3d\t", iter));end

		for j = 1:restrt
			if j==1
				V[:,j] = r;  # no memory problem with this line....
				z = M(r)
			else
				V[:,j] = w;  # no memory problem with this line....
				z = M(w)
			end
			if flexible
				Z[:,j] = z;
			end
			w = A(z); # w = A'*z;

			counter += 1;

			# ## modified Gram Schmidt:
			# for i=1:j
				# H[i,j] = dot(vec(V[:,i]),w);
				# w = w - H[i,j]*vec(V[:,i]);
			# end

		  	# Gram Schmidt (much faster than MGS even though zeros are multiplied, does relatively well):
			BLAS.gemv!('C', constOne, V, w,constZero,t);# t = V'*w;
			t[j+1:end] .= 0.0;
			H[1:restrt,j] = t;
			BLAS.gemv!('N', -constOne, V, t,constOne,w);# w = w - V*t;

			betta = norm(w);
			H[j+1,j] = betta;

			BLAS.scal!(n,(1/betta)*constOne,w,1); # w = w*(1/betta);

			# the following 2 lines are equivalent to the 2 next
			# y = H[1:j+1,1:j]\xi[1:j+1];
			# err = norm(H[1:j+1,1:j]*y - xi[1:j+1])/rnorm0
			Q = qr(H[1:j+1,1:j]).Q;
			err = abs(Q[1,end]*xi[1])/rnorm0

			if out==2 print(@sprintf("%1.1e ", err)); end
			resvec[counter] = err;
	        t1 = now()
			times[counter] = Dates.value.(t1-t0)
			if err <= tol
				if flexible
					Z[:,j+1:end] .= 0.0;
				else
					V[:,j+1:end] .= 0.0;
				end
				if out==2; print("\n"); end
				flag = 0; break
			end
		end # end for j to restrt

		y = pinv(H)*xi;

		if flexible
			BLAS.gemv!('N', constOne, Z, y,constZero,w); # w = Z*y  #This is the correction that corresponds to the residual.
		else
			BLAS.gemv!('N', constOne, V, y, constZero, w); #w = V*y
			z = M(w);
			w[:] = z;
		end

		x .+= w;

		if storeInterm; X[:,iter] = x; end

		if out==2; print("\n"); end
		if err <= tol
			flag = 0;
			break
		end
		if iter < maxIter
			r = copy(b);
			r.-=A(x);
			betta = norm(r)
		end

		# if out==2; print(@sprintf("\t %1.1e\n", err)); end
	end #for iter to maxiter
	if out>=0
		if flag==-1
			println(@sprintf("fgmres iterated maxIter (=%d) outer iterations without achieving the desired tolerance. Acheived: %1.2e",maxIter,err))
		elseif flag==0 && out>=1
			println(@sprintf("fgmres achieved desired tolerance at inner iteration %d. Residual norm is %1.2e.",counter,resvec[counter]))
		end
	end

	if storeInterm
		return X[:,1:iter],flag,resvec[counter],iter,resvec[1:counter],times[1:counter]
	else
		return x,flag,resvec[counter],iter,resvec[1:counter],times[1:counter]
	end
end #fgmres


function gpu_flexible_gmres(A::Function,b,restrt; tol,maxIter,M::Function,x,out::Int=0,storeInterm::Bool=false,flexible::Bool=false,mem::FGMRESmem =  getEmptyFGMRESmem())
	t0 = t1 = now()
	# initialization
	n  = length(b)
	TYPE = eltype(b)
	mem = checkMemorySize(mem,n,restrt,TYPE,flexible);

	if norm(b)==0.0
	  	return zeros(eltype(b),n)|>pu,-9,0.0,0,[0.0];
	end

	if Base.isempty(x)
	  	x = zeros(eltype(b),n)|>pu
	  	r = copy(b)|>pu;
	elseif norm(x) < eps(real(TYPE))
	  	r = copy(b)|>pu;
	else
	  	r = copy(b)
	  	r= r .- A(x)
	end

	if storeInterm
	  	X = zeros(eltype(b),n,maxIter)|>pu	# allocate space for intermediates
	end

	betta = r_type(norm(r));
	rnorm0 = r_type(norm(b));

	if rnorm0 == 0.0; rnorm0 = r_type(1.0); end

	err = norm( r )/rnorm0
	if err < tol; return x, err; end

	constOne = one(TYPE);
	constZero = zero(TYPE);

	restrt = min(restrt,n-1)
	Z = mem.Z;
	V = mem.V;
	H = zeros(TYPE,restrt+1,restrt)
	xi = zeros(TYPE,restrt+1)|>pu;
	t = zeros(TYPE,restrt)|>pu;

	resvec = zeros(restrt*maxIter)
	times = zeros(restrt*maxIter)
	if out==2
	   	println(@sprintf("=== fgmres ===\n%4s\t%7s\n","iter","relres"))
	end

	flag = -1

	counter = 0
	w = zeros(TYPE,n)|>pu;
	iter = 0
	while iter < maxIter
		iter+=1;
		xi[1] = betta;
		r=r ./ betta  # BLAS.scal!(n,(1/betta)*constOne,r,1); # w = w./betta
		H[:] .= r_type(0.0);
		t[:] .= r_type(0.0);

	  	if out==2;; print(@sprintf("%3d\t", iter));end

		for j = 1:restrt
			if j==1
				V[:,j] = r;  # no memory problem with this line....
				z = M(r)
			else
				V[:,j] = w;  # no memory problem with this line....
				z = M(w)
			end
			if flexible
				Z[:,j] = z;
			end
			w = A(z)# w = A'*z;

	  		counter += 1;

			# ## modified Gram Schmidt:
			# for i=1:j
				# H[i,j] = dot(vec(V[:,i]),w);
				# w = w - H[i,j]*vec(V[:,i]);
			# end

			# Gram Schmidt (much faster than MGS even though zeros are multiplied, does relatively well):
			d_vw = Dense(V', [r_type(0.0)])|>pu
			t = d_vw(w|>pu) # BLAS.gemv!('C', constOne, V, w,constZero,t);# t = V'*w;
			t[j+1:end] .= r_type(0.0);
			H[1:restrt,j] = t;

			d_vt = Dense(-V, w)|>pu
			w = d_vt(t|>pu) # BLAS.gemv!('N', -constOne, V, t,constOne,w);# w = w - V*t;
			betta = r_type(norm(w));
			H[j+1,j] = betta;

			w = w ./ betta #  BLAS.scal!(n,(1/betta)*constOne,w,1);

			# the following 2 lines are equivalent to the 2 next
			# y = H[1:j+1,1:j]\xi[1:j+1];
			# err = norm(H[1:j+1,1:j]*y - xi[1:j+1])/rnorm0
			Q = qr(H[1:j+1,1:j]).Q;
			err = abs(Q[1,end]*xi[1])/rnorm0

			if out==2 print(@sprintf("%1.1e ", err)); end
			resvec[counter] = err;
	        t1 = now()
			times[counter] = Dates.value.(t1-t0)
			if err <= tol
				if flexible
					Z[:,j+1:end] .= r_type(0.0);
				else
					V[:,j+1:end] .= r_type(0.0);
				end
				if out==2; print("\n"); end
				flag = 0; break
			end
		end # end for j to restrt
		pin_h = pinv(H)
		d_hx = Dense(pin_h, [0.0])|>pu
		y = d_hx(xi)

		if flexible
			d_zy = Dense(Z|>cpu, [0.0])|>pu
		 	w = d_zy(y|>pu) #BLAS.gemv!('N', constOne, Z, y,constZero,w); # This is the correction that corresponds to the residual.
		else
		 	w = V*y #BLAS.gemv!('N', constOne, V, y, constZero, w);
			z = M(w);
			w[:] = z;
		end
		x = x .+ w
		if storeInterm; X[:,iter] = x; end

		if out==2; print("\n"); end
		if err <= tol
			flag = 0;
			break
		end
		if iter < maxIter
			r = copy(b);
			r.-=A(x|>pu);
			betta = norm(r)
		end
	end #for iter to maxiter
	if out>=0
		if flag==-1
			println(@sprintf("fgmres iterated maxIter (=%d) outer iterations without achieving the desired tolerance. Acheived: %1.2e",maxIter,err))
		elseif flag==0 && out>=1
			println(@sprintf("fgmres achieved desired tolerance at inner iteration %d. Residual norm is %1.2e.",counter,resvec[counter]))
		end
	end

	if storeInterm
		return X[:,1:iter],flag,resvec[counter],iter,resvec[1:counter],times[1:counter]
	else
		return x,flag,resvec[counter],iter,resvec[1:counter],times[1:counter]
	end
end #fgmres
