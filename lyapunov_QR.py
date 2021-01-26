import numpy as np
from scipy.linalg import orth

def dxdt(x,model_params):
    # Define the function here
    return x
def dxdt_tan(x,dx,model_params):
    # Define the tangent map here
    return dx
def dxdt_lorenz63(x, sigma = 10., beta = 8/3, rho = 28.):
    return np.array([sigma*(- x[0] + x[1]),\
                     rho*x[0] - x[1] - x[0]*x[2],\
                     x[0]*x[1]-beta*x[2]])
def dxdt_lorenz63_tan(x, dx, sigma = 10., beta = 8/3, rho = 28.):
    return np.array([sigma*(- dx[0] + dx[1]),\
                     rho*dx[0] - x[0]*dx[2] - dx[0]*x[2] - dx[1],\
                     dx[0]*x[1]+x[0]*dx[1]-beta*dx[2]])
def rk4_integrator(x,tau,dxdt):
    # 4th order runge-kutta integrator for systems with no explicit time dependence
    k1 = dxdt(x)
    k2 = dxdt(x + k1/2*tau)
    k3 = dxdt(x + k2/2*tau)
    k4 = dxdt(x + tau*k3)
    
    xnext = x + 1/6*tau*(k1+2*k2+2*k3+k4)
    return xnext
def rk4_tan_integrator(x,dx,tau,dxdt):
    # 4th order runge-kutta integrator for tangent map of systems with no explicit time dependence
    k1 = dxdt(x,dx)
    k2 = dxdt(x,dx + k1/2*tau)
    k3 = dxdt(x,dx + k2/2*tau)
    k4 = dxdt(x,dx + tau*k3)
    
    dxnext = dx + 1/6*tau*(k1+2*k2+2*k3+k4)
    return dxnext
    
def lyapunov_lorenz63(sigma, beta, rho):
    num_elements  = 3     # Dimension of dynamical system
    num_exponents = 3     # Number of exponents to be calculated, must be <= dynamical system dimension
    tau           = 0.01  # Integration time step, !!MUST BE SMALL!!
    iterations    = 20000 # Total number of algorithm iterations
    iteration_length = 1000 # Number of steps in each iteration before normalization
    transient = 50000      # Length of transient
    
    dxdt = lambda x: dxdt_lorenz63(x,sigma,beta,rho)
    dxdt_tan = lambda x,dx: dxdt_lorenz63_tan(x,dx,sigma,beta,rho)
    
    x = np.random.rand(num_elements)
    for i in range(transient):
        x = rk4_integrator(x,tau,dxdt) # Allow initial condition to evolve to attractor

    vecs = np.random.rand(num_elements,num_exponents) # Initialize tangent vectors
    vecs = orth(vecs)

    Rs = np.zeros((num_exponents,iterations))

    # Begin computation
    for iterate in range(iterations):
        for step in range(iteration_length):
            for exponent in range(num_exponents):
                vecs[:,exponent] = rk4_tan_integrator(x,vecs[:,exponent],tau,dxdt_tan) # Compute tangent vector evolution
            x = rk4_integrator(x,tau,dxdt_lorenz63) # Compute vector evolution

        Q,R = np.linalg.qr(vecs) # Find QR factoriation
        Rs[:,iterate] = np.diag(R) # Extract lyapunov numbers over this iteration
        vecs = np.copy(Q) # Set renormalized tangent vectors

    Rs = Rs + 0.0j
    total_time = tau*iterations*iteration_length
    lyapunov_exps = np.real(np.sum(np.log(Rs),axis = 1))/total_time # Calculate exponents
    print(lyapunov_exps)
    return lyapunov_exps
