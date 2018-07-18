"""
Fluid Simulation of the Benchmark solution FSI1 proposed 
by Turek and Hron

Notes: 	- All units are in SI units (m,kg,s)
	- This code is tested on FEniCS 2017.2 version

2D 4 "Fluid_domain"
2D 5 "Beam_domain"
"""

# _________________________________________________________________________________________

#%% Import packages, set solver parameters (these are crucial for a succesfull simulation!)
# _________________________________________________________________________________________

from cbcpost import *
from fenics import *
from mshr import *
set_log_level(ERROR)

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["cpp_optimize"] = True

# Initialize parallelized computation and set target folder
processID = MPI.rank(mpi_comm_world())
foldername = "/data/jan/Benchmark_02_results"

# _________________________________________________________________________________________

#%% Define time constants, reference pressure and geometrical data (in m)
# _________________________________________________________________________________________

t_end = 20.		# in s
t_steady = 2.		# in s
dt = 1e-2		# in s

pref = 0.0 		# in Pa

L = 2.5			# channel length
H = 0.41 		# channel heigth
xc = 0.2		# x and y coordinates of the circle center C (in m)
yc = 0.2
r = 0.05		# radius of the circle (in m)
w_beam = 0.02		# beam width (in m)
l_beam = 0.35		# beam length (in m)

# _________________________________________________________________________________________

#%% Create Geometry with mshr
# _________________________________________________________________________________________

mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), '../Geometry/BenchmarkMesh.h5', 'r')
hdf.read(mesh, '/mesh', False)
cells = MeshFunction('size_t', mesh, 2)
hdf.read(cells, '/cells')
facets = MeshFunction('size_t', mesh, 1)
hdf.read(facets, '/facets')

mesh_f = utils.create_submesh(mesh, cells, 4)
mesh_s = utils.create_submesh(mesh, cells, 5)

facets_f = MeshFunction('size_t', mesh_f, 1)
facets_s = MeshFunction('size_t', mesh_s, 1)

cells_f = MeshFunction('size_t', mesh_f, 2)
cells_s = MeshFunction('size_t', mesh_s, 2)

# Measuring elements
da = Measure('ds', domain=mesh_f, subdomain_data=facets_f, metadata={'quadrature_degree': 2})
dv = Measure('dx', domain=mesh_f, subdomain_data=cells_f, metadata={'quadrature_degree': 2})
n = FacetNormal(mesh_f)

dA = Measure('ds', domain=mesh_s, subdomain_data=facets_s, metadata={'quadrature_degree': 2})
dV = Measure('dx', domain=mesh_s, subdomain_data=cells_s, metadata={'quadrature_degree': 2})
N = FacetNormal(mesh_s)

# Parametrizing boundaries of the fluid domain

inflow_profile = Expression(('1.5 * 2.0 * 4.0 / 0.1681 * x[1] * (0.41-x[1]) * (1-cos(pi/2.0 * time))/2.0','0.0'), time=0.0, degree=3) 

inflow   = CompiledSubDomain('near(x[0], 0)')
outflow  = CompiledSubDomain('near(x[0], 2.5)')
walls    = CompiledSubDomain('near(x[1], 0) || near(x[1], 0.41)')
#cylinder = CompiledSubDomain('on_boundary && ((x[0]>0.1 && x[0]<0.3) && (x[1]>0.1 && x[1]<0.3')) )
cylinder = CompiledSubDomain(' ( pow((x[0]-0.2),2) + pow((x[1]-0.2),2) ) <= 0.06*0.06 && on_boundary')
beam 	 = CompiledSubDomain('on_boundary && ( near(x[0], 0.6) || near(x[1], 0.19) || near(x[1], 0.21) )')

facets_f.set_all(0)
inflow.mark(facets_f,1)
outflow.mark(facets_f,2)
walls.mark(facets_f,3)
cylinder.mark(facets_f,4)
beam.mark(facets_f,5)

# Parametrizing boundaries of the structure domain
beamcylinder 	= CompiledSubDomain('on_boundary')
beamfluid 	= CompiledSubDomain('on_boundary && (x[0] > 0.25 || x[1] <= 0.19 || x[1] >= 0.21) ')

facets_s.set_all(0)
beamcylinder.mark(facets_s,1)
beamfluid.mark(facets_s,2)

# _________________________________________________________________________________________

#%% Define the Function spaces for the FEM
# _________________________________________________________________________________________

i, j, k, l, m = indices(5)
delta = Identity(2)

scalar = FiniteElement('P', triangle, 1)
vector1 = VectorElement('P', triangle, 1)
vector2 = VectorElement('P', triangle, 2)
tensor = TensorElement('P', triangle, 1)
mixed_element = MixedElement([scalar, vector1])

Space_f = FunctionSpace(mesh_f, mixed_element)
V_f_space = FunctionSpace(mesh_f, vector1)
T_f_space = FunctionSpace(mesh_f, tensor)

S_s_space = FunctionSpace(mesh_s, scalar)
V_s_space = FunctionSpace(mesh_s, vector1)
T_s_space = FunctionSpace(mesh_s, tensor)

# _________________________________________________________________________________________

#%% Defining solution, test and trial space
# _________________________________________________________________________________________

u_s = Function(V_s_space)
u0_s = Function(V_s_space)
u00_s = Function(V_s_space)
traction = Function(V_s_space)
del_u = TestFunction(V_s_space)
du = TrialFunction(V_s_space)

del_w = TestFunction(V_f_space)
w = Function(V_f_space)
dw = TrialFunction(V_f_space)

du_f = TrialFunction(Space_f)
del_u_f = TestFunction(Space_f)
u_f = Function(Space_f)
u0_f = Function(Space_f)
u00_f = Function(Space_f)

p0, v0 = split(u0_f)
p, v = split(u_f)
dp_f, dv_f = split(du_f)
del_p, del_v = split(del_u_f)

time_list = [0.]
timestep_list = [0]
Xdisplacement_list = [0.]
Ydisplacement_list = [0.]



def simulate(t, timestep, Xdisplacement_list, Ydisplacement_list, dt):

	if t == 0.:
		u_init = Expression( ('p','0.0','0.0' ) , degree=1, p=pref)
		u_f.interpolate(u_init)
		u0_f.assign(u_f)

		u_s_init = Expression(('0.0','0.0' ) , degree=0)
		u_s.interpolate(u_s_init)
		u0_s.assign(u_s)
		u00_s.assign(u_s)

		pp = PostProcessor(dict(casedir=foldername, clean_casedir=True))
		pp.add_fields([
			SolutionField("p", dict(save=True, save_as=["hdf5", "xdmf"] ) ) ,
			SolutionField("v", dict(save=True, save_as=["hdf5", "xdmf"] ) ) ,
			SolutionField("u", dict(save=True, save_as=["hdf5", "xdmf"] ) ) 
			])

		pp.update_all({\
			"p": lambda: u_f.split(deepcopy=True)[0], \
			"v": lambda: u_f.split(deepcopy=True)[1], \
			"u": lambda: u_s}, t, timestep )

		sigma_s = Function(T_s_space)
		sigma_f = Function(T_f_space)

	else:

		res = Restart(dict(casedir=foldername, restart_times=t, solution_names=["p","v","u"], rollback_casedir=True) )
		res_data = res.get_restart_conditions()

		u_f.split()[0].assign(res_data.values()[0]["p"])
		u_f.split()[1].assign(res_data.values()[0]["v"])
		u0_f.assign(u_f)

		u_s.assign(res_data.values()[0]["u"])
		
		pp = PostProcessor(dict(casedir=foldername, clean_casedir=False))
		pp.add_fields([
			SolutionField("p", dict(save=True, save_as=["hdf5", "xdmf"] ) ) ,
			SolutionField("v", dict(save=True, save_as=["hdf5", "xdmf"] ) ) ,	
			SolutionField("u", dict(save=True, save_as=["hdf5", "xdmf"] ) ) ])
			
		pp.update_all({\
			"p": lambda: u_f.split(deepcopy=True)[0], \
			"v": lambda: u_f.split(deepcopy=True)[1], \
			"u": lambda: u_s}, t, timestep )


	#%%%%%%%%%%%%%%%%%%%%%%%% Structure variational form %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	rho_s = 1e0     # in tonne/m^3
	nu_s = 0.4      # in - (Poisson ration)
	mu_s = 2e3     	# in tonne/(ms^2), try some stiffer values for the beginning
	lam_s = 2*nu_s*mu_s / (1-2*nu_s)
	f = Constant(('0.0','0.0'))

	F_s = as_tensor( u_s[k].dx(i) + delta[k,i], (k,i) )
	J_s = det(F_s)
	C_s = as_tensor( F_s[k,i]*F_s[k,j], (i,j) )
	E_s = as_tensor( 1./2.*(C_s[i,j]-delta[i,j]), (i,j) )
	S_s = as_tensor( lam_s*E_s[k,k]*delta[i,j] + 2.*mu_s*E_s[i,j], (i,j) )
	P_s = as_tensor( F_s[i,j]*S_s[k,j], (k,i) )

	bc_s = []
	bc_s.append( DirichletBC(V_s_space, Constant((0, 0)), facets_s, 1) )

	#%%%%%%%%%%%%%%%%%%%%%%%%%%% Mesh variational form %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	a_m = 1e-5	# artificial viscosity in Pa/s (original value: 1e-5)
	Form_m = a_m*sym(grad(w))[i,j]*del_w[i].dx(j)*dv
	Gain_m = derivative(Form_m, w, dw)

	#%%%%%%%%%%%%%%%%%%%%%%%%%% Fluid variational form %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	rho = 1.0		# tonne/m^3
	nu = 1e-3		# m^2/s
	mu = rho*nu 		# in kPa s 
	la = 0.6		# original value: 0.6

	tic()
	success_rate = 0

	while t < t_end:

		if success_rate > 5 : success_rate=0
		if success_rate < -5 : return 'continue', dt
		t += dt
		timestep += 1
		if t > t_steady: inflow_profile.time = t_steady
		else: inflow_profile.time = t
		
		for ii in range(1):
			if processID == 0: print 'Solving step:', ii+1, '/3'

			#%%%%%%%%%%%%%%%%%%% Solve Structure %%%%%%%%%%%%%%%%%%%%%%
			if processID == 0: print('Solving structure deformation...')
			
			t_hat = as_tensor( J_s*inv(F_s)[k,j]*sigma_s[j,i]*N[k] , (i,) )

			Form_s = ( rho_s*(u_s-2.*u0_s+u00_s)[i]/(dt*dt)*del_u[i] \
				+ P_s[k,i]*del_u[i].dx(k) - rho_s*f[i]*del_u[i] )*dV - t_hat[i]*del_u[i]*dA(2)

			Gain_s = derivative(Form_s, u_s, du)

			solve(Form_s == 0, u_s, bc_s, J=Gain_s, \
				solver_parameters={"newton_solver":{"linear_solver": "mumps","relative_tolerance": 1e-3, "maximum_iterations":100}},
				form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs"})			

			v_s = project((u_s-u0_s)/dt, V_f_space, solver_type="mumps", form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2})

			#%%%%%%%%%%%%%%%%%%%%% Solve Mesh %%%%%%%%%%%%%%%%%%%%%%%%%
			if processID == 0: print('Solving mesh velocity...')

			bc_m = []
			bc_m.append( DirichletBC(V_f_space, Constant((0, 0)), facets_f, 1) )
			bc_m.append( DirichletBC(V_f_space, Constant((0, 0)), facets_f, 2) )
			bc_m.append( DirichletBC(V_f_space, Constant((0, 0)), facets_f, 3) )
			bc_m.append( DirichletBC(V_f_space, Constant((0, 0)), facets_f, 4) )
			bc_m.append( DirichletBC(V_f_space, v_s, facets_f, 5) )

			solve(Form_m == 0, w, bc_m, J=Gain_m, \
				solver_parameters={"newton_solver":{"linear_solver": "mumps","relative_tolerance": 1e-3, "maximum_iterations":100}},
				form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs"})	
		
			#%%%%%%%%%%%%%%%%%%%%% Solve Fluid %%%%%%%%%%%%%%%%%%%%%%%%
			if processID == 0: print('Solving fluid dynamic...')

			#%% Define boundary conditions for the fluid
			bc = []
			bc.append( DirichletBC(Space_f.sub(1), inflow_profile, facets_f, 1) )
			bc.append( DirichletBC(Space_f.sub(1), Constant((0, 0)), facets_f, 3) )
			bc.append( DirichletBC(Space_f.sub(1), Constant((0, 0)), facets_f, 4) )
			bc.append( DirichletBC(Space_f.sub(1), Constant((0, 0)), facets_f, 5) )
			bc.append( DirichletBC(Space_f.sub(0), pref, facets_f, 2) )
			# bc.append( DirichletBC(Space.sub(1).sub(1), 0.0, facets_f, 2) )
			
			# _________________________________________________________________________________________

			#%% Picard solver
			# _________________________________________________________________________________________

			L2_error = 1.0  # error measure ||u-u_k||
			tol = 1e-5     	# tolerance
			it = 0          # iteration counter
			maxiter = 200	# iteration limit		
			
			uk = interpolate(u0_f, Space_f)
			p0, v0 = split(u0_f)
			
			while L2_error > tol and it < maxiter:
				
				it += 1				
				p_k, v_k = split(uk)
				d = as_tensor( 1./2.*(dv_f[i].dx(j)+dv_f[j].dx(i)) , [i,j] )
				tau = as_tensor( la*d[k,k]*delta[i,j] + 2.*mu*d[i,j] , [i,j] )
				
				F_1 = dv_f[i].dx(i)*del_p*dv
				F_2 = ( ( rho*(dv_f-v0)[j]/dt  + rho*(v_k[i]*dv_f[j]).dx(i) + dp_f.dx(j) ) *del_v[j]  + tau[i,j]*del_v[j].dx(i)  )*dv
				F_3 = ( (dv_f-v0)[j] + dt*(v_k[i]*dv_f[j]).dx(i) - dt/rho*(-dp_f*delta[i,j] + tau[i,j]).dx(i) ) *del_p.dx(j)*dv

				Form_f = F_1 + F_2 + F_3
				a_f = lhs(Form_f)
				L_f = rhs(Form_f)

				solve(a_f == L_f, u_f, bcs=bc, solver_parameters = {"linear_solver":"mumps"}) 
				L2_error = assemble(((u_f-uk)**2)*dx)
				if processID == 0: print 'it=%d: L2-error=%g' % (it, L2_error)
				uk.assign(u_f)
				
				if it == maxiter and processID == 0: print 'Solver did not converge!'

			dofs=len(u_f.vector())
			p_ = u_f.split(deepcopy=True)[0]
			vel = u_f.split(deepcopy=True)[1]
			Re = 2./3. * vel(Point(0.,0.2))[0]/L*rho/mu
			if processID == 0: print 'time: ', t, 's, Re: ', Re, ' with ',dofs,' dofs, time step took ', toc(), ' s'
			u0_f.assign(u_f)
			success_rate += 1
			dt = dt
			tic()
			
			u00_s.assign(u0_s)
			u0_s.assign(u_s)

			d_ = as_tensor( 1./2.*(vel[i].dx(j)+vel[j].dx(i)) , [i,j] )
			tau_ = as_tensor( la*d_[k,k]*delta[i,j] + 2.*mu*d_[i,j] , [i,j] )
			sigma_f = project(-p_*delta+tau_, T_f_space, solver_type="mumps",\
					form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2})

			sigma_s = project(sigma_f, T_s_space, solver_type="mumps", form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2})

		#for x in mesh.coordinates(): x[:] += dt*w(x)[:]

		mesh_f_old = Mesh(mesh_f)
		Space_f_old = FunctionSpace(mesh_f_old, mixed_element)

		u_f_old = Function(Space_f_old)
		u0_f_old = Function(Space_f_old)
		u_f_old.vector()[:] = u_f.vector().get_local()
		u0_f_old.vector()[:] = u0_f.vector().get_local()

		solvertype="mumps"
		for x in mesh_f.coordinates(): x[:] += dt*w(x)[:]
		u_f.assign( project(u_f_old , Space_f , solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
					      "representation": "uflacs",
					      "quadrature_degree": 2})  )
		u0_f.assign( project(u0_f_old , Space_f , solver_type=solvertype, \
		    form_compiler_parameters={"cpp_optimize": True,
					      "representation": "uflacs",
					      "quadrature_degree": 2}) )


		
		pp.update_all({\
			"p": lambda: u_f.split(deepcopy=True)[0], \
			"v": lambda: u_f.split(deepcopy=True)[1], \
			"u": lambda: u_s}, t, timestep )
		
		time_list.append(t)
		timestep_list.append(timestep)
		Xdisplacement_list.append( u_s(Point(0.6,0.2))[0] )
		Ydisplacement_list.append( u_s(Point(0.6,0.2))[1] )
		
		if processID == 0: 	
			print 'Tip deflection (in m):', u_s(Point(0.6,0.2)), '\n'

		if it == maxiter: t = t_end

	return 'finished', dt, Xdisplacement_list, Ydisplacement_list, time_list

state = 'continue'
rel_param = 0.8
while state == 'continue':
	state, dt, Xdisplacement_list, Ydisplacement_list, time_list = simulate(time_list[-1], timestep_list[-1], Xdisplacement_list, Ydisplacement_list, dt)
	if processID == 0: print 'restarting'
	tic()

plt.figure()
plt.plot(time_list, Xdisplacement_list)
plt.grid(True)
plt.xlabel('Time in s')
plt.ylabel('Tip deflection in x-direction in m')
plt.savefig(foldername+'/Xdisplacement.pdf')

plt.figure()
plt.plot(time_list, Ydisplacement_list)
plt.grid(True)
plt.xlabel('Time in s')
plt.ylabel('Tip deflection in y-direction in m')
plt.savefig(foldername+'/Ydisplacement.pdf')

