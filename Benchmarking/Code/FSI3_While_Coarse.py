"""
Simulation of the Benchmark solution FSI3 proposed by Turek and Hron
Elastic beam attached to off center placed rigid cylinder in a channel
The channel flow of glycerine leads to deformations of the elastic structure

Subscript _s refers to structure 
Subscript _f refers to fluid
Subscript _ref refers to reference configuration
Subscript _act refers to actual configuration

Domains: - 2D: 6 "Fluid_domain"
	 - 2D: 7 "Beam_domain"
"""

__author__ = "J. Swierczek-Jereczek (jan.jereczek@gmail.com) and B. Emek Abali (bilenemek@abali.org)"
__date__ = "2018-08-13"

# This code is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. 
# 
# This code is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
# GNU Lesser General Public License for more details. 
# 
# For the GNU Lesser General Public License see <http://www.gnu.org/licenses/>. 
#
# This code is tested on FEniCS 2017.2 version


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%% Import packages and set solver parameters %%%%%%%%%%%%%%%%%%%%%% 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from cbcpost import *
from fenics import *
from mshr import *
set_log_level(ERROR)

import numpy as np
import time
import datetime
import pickle
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["cpp_optimize"] = True

# Initialize processor ID for parallelized computation and set target folder
processID = MPI.rank(mpi_comm_world())
foldername = "data/FSI3_While_Coarse"

if processID == 0: print '\n'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%% Define time constants, reference pressure and geometrical data %%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t = 0.0			# Initialize time
t_end = 10.		# End time of the simulation, in s
t_steady = 2.		# Time at which the inlet speed of the fluid gets steady, in s
dt = 1e-2		# Time step of the simulation, in s

pref = 0.0 		# Reference pressure of fluid, in kPa

L = 2.5			# channel length, in m
H = 0.41 		# channel heigth, in m
xc = 0.2		# x and y coordinates of the circle center C, in m
yc = 0.2
r = 0.05		# radius of the circle, in m
w_beam = 0.02		# beam width, in m
l_beam = 0.35		# beam length, in m


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%% Import Geometry from Salome and create the submeshes %%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if processID == 0: print('Generating mesh...')

# Import h5-File
mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), 'BenchmarkMesh_2.h5', 'r')
hdf.read(mesh, '/mesh', False)
cells = MeshFunction('size_t', mesh, 2)
hdf.read(cells, '/cells')
facets = MeshFunction('size_t', mesh, 1)
hdf.read(facets, '/facets')

# Create 3 submeshes: 
mesh_f_ref = utils.create_submesh(mesh, cells, 6)	# Fluid domain in reference configuration
mesh_f_act = utils.create_submesh(mesh, cells, 6)	# Fluid domain in actual configuration
mesh_s = utils.create_submesh(mesh, cells, 7)		# Structure domain in reference configuration

# Define facets and cells for reference configurations 
# (for the actual configuration, it will be defined in the time loop)
facets_f_ref = MeshFunction('size_t', mesh_f_ref, 1)
facets_s = MeshFunction('size_t', mesh_s, 1)

cells_f_ref = MeshFunction('size_t', mesh_f_ref, 2)
cells_s = MeshFunction('size_t', mesh_s, 2)

# Measuring elements and getting facet normal
da_ref = Measure('ds', domain=mesh_f_ref, subdomain_data=facets_f_ref, metadata={'quadrature_degree': 2})
dv_ref = Measure('dx', domain=mesh_f_ref, subdomain_data=cells_f_ref, metadata={'quadrature_degree': 2})
n_ref = FacetNormal(mesh_f_ref)

dA = Measure('ds', domain=mesh_s, subdomain_data=facets_s, metadata={'quadrature_degree': 2})
dV = Measure('dx', domain=mesh_s, subdomain_data=cells_s, metadata={'quadrature_degree': 2})
N = FacetNormal(mesh_s)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%% Parametrizing boundaries and inflow profile of the fluid %%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if processID == 0: print('Parametrizing boundaries...')

# Inflow profile
U = 2.0
inflow_profile = Expression(('1.5 * U_bar * 4.0 / 0.1681 * x[1] * (0.41-x[1]) * (1-cos(pi/2.0 * time))/2.0','0.0'), time=0.0, U_bar=U, degree=3)

# Fluid domain boundaries
inflow   = CompiledSubDomain('near(x[0], 0)')
outflow  = CompiledSubDomain('near(x[0], 2.5)')
walls    = CompiledSubDomain('near(x[1], 0) || near(x[1], 0.41)')
cylinder = CompiledSubDomain(' ( pow((x[0]-0.2),2) + pow((x[1]-0.2),2) ) <= 0.0501*0.0501 && on_boundary')
# beam 	 = CompiledSubDomain('on_boundary && ( ( x[0] <= 0.7 && x[0] >= 0.5 && x[1] >= 0.189 && x[1] <= 0.201) || (x[0] > 0.2 && near(x[1], 0.19)) || (x[0] > 0.2 && near(x[1], 0.21) ))')
beam 	 = CompiledSubDomain('on_boundary && x[0] >= 0.24 && x[0] <= 0.7 && x[1] >= 0.1899 && x[1] <= 0.2101')

# Numerate fluid boundaries
facets_f_ref.set_all(0)
inflow.mark(facets_f_ref,1)
outflow.mark(facets_f_ref,2)
walls.mark(facets_f_ref,3)
cylinder.mark(facets_f_ref,4)
beam.mark(facets_f_ref,5)

#File_facets = File(foldername+'Facets_Fluid.pvd')
#File_facets << facets_f_ref
#exit()

# Structure domain boundaries
beamcylinder 	= CompiledSubDomain('on_boundary')
beamfluid 	= CompiledSubDomain('on_boundary && (x[0] > 0.25 || x[1] <= 0.19 || x[1] >= 0.21) ')

# Numerate structure boundaries
facets_s.set_all(0)
beamcylinder.mark(facets_s,1)
beamfluid.mark(facets_s,2)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Defining some usefull tools %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if processID == 0: print('Initializing tools...')

# indices for tensor calculus, Kroenecker delta in 2D and gravity force (neglegated here)
i, j, k, l, m = indices(5)
delta = Identity(2)
f = Constant(('0.0','0.0'))

# Lists to save for postprocessing of the data
time_list = [0.]
Xdisplacement_list = [0.]
Ydisplacement_list = [0.]

# .pvd-files to display results in ParaView
File_u_s = File(foldername+'/u.pvd')
File_p_f = File(foldername+'/p.pvd')
File_v_f = File(foldername+'/v.pvd')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%% Defining function spaces for the FEM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if processID == 0: print('Defining function spaces...')

scalar = FiniteElement('P', triangle, 1)
vector = VectorElement('P', triangle, 1)
tensor = TensorElement('P', triangle, 1)
mixed_element = MixedElement([scalar, vector])

MixedSpace_f = FunctionSpace(mesh_f_act, mixed_element)

V_f_ref = FunctionSpace(mesh_f_ref, vector)
T_f_ref = FunctionSpace(mesh_f_ref, tensor)

S_s_space = FunctionSpace(mesh_s, scalar)
V_s_space = FunctionSpace(mesh_s, vector)
T_s_space = FunctionSpace(mesh_s, tensor)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%% Defining functions, test functions and trial functions %%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

u_s = Function(V_s_space)		# Structure displacement at t
uk_s = Function(V_s_space)		# Structure displacement at t, previous for-iteration
u0_s = Function(V_s_space)		# Structure displacement at t-dt
u00_s = Function(V_s_space)		# Structure displacement at t-2*dt	
del_u = TestFunction(V_s_space)		# Test function for structure displacement
du = TrialFunction(V_s_space)		# Trial function for structure displacement

u_m = Function(V_f_ref)			# Mesh displacement at t
uk_m = Function(V_f_ref)		# Mesh displacement at t, previous for-iteration
u0_m = Function(V_f_ref)		# Mesh displacement at t-dt
del_u_m = TestFunction(V_f_ref)		# Test function for mesh displacement
du_m = TrialFunction(V_f_ref)		# Trial function for mesh displacement

u_f = Function(MixedSpace_f)		# Fluid pressure and velocity at t 
u0_f = Function(MixedSpace_f)		# Fluid pressure and velocity at t-dt
del_u_f = TestFunction(MixedSpace_f)	# Test function for fluid pressure and velocity
du_f = TrialFunction(MixedSpace_f)	# Trial function for fluid pressure and velocity

# Split the mixed function for fluid FEM
p0, v0 = split(u0_f)
p, v = split(u_f)
dp_f, dv_f = split(du_f)
del_p, del_v = split(del_u_f)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialize solutions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

u_init = Expression( ('p','0.0','0.0' ) , degree=1, p=pref)
u_f.interpolate(u_init)
u0_f.assign(u_f)

u_s_init = Expression( ('0.0','0.0') , degree=0)
u_s.interpolate(u_s_init)
u0_s.assign(u_s)
u00_s.assign(u_s)

t_s_hat = Constant((0.0, 0.0))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%% Constants, tensors, and boundary conditions for strucuture %%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rho_s = 1e0     # Density, in tonne/m^3
nu_s = 0.4      # Poisson ration, in 1
mu_s = 2e3    	# in ton/(ms^2), original value 2e3
lam_s = 2*nu_s*mu_s / (1-2*nu_s)

F_s = as_tensor( u_s[k].dx(i) + delta[k,i], (k,i) )
J_s = det(F_s)
C_s = as_tensor( F_s[k,i]*F_s[k,j], (i,j) )
E_s = as_tensor( 1./2.*(C_s[i,j]-delta[i,j]), (i,j) )
S_s = as_tensor( lam_s*E_s[k,k]*delta[i,j] + 2.*mu_s*E_s[i,j], (i,j) )
P_s = as_tensor( F_s[i,j]*S_s[j,k], (k,i) )

bc_s = []
bc_s.append( DirichletBC(V_s_space, Constant((0, 0)), facets_s, 1) )

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%% Constants, tensors, and boundary conditions for structure %%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nu_m = 0.0    				# Poisson ratio, in 1
mu_m = 1.0/CellSize(mesh_f_ref)**4	# Stifness inverse proportional to cell size, in in ton/(ms^2)
la_m = 2*nu_m*mu_m / (1-2*nu_m)

eps_m = as_tensor(1.0/2.0*(u_m[i].dx(j)+u_m[j].dx(i)), (i,j))
sigma_m = as_tensor( la_m*eps_m[k,k]*delta[i,j] + 2.0*mu_m*eps_m[i,j], (i,j))
a_m = sigma_m[j,i]*del_u_m[i].dx(j)*dv_ref
L_m = 0.0 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fluid constants %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rho = 1.0		# Density, in ton/m^3
nu = 1e-3		# kinematic viscosity, in m^2/s
mu = rho*nu 		# dynamic viscosity, in kPa s 
la = 1e-2		# original value: 0.6

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Time simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t1 = datetime.datetime.now()	# initialize time counting of simulation 
if processID == 0: print('Starting transient simulation... \n')

while t < t_end:

	# Update time and inflow profile
	tic()
	t += dt 
	if t > t_steady: inflow_profile.time = t_steady
	else: inflow_profile.time = t

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#%%%%%%%%%%%%%%%%%%%%%%% Beginning of global FOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	L2_abs = 1.0			# initialize L2-norm for structure deformation
	L2_rel = 1.0
	abs_tol = 1e-9
	min_trial = 2
	count = 0

	uk_s.assign(u0_s)
	uk_m.assign(u0_m)

	while count < min_trial or L2_abs > abs_tol:
	
		count += 1
		if processID == 0: print 'Solving step:', count, ', at time: ', t
		
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		#%%%%%%%%%%%%%%%%%%%%%% Solve structure deformation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		if processID == 0: print '     Solving structure deformation...'

		Form_s = ( rho_s*(u_s-2.*u0_s+u00_s)[i]/(dt*dt)*del_u[i] \
			+ P_s[k,i]*del_u[i].dx(k) - rho_s*f[i]*del_u[i] )*dV - t_s_hat[i]*del_u[i]*dA(2)

		Gain_s = derivative(Form_s, u_s, du)

		solve(Form_s == 0, u_s, bc_s, J=Gain_s, \
			solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1e-3, "maximum_iterations":10000}},
			form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs"})			

		# Project the solution on reference configuration of fluid domain, in order to give the structure displacement as DirichletBC for mesh displacement
		u_m_bound = project(u_s, V_f_ref, solver_type="mumps", \
			form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2})


		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		#%%%%%%%%%%%%%%%%%%%%%% Solve mesh displacement %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		 
		if processID == 0: print '     Solving mesh displacement...'

		# Define voundary conditions for mesh displacement
		bc_m = []

#		bc_m.append( DirichletBC(V_f_ref, Constant((0, 0)), facets_f_ref, 1) )
#		bc_m.append( DirichletBC(V_f_ref, Constant((0, 0)), facets_f_ref, 2) )
#		bc_m.append( DirichletBC(V_f_ref, Constant((0, 0)), facets_f_ref, 3) )

		bc_m.append( DirichletBC(V_f_ref.sub(0), 0.0, facets_f_ref, 1) )
		bc_m.append( DirichletBC(V_f_ref.sub(0), 0.0, facets_f_ref, 2) )
		bc_m.append( DirichletBC(V_f_ref.sub(1), 0.0, facets_f_ref, 3) )
		bc_m.append( DirichletBC(V_f_ref, Constant((0, 0)), facets_f_ref, 4) )
		bc_m.append( DirichletBC(V_f_ref, u_m_bound, facets_f_ref, 5) )
		

		# Get the pressure and velocity value at t-dt
		mesh_f_old = Mesh(mesh_f_act)
		MixedSpace_f_old = FunctionSpace(mesh_f_old, mixed_element)
		u0_f_old = Function(MixedSpace_f_old)
		u0_f_old.assign(u0_f)

		# Get new mesh displacement
		try: solve(a_m==L_m, u_m, bcs=bc_m)
		except: 
			u_m.assign(uk_m)
			if processID == 0: print '     CAUTION: I had to do an exception...'

		# Get mesh velocity	
		v_m_ref = project((u_m-u0_m)/dt, V_f_ref, solver_type="mumps", form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs"})
		
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		#%%%%%%%%%%%% Build new mesh for the actual configuration %%%%%%%%%%%%%%%%%%%%%%%%%%
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		if processID == 0: print '     Building new mesh...'

		mesh_f_act = Mesh(mesh_f_ref)

		facets_f_act = MeshFunction('size_t', mesh_f_act, 1)
		cells_f_act = MeshFunction('size_t', mesh_f_act, 2)

		da_act = Measure('ds', domain=mesh_f_act, subdomain_data=facets_f_act, metadata={'quadrature_degree': 2})
		dv_act = Measure('dx', domain=mesh_f_act, subdomain_data=cells_f_act, metadata={'quadrature_degree': 2})
		n_act = FacetNormal(mesh_f_act)

		facets_f_act.set_all(0)
		inflow.mark(facets_f_act,1)
		outflow.mark(facets_f_act,2)
		walls.mark(facets_f_act,3)
		cylinder.mark(facets_f_act,4)
		beam.mark(facets_f_act,5)

		MixedSpace_f = FunctionSpace(mesh_f_act, mixed_element)
		V_f_act = FunctionSpace(mesh_f_act, vector)
		T_f_act = FunctionSpace(mesh_f_act, tensor)

		# Define functions for fluid on the new mesh
		du_f = TrialFunction(MixedSpace_f)
		del_u_f = TestFunction(MixedSpace_f)
		u_f = Function(MixedSpace_f)
		u0_f = Function(MixedSpace_f)
		uk_f = Function(MixedSpace_f)

		p, v = split(u_f)
		dp_f, dv_f = split(du_f)
		del_p, del_v = split(del_u_f)

		# Deform the new mesh
		for x in mesh_f_act.coordinates(): x[:] += u_m(x)[:]
		mesh_f_act.bounding_box_tree().build(mesh_f_act)

		# Pass the fluid velocity of previous time step and mesh velocity values to the current configurations
		v_m_act = Function(V_f_act)
		v_m_act.vector()[:] = v_m_ref.vector().get_local()

		u0_f.assign( project(u0_f_old , MixedSpace_f , solver_type="mumps", \
		    form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}) )

		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		#%%%%%%%%%%%%%%%%%%%%%% Solve fluid dynamic %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		if processID == 0: print '     Solving fluid dynamic...'

		#%% Define boundary conditions for the fluid
		bc_f = []
		bc_f.append( DirichletBC(MixedSpace_f.sub(1), inflow_profile, facets_f_act, 1) )
		bc_f.append( DirichletBC(MixedSpace_f.sub(1), Constant((0, 0)), facets_f_act, 3) )
		bc_f.append( DirichletBC(MixedSpace_f.sub(1), Constant((0, 0)), facets_f_act, 4) )
		bc_f.append( DirichletBC(MixedSpace_f.sub(1), Constant((0, 0)), facets_f_act, 5) )
		bc_f.append( DirichletBC(MixedSpace_f.sub(0), pref, facets_f_act, 2) )
		
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Picard Iteration %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		L2_error = 1.0  # error measure ||u-u_k||
		tol = 1e-5     	# tolerance
		it = 0          # iteration counter
		maxiter = 1000	# iteration limit		
		
		uk_f.assign(u0_f)
		p0, v0 = split(u0_f)
		
		while L2_error > tol and it < maxiter:
			
			it += 1				
			p_k, v_k = split(uk_f)
			d = as_tensor( 1./2.*(dv_f[i].dx(j)+dv_f[j].dx(i)) , [i,j] )
			tau = as_tensor( la*d[k,k]*delta[i,j] + 2.*mu*d[i,j] , [i,j] )
			
			F_1 = dv_f[i].dx(i)*del_p*dv_act
			F_2 = ( ( rho*(dv_f-v0)[j]/dt  + rho*(v_k[i]*dv_f[j]).dx(i) + dp_f.dx(j) ) *del_v[j]  + tau[i,j]*del_v[j].dx(i)  )*dv_act
			F_3 = ( (dv_f-v0)[j] + dt*(v_k[i]*dv_f[j]).dx(i) - dt/rho*(-dp_f*delta[i,j] + tau[i,j]).dx(i) ) *del_p.dx(j)*dv_act

			Form_f = F_1 + F_2 + F_3
			a_f = lhs(Form_f)
			L_f = rhs(Form_f)

			solve(a_f == L_f, u_f, bcs=bc_f, solver_parameters = {"linear_solver":"mumps"}) 
			L2_error = assemble(((u_f-uk_f)**2)*dx)
			if processID == 0: print '          it=%d: L2-error=%g' % (it, L2_error)
			uk_f.assign(u_f)
			
			if it == maxiter and processID == 0: print 'Solver did not converge!'

		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Picard Iteration - End %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		p_ = u_f.split(deepcopy=True)[0]
		v_ = u_f.split(deepcopy=True)[1]

		# Compute the L2-Norm of the structure	
		L2_abs = assemble((((u_s-uk_s)**2.0)**0.5)*dx)
		uk_vec = uk_s.vector()
		us_vec = u_s.vector()
		# if processID == 0: print uk_vec, us_vec
		
		# Compute fluid stress on current configuration
		d_ = as_tensor( 1./2.*( v_[i].dx(j)+v_[j].dx(i) ) , [i,j] )
		tau_ = as_tensor( la*d_[k,k]*delta[i,j] + 2.*mu*d_[i,j] , [i,j] )
		sigma_f_act = project( -p_*delta+tau_ , T_f_act, solver_type="mumps", \
		 	form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs"} )

		# Pass the values to the reference configurations
		sigma_f_ref = Function(T_f_ref)
		sigma_f_ref.vector()[:] = sigma_f_act.vector().get_local()

		# Project the values on the structure domain in order to get the traction vector
		sigma_s = project( sigma_f_ref , T_s_space, solver_type="mumps", \
		 	form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs"} )

		P_f_ref = as_tensor( J_s * inv(F_s)[k,j] * sigma_s[j,i], (k,i) )
		t_s_hat = as_tensor( N[k] * P_f_ref[k,i], (i,) )

		if processID == 0: 
			print '     Absolute L2-norm for convergence:', L2_abs 
			# print '     Relative L2-norm for convergence:', L2_rel 
			print '     Tip deflection (in m):', u_s(Point(0.6,0.2))

		# Assign values for next FOR-Iteration
		uk_m.assign(u_m)
		uk_s.assign(u_s)


	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#%%%%%%%%%%%%%%%%%%%%%%%%%%% End of global FOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	if processID == 0: print '\n'

	u00_s.assign(u0_s)
	u0_s.assign(u_s)
	u0_m.assign(u_m)
	u0_f.assign(u_f)

	# Solve results as .pvd 
	if round(t*1e7)%100000 == 0:
		p_save = u_f.split(deepcopy=True)[0]
		v_save = u_f.split(deepcopy=True)[1]

		p_save.rename("p", "tmp")
		v_save.rename("v", "tmp")
		u_s.rename("u", "tmp")

		File_u_s << (u_s, round(t*1000)/1000.0)
		File_p_f << (p_save, round(t*1000)/1000.0)
		File_v_f << (v_save, round(t*1000)/1000.0)
		
		if processID == 0: print 'Results saved'

	time_list.append(t)
	Xdisplacement_list.append( u_s(Point(0.6,0.2))[0] )
	Ydisplacement_list.append( u_s(Point(0.6,0.2))[1] )

if processID == 0: print 'Simulation took: ', datetime.datetime.now() - t1, 's'



