from fenics import *
import sys
name=str(sys.argv[1])
mesh = Mesh(name+'.xml')
cells = MeshFunction('size_t',mesh,name+'_physical_region.xml')
facets = MeshFunction('size_t',mesh,name+'_facet_region.xml')
hdf = HDF5File(mesh.mpi_comm(),name+'.h5','w')
hdf.write(mesh, '/mesh')
hdf.write(cells, '/cells')
hdf.write(facets, '/facets')

