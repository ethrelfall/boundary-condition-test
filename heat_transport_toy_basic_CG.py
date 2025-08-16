# heat_transport_toy_basic_CG.py
# this is for testing out boundary conditions e.g. those used in NESO-tokamak

from firedrake import *

mesh=Mesh("bi_unit_square.msh")

V = FunctionSpace(mesh, "CG", 6)  # use e.g. 6 with the bi_unit_square, which is a crude triangle mesh

u = TrialFunction(V)
v = TestFunction(V)

x, y = SpatialCoordinate(mesh)

# source function
# pick something without sharp edges (I used "Frankenstein's Monster"-type curve)
bf = 0.0+conditional(And(ge(y, -0.25), le(y, 0.25)), exp(16-1/(0.25*0.25-y**2)), 0)

bf_func = Function(V)
bf_func.interpolate(0.0+1.0*x)

bcLL = DirichletBC(V, bf, 16)  # source on LHS edge

theta = 20.0*pi/180 # nonzero angle may cause artifacts for some BC choices
bhat=as_vector([cos(theta),sin(theta)])

k_par = 1.0
k_per = 1.0e-6

flux = k_par * bhat * dot(bhat, grad(u)) + k_per * (grad(u) - bhat * dot(bhat, grad(u)))

#a = inner(flux, grad(v))*dx

# version including oblique outflow BC as in NESO-Tokamak presentation
# this gives weird artifacts
a = inner(flux, grad(v))*dx - (k_par*dot(bhat,grad(u))+u)*v*(ds(15)+ds(17)+ds(18)+ds(19)+ds(20))

# a more standard Robin BC
# this doesn't produce weird artifacts
#norm = FacetNormal(mesh)
#a = inner(flux, grad(v))*dx + (dot(norm,flux)+u)*v*(ds(15)+ds(17)+ds(18)+ds(19)+ds(20))


f = Function(V)
f.interpolate(0.0*x)
L = inner(f,v)*dx

T = Function(V)

solve( a==L, T, bcs=[bcLL])

File("heat_transport_toy_basic_CG.pvd").write(T)  
