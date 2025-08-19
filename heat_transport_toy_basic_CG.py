# heat_transport_toy_basic_CG.py
# this is for testing out boundary conditions e.g. those used in NESO-tokamak
# this is set up to do a grazing incidence (2deg) on the top boundary

from firedrake import *

mesh=Mesh("bi_unit_square.msh")

V = FunctionSpace(mesh, "CG", 6)  # use e.g. 6 with the bi_unit_square, which is a crude triangle mesh

u = TrialFunction(V)
v = TestFunction(V)

x, y = SpatialCoordinate(mesh)

# source function
# pick something without sharp edges (I used "Frankenstein's Monster"-type curve)
offset = 0.4
bf = 0.0+conditional(And(ge(y, -0.25+offset), le(y, 0.25+offset)), exp(16-1/(0.25*0.25-(y-offset)**2)), 0)

bf_func = Function(V)
bf_func.interpolate(0.0+1.0*x)

bcLL = DirichletBC(V, bf, 16)  # source on LHS edge

theta = 2.0*pi/180 # nonzero angle may cause artifacts for some BC choices
bhat=as_vector([cos(theta),sin(theta)])

k_par = 1.0
k_per = 1.0e-6

flux = k_par * bhat * dot(bhat, grad(u)) + k_per * (grad(u) - bhat * dot(bhat, grad(u)))

#a = inner(flux, grad(v))*dx

# version including oblique outflow BC as in NESO-Tokamak presentation
# this gives weird artifacts
#a = inner(flux, grad(v))*dx - (k_par*dot(bhat,grad(u))+u)*v*(ds(15)+ds(17)+ds(18)+ds(19)+ds(20))

# here is a modified version of the above from examining "The influence of boundary and edge-plasma modeling in computations of axisymmetric vertical displacement" by Bunkers and Sovinec (Appendix A in paper)
# this seems to work - no artifacts - and it is not identical to the straight Robin condition
norm = FacetNormal(mesh)
a = inner(flux, grad(v))*dx + (k_par*dot(norm,flux)+u*dot(norm,bhat))*v*(ds(15)+ds(17)+ds(18)+ds(19)+ds(20))

# a more standard Robin BC
# this doesn't produce weird artifacts but I don't think it's quite the correct physics
#norm = FacetNormal(mesh)
#a = inner(flux, grad(v))*dx + (dot(norm,flux)+u)*v*(ds(15)+ds(17)+ds(18)+ds(19)+ds(20))


f = Function(V)
f.interpolate(0.0*x)
L = inner(f,v)*dx

T = Function(V)

solve( a==L, T, bcs=[bcLL])

File("heat_transport_toy_basic_CG.pvd").write(T)  
