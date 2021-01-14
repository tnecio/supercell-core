import supercell_core as sc
import matplotlib.pyplot as plt

# Read graphene and NiPS3 definition from POSCAR
graphene = sc.read_POSCAR("supercell_core/resources/vasp/POSCAR_Gr")
nips3 = sc.read_POSCAR("supercell_core/resources/vasp/POSCAR_NiPS3", atomic_species=['Ni', 'P', 'S'])
h = sc.heterostructure().set_substrate(graphene)\
.add_layer(nips3)

res = h.opt(max_el=8, thetas=[
np.arange(0, 30 * sc.DEGREE, 0.25*sc.DEGREE)])

# Draw the resulting supercell
res.superlattice().draw()
plt.title("""Best supercell found for $\\theta$ = 
{} max strain = {:.4g}"""
.format(res.thetas()[0], res.max_strain()))
