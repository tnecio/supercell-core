import supercell_core as sc

# Defining unit cell of graphene
graphene = sc.lattice()
graphene.set_vectors([2.13, -1.23], [2.13, 1.23])
# "C" (carbon) atoms in the unit cell in either
# angstrom or direct coordinates
graphene.add_atom("C", (0, 0, 0))\
.add_atom("C", (2/3, 2/3, 0),
unit=sc.Unit.Crystal)

# Combining graphene layers
h = sc.heterostructure().set_substrate(graphene)\
.add_layer(graphene)

import numpy as np
# Optimise with theta as a free parameter
res = h.opt(max_el=12, thetas=\
[np.arange(0, 7*sc.DEGREE, 0.1*sc.DEGREE)])

# Save supercell to VASP POSCAR
res.superlattice().save_POSCAR("POSCAR")
