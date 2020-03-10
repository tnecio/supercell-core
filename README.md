# supercell-core
Package for investigation of mulitlayer 2D heterostructures lattices

Author: Tomasz Necio, University of Warsaw

Copyright (C) 2019-2020 University of Warsaw

# Installation:
`pip3 install supercell-core --user`

# Documentation

https://readthedocs.org/projects/supercell-core/

# Usage example

```
# Load supercell_core
import supercell_core as sc

# Read NiPS3 data from a POSCAR file into `nips3` Lattice object
# We need to provide names of chemical elements because they don't have to
# appear in POSCAR, but supercell-core needs them
# This will only work if you actually have a file named "POSCAR_nips3",
# you can copy it from /supercell_core/resources/vasp/NiPS3 directory
nips3 = sc.read_POSCAR("POSCAR_nips3", ["Ni", "P", "S"])

# Let's create `graphene` Lattice object by hand
graphene = sc.lattice()
graphene.set_vectors([2.133, -1.23], [2.133, 1.23])
# atom "C" at position (0, 0, 0)
graphene.add_atom("C", (0, 0, 0))

# Methods can be chained
nips3_with_a_rouge_atom = sc.lattice().set_vectors(*nips3.vectors()) \
                            .add_atom("U", (0, 0, 1), spin=(0, 0, 1))

# Define heterostructure
h = sc.heterostructure().set_substrate(graphene).add_layer(nips3)

# Calculate strain tensor for particular supercell configuration
# (see help(sc.heterostructure()) for details)
M = [[1, -9], [8, 1]]
theta = 21.9 * sc.DEGREE
res = h.calc(M=M, thetas=[theta])
print(res.strain_tensors()) # Use help(res) to see all public methods of `Result`

# Optimise strain to find best supercell, with max repetition along any
# axis <= 12 substrate unit cells, and relative angle only from 0 to 7 degrees
# with resolution 0.1 deg
res = h.opt(max_el=12, thetas=[np.arange(0, 7*sc.DEGREE, 0.1*sc.DEGREE)])
superlattice = res.superlattice() # Lattice object
superlattice.save_POSCAR("POSCAR_sc") # save to file

fig, ax = superlattice.draw()
fig.show() # Show unit cell using matplotlib library

# To learn more use built-in documentation, e.g.:
help(nips3) # shows help on Lattice object when Python is in interactive mode (use 'q' to quit)
print(help(nips3)) # prints to stdout
```
