import unittest as ut

import supercell_core as sc

class TestSupercell(ut.TestCase):
    """
    Test of the supercell_core interface as a whole
    """
    # TODO: fix this, write how-tos for documentation

    def test_primary_use_case(self):
        """
        Test primary use case: specifying supercell layers by hand, and finding
        optimal parameters of making them into a heterostructure
        """

        # Creating a model of the system under study

        substrate = sc.read_POSCAR("../resources/vasp/graphene/POSCAR")
        lay1 = sc.read_POSCAR("../resources/vasp/NiPS3/POSCAR") # TODO: read_pwx
        lay2 = sc.lattice()
        lay2.set_vectors((4, 0), (0, 5))
        lay2.add_atom("He", (0, 0, 0.5)) # TODO: add spin information
        lay2.add_atom(sc.element_symbol(1), (2, 0), unit=sc.Unit.Angstrom)
        lay2.add_atoms([(sc.element_symbol(1), (0.2, 0.2)),
                        (sc.element_symbol(1), (0.5, 0.5))], unit=sc.Unit.Crystal)

        # TODO: lattice separation, lattice-specific thetas
        # TODO: z-size, verbosity, output
        system = sc.heterostructure()
        system.set_substrate(substrate)
        system.add_layers([(lay2, 45.0*sc.DEGREE)])
        system.add_layer(lay1, pos=1)
        # Different kinds of calculations on the system

        opt = system.opt(sc.Quantity.MaxStrainElement,
                         max_el=4,
                         max_supercell_size=10000)

        res = system.calc(sc.Quantity.Strain,
                          M=[[2, 3], [2, -3]],
                          thetas=[90*sc.DEGREE, None])

        print(opt.meta.supercell_in)
        print(opt.meta.runtime, opt.meta.supercell_version)
        print(opt.qty)
        for layer in opt.layers():
            print(layer.no, layer.strain_tensor, layer.theta,
                  layer.lattice, layer.stretched_vectors,
                  layer.supercell_base_matrix,
                  layer.strain_measure(sc.Quantity.MaxStrainElement),
                  layer.strain_measure(sc.Quantity.Strain))

        print(opt.substrate.supercell_base_matrix)
        print(opt.supercell.vectors())
        print(opt.supercell.atoms())

        opt.save_txt("../resources/out/test.txt")
        opt.supercell.save_POSCAR("../resources/out/POSCAR")