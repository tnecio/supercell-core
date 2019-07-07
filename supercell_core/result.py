from .lattice import Lattice


class MetaInformation:
    """
    A class containing information on data not related to the physics
    of the problem, but rather execution of the program itself

    Attributes
    ----------
    supercell_in : str
        contains an input string that if read from a file by 'read_supercell_in'
        function, would produce this output
    runtime : float
        time it took for the program to finish execution (in seconds)
    supercell_version : str
        version of the supercell package used
    """

    supercell_in: str = ""
    runtime: float = 0.0
    supercell_version: str = "0.0.1"

    def __init__(self):
        pass


class LatticeInformation:
    """
    A class containing calculation results regarding specific lattice

    Attributes
    ----------
    supercell_base_matrix : ((int, int), (int, int))
        This matrix represents lattice vectors of the supercell expressed in
        base of stretched lattice vectors of a given lattice.

        Suppose supercell_base_matrix = ((M_11, M12), (M_21, M_22)),
        c_1, c_2 – supercell lattice vectors,
        b_1, b_2 – vectors of lattice described by this LatticeInformation,
            slighlty different to original lattice vectors because of strain
            forces that come from interaction when this crystal is put between
            other crystals, and rotated around z-axis by angle theta
        Then: c_1 = M_11 * b_1 + M_12 * b_2, c_2 = M_21 * b_1 + M_22 * b_2
    lattice : Lattice
        Lattice object containing description of the original lattice
    """
    pass


class LayerInformation(LatticeInformation):
    """
    A class containing calculation results regarding specific lattice
    that is above the substrate

    Attributes
    ----------
    no : int
        Layer number (first layer above substrate is no. 1, the one laid on top
        of it is no. 2 etc.)
    strain_tensor : ((float, float), (float, float))
        Strain tensor of the layer, when embedded in the system under study
    theta : float
        Value of the optimal, or specified, angle between this layer and
        the substrate (in radians)
    stretched_vectors : (Vec3D, Vec3D, Vec3D)
        Lattice vectors when under strain because of embedding in the system
        Note: Vec3D = (float, float, float)

    Methods
    -------
    strain_measure(Quantity) : float
        Value of the strain measure quantity specified in the argument
    """
    pass


class Result:
    """
    A class representing results of the calculation

    Attributes
    ----------
    meta : MetaInformation
        Contains information on the calculation itself (see: MetaInformation)
    substrate : LatticeInformation
        Object containing results regarding the substrate
    layers : List[LayerInformation]
        List of LayerInformation objects containing result regarding specific
        lattice layer above substrate. This list is ordered, and starts with
        the layer closest to the substrate
    supercell : Lattice
        Lattice object representing lattice of a heterostructure under study
    heterostructure_object : Heterostructure
        Heterostructure object whose method generated this results container
    qty_desc : Quantity
        Enumerated value representing quantity under study (see: Quantity)
    qty : float
        Aggregated value of quantity `qty_desc`. This value should be minimal
        under constraints given to the `opt` function that generated this Result
        Usually `qty` is just sum of the values of this quantity for all layers
        in the system.

    Methods
    -------
    save_txt(filename: str) : None
        Saves the result in human-readable TXT file
        Note: to save resulting heterostructure lattice in a format readable
        by other programs such as VASP or QE use methods of supercell attribute
    """

    def save_txt(self, filename: str) -> None:
        """
        Saves result in a human-readable TXT file

        Parameters
        ----------
        filename : str

        Returns
        -------
        None
        """
        pass
