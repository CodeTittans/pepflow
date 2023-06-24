from Bio import PDB
from openmm.app.pdbfile import PDBFile
from openmm.app.topology import Topology
from openmm.app.modeller import Modeller
from openmm.app.element import Element
from openmm.app.forcefield import ForceField
from pepflow.utils import constants
import numpy as np
import io
import os
import pdbfixer


class ValidAtoms(PDB.Select):

    def __init__(self, chains, residues):
        super().__init__()

        self.chains = chains

        self.residues = residues

    def accept_residue(self, residue):
        return 1 if residue.id[0] == " " and residue.id in self.residues else 0

    def accept_chain(self, chain):
        return 0 if chain.id not in self.chains else 1


    
class DataPipeline():

    def __init__(self, fragment_path="fragment_chains/", ignore_hydrogens=False):

        self.fragment_path = fragment_path

        self.forcefield = ForceField('amber99sbildn.xml', 'amber99_obc.xml')
        
        self.ignore_hydrogens = ignore_hydrogens

    def extract_fragment(self, pdb_code, fragment_position, fragment_length):

        # read full PDB chain

        parser = PDB.PDBParser(QUIET=True)

        structure = parser.get_structure(
            pdb_code, os.path.join(self.fragment_path, pdb_code + ".pdb"))

        chain_id = pdb_code.split("_")[1]

        model = list(structure.get_models())[0]

        chain = model[chain_id]

        # filter out the fragment residues

        start_index = fragment_position

        end_index = start_index + fragment_length

        residues = list(chain.get_residues())[start_index:end_index]

        residues = [residue.id for residue in residues]

        writer = PDB.PDBIO()

        select = ValidAtoms(chains=[chain_id], residues=residues)

        writer.set_structure(structure)

        writer.save(os.path.join(self.fragment_path, pdb_code + "_fragment_" +
                    str(fragment_position)+".pdb"), select=select)

        pdbstring = open(os.path.join(self.fragment_path, pdb_code + "_fragment_" +
                                      str(fragment_position)+".pdb"), 'r').read()

        pdbfile = io.StringIO(pdbstring)

        os.remove(os.path.join(self.fragment_path, pdb_code + "_fragment_" +
                               str(fragment_position)+".pdb"))

        # add missing atoms to the fragment

        fixer = pdbfixer.PDBFixer(
            pdbfile=pdbfile)

        fixer.findMissingResidues()

        fixer.findMissingAtoms()

        fixer.addMissingAtoms()

        PDBFile.writeFile(fixer.topology, fixer.positions,
                          open(os.path.join(self.fragment_path, pdb_code + "_fragment_" +
                                            str(fragment_position)+".pdb"), 'w'),
                          keepIds=True)

        # parse corrected structure

        structure = parser.get_structure(pdb_code, os.path.join(self.fragment_path, pdb_code + "_fragment_" +
                                                                str(fragment_position)+".pdb"))

        os.remove(os.path.join(self.fragment_path, pdb_code + "_fragment_" +
                               str(fragment_position)+".pdb"))

        model = list(structure.get_models())[0]

        chain = model[chain_id]

        # add missing atoms using forcefield

        atom_pos = {}

        for index, residue in enumerate(chain.get_residues()):

            for atom in residue.get_atoms():

                atom_pos[str(index) + " " + atom.get_name()] = atom.get_coord()

        topology = Topology()

        chain_struct = topology.addChain(id=chain_id)

        all_coords = []

        all_atom_dicts = {}

        for index, residue in enumerate(chain.get_residues()):

            name = residue.get_resname()
            
            # return None if theres an unknown residue
            
            if name == 'UNK':
                return None
            
            residue = topology.addResidue(name, chain_struct, id=index)

            atom_actual_dict = {}

            for atom in constants.ATOM_DICT[name]:

                if (atom == "OXT" and index != fragment_length-1) or\
                        (atom.startswith("H") and str(index) + " " + atom not in atom_pos):
                    continue

                atom_actual = topology.addAtom(
                    atom, Element.getBySymbol(atom[0]), residue)

                atom_actual_dict[atom] = atom_actual

                all_coords.append(atom_pos[str(index) + " " + atom])

            all_atom_dicts[index] = atom_actual_dict

            for pair in constants.COVALENT_BOND_DICT[name]:

                if (pair[0] == "OXT" and index != fragment_length-1) or \
                    (pair[1] == "OXT" and index != fragment_length-1) or\
                    (pair[0].startswith("H") and str(index) + " " + pair[0] not in atom_pos) or\
                        (pair[1].startswith("H") and str(index) + " " + pair[1] not in atom_pos):
                    continue

                topology.addBond(atom_actual_dict[pair[0]],
                                 atom_actual_dict[pair[1]])

            if index != 0:

                topology.addBond(
                    atom_actual_dict["N"], all_atom_dicts[index-1]["C"])

        all_coords = np.array(all_coords)/10

        modeller = Modeller(topology, all_coords)

        modeller.addHydrogens(self.forcefield)

        PDBFile.writeFile(modeller.topology, modeller.getPositions(),
                          file=open(os.path.join(self.fragment_path, pdb_code + "_fragment_" +
                                                 str(fragment_position)+".pdb"), 'w'))

        structure = parser.get_structure(pdb_code, os.path.join(self.fragment_path, pdb_code + "_fragment_" +
                                                                str(fragment_position)+".pdb"))

        os.remove(os.path.join(self.fragment_path, pdb_code + "_fragment_" +
                               str(fragment_position)+".pdb"))

        model = list(structure.get_models())[0]

        chain = model["A"]

        return chain

    def compute_features(self, pdb_code, fragment_position, fragment_length):

        fragment_structure = self.extract_fragment(
            pdb_code, fragment_position, fragment_length)
        
        if fragment_structure == None:
            return None

        coordinates = []

        amino_acid_pos = []

        atoms = []

        atom_names = []

        fragment_seq = []

        peptide_length = len(list(fragment_structure.get_residues()))
        
        for index, res in enumerate(fragment_structure.get_residues()):

            fragment_seq.append(constants.residue_order[PDB.Polypeptide.three_to_one(
                res.get_resname())])

            for atom in res.get_atoms():



                if (PDB.Polypeptide.three_to_one(res.get_resname()),
                    atom.get_name().strip())\
                        in constants.ambiguous_mapping:
                    atoms.append(constants.atom_order
                                 [constants.ambiguous_mapping
                                  [(PDB.Polypeptide.three_to_one(res.get_resname()),
                                    atom.get_name().strip())]])
                elif index == peptide_length-1 and atom.get_name().strip() == "OXT":
                    atoms.append(constants.atom_order
                                 ["O"])
                elif not atom.get_name().strip().startswith("H"):
                    atoms.append(
                        constants.atom_order[atom.get_name().strip()])
                else:
                    if self.ignore_hydrogens:
                        continue
                    
                    atoms.append(constants.atom_order["H"])
                
                coordinates.append(atom.get_coord())

                amino_acid_pos.append(index+1)
                
                atom_names.append(atom.get_name().strip())

        coordinates = np.array(coordinates)

        amino_acid_pos = np.array(amino_acid_pos)

        atoms = np.array(atoms)

        fragment_seq = np.array(fragment_seq)

        # generate matrix of covalent bonds

        bond_matrix = np.zeros((len(atoms), len(atoms)))

        for i in range(len(atoms)):
            for j in range(len(atoms)):
                
                if amino_acid_pos[i] == amino_acid_pos[j]:
                    
                    one_letter = constants.RESIDUE_LIST[fragment_seq[amino_acid_pos[i]-1]]
                    three_letter = PDB.Polypeptide.one_to_three(
                        one_letter)
                    
                    
                    if (atom_names[i], atom_names[j]) in constants.COVALENT_BOND_DICT[three_letter.upper()]\
                            or (atom_names[j], atom_names[i]) in constants.COVALENT_BOND_DICT[three_letter.upper()]:

                        bond_matrix[i, j] = 1

                elif amino_acid_pos[i] == amino_acid_pos[j] - 1:
                    if atom_names[j] == "N" and atom_names[i] == "C":

                        bond_matrix[i, j] = 1

                elif amino_acid_pos[i] == amino_acid_pos[j] + 1:
                    if atom_names[j] == "C" and atom_names[i] == "N":

                        bond_matrix[i, j] = 1

        return fragment_seq, coordinates, amino_acid_pos,\
            atoms, atom_names, bond_matrix


def num_atoms(model):
    
    
    chain = list(model.get_chains())[0]
    
    atoms = np.sum([len(list(i.get_atoms())) for i in chain.get_residues()])
    
    return atoms

class DataPipelinePED():

    def __init__(self, ped_path="full_ensembles/"):

        self.ped_path = ped_path



    def compute_features(self, peptide):

        parser = PDB.PDBParser(QUIET=True)
        
        trajectory = parser.get_structure(peptide, os.path.join(self.ped_path, peptide + ".pdb"))
        
        
        
        atom_nums = [num_atoms(model) for model in trajectory.get_models()]
        
        all_models = np.argwhere(atom_nums == np.max(atom_nums)).flatten()
                         
        if len(all_models) > 20:
            all_models = all_models[::len(all_models)//20]
            
        
        coordinates_all = []

            
        for model in all_models:
            
            structure = trajectory.get_list()[model].get_list()[0]
            
            
            coordinates = []
    
            amino_acid_pos = []
    
            atoms = []
    
            atom_names = []
    
            peptide_seq = []
    
            peptide_length = len(list(structure.get_residues()))
            
            for index, res in enumerate(structure.get_residues()):
    
                peptide_seq.append(constants.residue_order[PDB.Polypeptide.three_to_one(
                    res.get_resname())])
    
                for atom in res.get_atoms():
    
                    coordinates.append(atom.get_coord())
    
                    amino_acid_pos.append(index+1)
    
                    if (PDB.Polypeptide.three_to_one(res.get_resname()),
                        atom.get_name().strip())\
                            in constants.ambiguous_mapping:
                        atoms.append(constants.atom_order
                                     [constants.ambiguous_mapping
                                      [(PDB.Polypeptide.three_to_one(res.get_resname()),
                                        atom.get_name().strip())]])
                    elif index == peptide_length-1 and atom.get_name().strip() == "OXT":
                        atoms.append(constants.atom_order
                                     ["O"])
                    elif not atom.get_name().strip().startswith("H"):
                        atoms.append(
                            constants.atom_order[atom.get_name().strip()])
                    else:
                        atoms.append(constants.atom_order["H"])
    
                    atom_names.append(atom.get_name().strip())
    
            coordinates = np.array(coordinates)
    
            amino_acid_pos = np.array(amino_acid_pos)
    
            atoms = np.array(atoms)
    
            peptide_seq = np.array(peptide_seq)
    
            # generate matrix of covalent bonds
    
            bond_matrix = np.zeros((len(atoms), len(atoms)))
    
            for i in range(len(atoms)):
                for j in range(len(atoms)):
                    
                    if amino_acid_pos[i] == amino_acid_pos[j]:
                        
                        one_letter = constants.RESIDUE_LIST[peptide_seq[amino_acid_pos[i]-1]]
                        three_letter = PDB.Polypeptide.one_to_three(
                            one_letter)
                        
                        
                        if (atom_names[i], atom_names[j]) in constants.COVALENT_BOND_DICT[three_letter.upper()]\
                                or (atom_names[j], atom_names[i]) in constants.COVALENT_BOND_DICT[three_letter.upper()]:
    
                            bond_matrix[i, j] = 1
    
                    elif amino_acid_pos[i] == amino_acid_pos[j] - 1:
                        if atom_names[j] == "N" and atom_names[i] == "C":
    
                            bond_matrix[i, j] = 1
    
                    elif amino_acid_pos[i] == amino_acid_pos[j] + 1:
                        if atom_names[j] == "C" and atom_names[i] == "N":
    
                            bond_matrix[i, j] = 1
                            
            coordinates_all.append(coordinates)

        
        coordinates_all = np.array(coordinates_all)

        return peptide_seq, coordinates_all, amino_acid_pos,\
            atoms, atom_names, bond_matrix
            
            
class DataPipelineMD():

    def __init__(self, md_path="md_pdbs/", sample_mds=None, ignore_hydrogens=False):

        self.md_path = md_path

        self.sample_mds = sample_mds
        
        self.ignore_hydrogens = ignore_hydrogens

    def compute_features(self, peptide):

        parser = PDB.PDBParser(QUIET=True)
        
        trajectory = parser.get_structure(peptide, os.path.join(self.md_path, peptide + ".pdb"))
        
        num_models = len(list(trajectory.get_models()))
        
        if self.sample_mds != None:
            
            all_models = np.random.choice(list(range(0, num_models)), self.sample_mds)
            
        else:
            
            all_models = list(range(0, num_models))
            
        coordinates_all = []

            
        for model in all_models:
            
            structure = trajectory.get_list()[model].get_list()[0]
            
            
            coordinates = []
    
            amino_acid_pos = []
    
            atoms = []
    
            atom_names = []
    
            peptide_seq = []
    
            peptide_length = len(list(structure.get_residues()))
            
            for index, res in enumerate(structure.get_residues()):
    
                peptide_seq.append(constants.residue_order[PDB.Polypeptide.three_to_one(
                    res.get_resname())])
    
                for atom in res.get_atoms():
                    
                    if (PDB.Polypeptide.three_to_one(res.get_resname()),
                        atom.get_name().strip())\
                            in constants.ambiguous_mapping:
                        atoms.append(constants.atom_order
                                     [constants.ambiguous_mapping
                                      [(PDB.Polypeptide.three_to_one(res.get_resname()),
                                        atom.get_name().strip())]])
                    elif index == peptide_length-1 and atom.get_name().strip() == "OXT":
                        atoms.append(constants.atom_order
                                     ["O"])
                    elif not atom.get_name().strip().startswith("H") and atom.get_name().strip() in constants.atom_order:
                        atoms.append(
                            constants.atom_order[atom.get_name().strip()])
                    elif atom.get_name().strip().startswith("H") and atom.get_name().strip() in constants.ATOM_DICT[res.get_resname()]:
                        
                            
                        if self.ignore_hydrogens:
                            continue
                        
                        atoms.append(constants.atom_order["H"])
                    
                    if len(atoms) > len(atom_names):
                        atom_names.append(atom.get_name().strip())

                        coordinates.append(atom.get_coord())

                        amino_acid_pos.append(index+1)
    
            coordinates = np.array(coordinates)
    
            amino_acid_pos = np.array(amino_acid_pos)
    
            atoms = np.array(atoms)
    
            peptide_seq = np.array(peptide_seq)
    
            # generate matrix of covalent bonds
    
            bond_matrix = np.zeros((len(atoms), len(atoms)))
    
            for i in range(len(atoms)):
                for j in range(len(atoms)):
                    
                    if amino_acid_pos[i] == amino_acid_pos[j]:
                        
                        one_letter = constants.RESIDUE_LIST[peptide_seq[amino_acid_pos[i]-1]]
                        three_letter = PDB.Polypeptide.one_to_three(
                            one_letter)
                        
                        
                        if (atom_names[i], atom_names[j]) in constants.COVALENT_BOND_DICT[three_letter.upper()]\
                                or (atom_names[j], atom_names[i]) in constants.COVALENT_BOND_DICT[three_letter.upper()]:
    
                            bond_matrix[i, j] = 1
    
                    elif amino_acid_pos[i] == amino_acid_pos[j] - 1:
                        if atom_names[j] == "N" and atom_names[i] == "C":
    
                            bond_matrix[i, j] = 1
    
                    elif amino_acid_pos[i] == amino_acid_pos[j] + 1:
                        if atom_names[j] == "C" and atom_names[i] == "N":
    
                            bond_matrix[i, j] = 1
                            
            coordinates_all.append(coordinates)

        
        coordinates_all = np.array(coordinates_all)

        return peptide_seq, coordinates_all, amino_acid_pos,\
            atoms, atom_names, bond_matrix
