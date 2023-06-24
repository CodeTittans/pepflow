from Bio.PDB import StructureBuilder, Chain, Polypeptide, PDBIO, Residue, Atom
from pepflow.utils.constants import COVALENT_BOND_DICT, residue_mapping
from openmm.app.topology import Topology
from openmm.app.element import Element
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy
import numpy as np
import mdtraj

def create_system(encoded_pep_seq, amino_acid_pos, atom_names, forcefield, 
                  cyclic_bonds=None, return_topology=False):
    
    '''
    creates a system to compute energies in openmm
    '''
    
    topology = Topology()
    
    chain = topology.addChain(id="A")

    
    all_dicts = {}
    
    for index, char in enumerate(encoded_pep_seq):
        if char == 20:
            break
        name = Polypeptide.one_to_three(residue_mapping[char])
        
        residue = topology.addResidue(name, chain, id=index)
        
        residue_atom_names = np.array(atom_names)[amino_acid_pos[:len(atom_names)]==index+1].tolist()
        atom_actual_dict = {}
        
        for atom in residue_atom_names:
            
       
            
            atom_actual = topology.addAtom(atom, Element.getBySymbol(atom[0]), residue)
            

            atom_actual_dict[atom] = atom_actual
            
            
        all_dicts[index] = atom_actual_dict
        
        for pair in COVALENT_BOND_DICT[name]:
            
           
            if pair[0] not in atom_actual_dict or pair[1] not in atom_actual_dict:
                
                continue
            
            topology.addBond(atom_actual_dict[pair[0]],
                             atom_actual_dict[pair[1]])
        if index != 0:
            
            topology.addBond(atom_actual_dict["N"], all_dicts[index-1]["C"])
            
     
    if cyclic_bonds != None:
        
        for bond in cyclic_bonds:
            
            topology.addBond(all_dicts[int(bond[0].split("_")[0])-1][bond[0].split("_")[1]],
                             all_dicts[int(bond[1].split("_")[0])-1][bond[1].split("_")[1]])
            
    system = forcefield.createSystem(topology)
    
    
    if not return_topology:
        return system
    else:
        return system, topology

def output_structure(coordinates, atom_names, pep_seq, amino_acid_pos, output_file):
    
    '''
    Outputs structures to PDB files
    '''
    
    structure_builder = StructureBuilder.StructureBuilder()
    
    structure_builder.init_structure(output_file)
    

    
    for sample_id, coords in enumerate(coordinates):
        
        structure_builder.init_model(sample_id)
        
        structure = structure_builder.get_structure()
    
        model_object = structure.get_list()[sample_id]
    
        chain = Chain.Chain('A')
        
        model_object.add(chain)
    
        for residue_id, residue in enumerate(pep_seq):
        
            residue_object = Residue.Residue(id=(" ", residue_id+1, " "), resname=Polypeptide.one_to_three(residue),
                                             segid="")
            
            
            
            atoms = np.where(np.array(amino_acid_pos) == residue_id + 1)[0]
            
       
            
            
            for i in atoms:
                
                atom_name = atom_names[i]
              
                        
                atom_object = Atom.Atom(name=atom_name,
                                   coord=coords[i],
                                    bfactor=0, occupancy=1,
                                    altloc=" ",
                                    fullname=atom_name,
                                    element=atom_name[0],
                                    serial_number=i+1)
               
                residue_object.add(atom_object)
         
            chain.add(residue_object)
            
            
        

    pdbio = PDBIO()
    
    pdbio.set_structure(structure)
    
    pdbio.save(output_file)


def get_single_structure_prediction(file):
    traj = mdtraj.load(file)
    
    
    
    distances = np.empty((traj.n_frames, traj.n_frames))
    
    atom_indices = [a.index for a in traj.topology.atoms if a.name == 'CA']
    
    traj.center_coordinates()
    traj.superpose(traj, 0, atom_indices, atom_indices)
    
    for i in range(traj.n_frames):
        distances[i] = mdtraj.rmsd(traj, traj, i, atom_indices=atom_indices)
        
        
    reduced_distances = squareform(distances, checks=False)
    


    linkage = scipy.cluster.hierarchy.linkage(reduced_distances, method='average')

    
    clusters = scipy.cluster.hierarchy.fcluster(linkage, 0.15, criterion='distance')

    
    
    mode_cluster = 0
    
    sum_cluster = 0
    
    for cluster in np.unique(clusters):
        
        sum_curr = np.sum(clusters == cluster)
        
        if sum_curr > sum_cluster:
            
            mode_cluster = cluster
            
            
            sum_cluster = sum_curr
    
    
    distances_cluster  = distances[clusters == mode_cluster, :][:, clusters == mode_cluster]
    beta = 1
    index = np.exp(-beta*distances_cluster / distances_cluster.std()).sum(axis=1).argmax()
    
   
    index_centroid = np.where([clusters == mode_cluster])[1][index]
    
    return index_centroid


def get_single_structure_prediction_deviations(file, deviations):
    
    if np.sum(deviations[-1, 0] < 1.5) < 2:
        
        return np.argmin(deviations[-1, 0])
        
    traj = mdtraj.load(file)[deviations[-1, 0] < 1.5]
    
    distances = np.empty((traj.n_frames, traj.n_frames))
    
    atom_indices = [a.index for a in traj.topology.atoms if a.name == 'CA']
    
    traj.center_coordinates()
    traj.superpose(traj, 0, atom_indices, atom_indices)
    
    for i in range(traj.n_frames):
        distances[i] = mdtraj.rmsd(traj, traj, i, atom_indices=atom_indices)
        
        
    reduced_distances = squareform(distances, checks=False)
    


    linkage = scipy.cluster.hierarchy.linkage(reduced_distances, method='average')

    
    clusters = scipy.cluster.hierarchy.fcluster(linkage, 0.15, criterion='distance')

    
    
    mode_cluster = 0
    
    sum_cluster = 0
    
    threshold = 0.15
    
    while sum_cluster < 2:
        clusters = scipy.cluster.hierarchy.fcluster(linkage, threshold, criterion='distance')
    
        
        
        
        for cluster in np.unique(clusters):
            
            sum_curr = np.sum(clusters == cluster)
           
            if sum_curr > sum_cluster:
                
                mode_cluster = cluster
                
                
                sum_cluster = sum_curr
        
            
        threshold += 0.05
    
    
    distances_cluster  = distances[clusters == mode_cluster, :][:, clusters == mode_cluster]
    beta = 1
    index = np.exp(-beta*distances_cluster / distances_cluster.std()).sum(axis=1).argmax()
    
   
    index_centroid = np.where([clusters == mode_cluster])[1][index]
    
    return np.argwhere([deviations[-1, 0] < 1.5])[:, 1][index_centroid]