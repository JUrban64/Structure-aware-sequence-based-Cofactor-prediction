import numpy as np
from Bio.PDB import PDBParser, Selection
from scipy.spatial.distance import cdist
import sys 
import glob


class BindingSiteExtractor:
    """
    Extrahuje binding site residues z PDB struktury
    """
    
    def __init__(self, distance_threshold=6.0):
        self.threshold = distance_threshold
        self.parser = PDBParser(QUIET=True)
    
    def extract_binding_site(self, pdb_file, ligand_name='NAD'):
        """
        Args:
            pdb_file: path to PDB file
            ligand_name: residue name of ligand (NAD, ATP, FAD, etc.)
        
        Returns:
            binding_site_info: dict with binding site data
        """
        structure = self.parser.get_structure('protein', pdb_file)
        
        # Get protein chain (usually chain A)
        model = structure[0]
        protein_chain = None
        for chain in model:
            if len(list(chain.get_residues())) > 10:  # Skip small chains
                protein_chain = chain
                break
        
        if protein_chain is None:
            raise ValueError("No protein chain found")
        
        # Extract full sequence
        full_sequence = self._get_sequence(protein_chain)
        
        # Get ligand coordinates
        ligand_coords = self._get_ligand_coords(structure, ligand_name)
        
        if ligand_coords is None:
            raise ValueError(f"Ligand {ligand_name} not found")
        
        # Find binding site residues
        binding_site_residues = []
        binding_site_indices = []
        
        for i, residue in enumerate(protein_chain.get_residues()):
            if not self._is_aa(residue):
                continue
            
            # Get all atom coordinates for this residue
            residue_coords = np.array([
                atom.get_coord() 
                for atom in residue.get_atoms()
            ])
            
            # Compute minimum distance to ligand
            distances = cdist(residue_coords, ligand_coords)
            min_dist = distances.min()
            
            if min_dist <= self.threshold:
                binding_site_residues.append(residue)
                binding_site_indices.append(i)
        
        # Extract local contact map
        local_contact_map = self._compute_local_contact_map(
            binding_site_residues
        )
        
        # Get sequences
        bs_sequence = ''.join([
            self._three_to_one(res.get_resname()) 
            for res in binding_site_residues
        ])
        
        return {
            'full_sequence': full_sequence,
            'binding_site_sequence': bs_sequence,
            'binding_site_indices': binding_site_indices,
            'binding_site_residues': binding_site_residues,
            'contact_map': local_contact_map,
            'n_binding_site': len(binding_site_residues),
            'ligand_name': ligand_name,
            'pdb_file': pdb_file
        }
    
    def _get_ligand_coords(self, structure, ligand_name):
        """Extract ligand atom coordinates"""
        ligand_coords = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() == ligand_name:
                        for atom in residue.get_atoms():
                            ligand_coords.append(atom.get_coord())
        
        if len(ligand_coords) == 0:
            return None
        
        return np.array(ligand_coords)
    
    def _compute_local_contact_map(self, residues, threshold=8.0):
        """
        Compute contact map only within binding site residues
        """
        n = len(residues)
        contact_map = np.zeros((n, n))
        
        # Get CA coordinates
        ca_coords = []
        for res in residues:
            if 'CA' in res:
                ca_coords.append(res['CA'].get_coord())
            else:
                # Fallback to centroid if no CA
                coords = [atom.get_coord() for atom in res.get_atoms()]
                ca_coords.append(np.mean(coords, axis=0))
        
        ca_coords = np.array(ca_coords)
        
        # Compute pairwise distances
        dist_matrix = cdist(ca_coords, ca_coords)
        
        # Contact if distance < threshold
        contact_map = (dist_matrix < threshold).astype(float)
        
        return contact_map
    
    def _is_aa(self, residue):
        """Check if residue is amino acid"""
        return residue.get_id()[0] == ' '
    
    def _get_sequence(self, chain):
        """Extract sequence from chain"""
        three_to_one = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        
        sequence = []
        for residue in chain.get_residues():
            if self._is_aa(residue):
                resname = residue.get_resname()
                if resname in three_to_one:
                    sequence.append(three_to_one[resname])
        
        return ''.join(sequence)
    
    def _three_to_one(self, three_letter):
        """Convert 3-letter to 1-letter AA code"""
        conversion = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        return conversion.get(three_letter, 'X')


# Použití
extractor = BindingSiteExtractor(distance_threshold=6.0)

pdb_files = glob.glob('./*.pdb')  # Zadejte cestu k PDB souborům



print(extractor.extract_binding_site(pdb_files[0], ligand_name='NAD'))

binding_sites = []
for pdb_file in pdb_files:  # 2000 PDB struktur
    try:
        bs_info = extractor.extract_binding_site(pdb_file, ligand_name='NAD')
        binding_sites.append(bs_info)
        print(f"{pdb_file}: {bs_info['n_binding_site']} residues in binding site")
    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")

print(f"Successfully processed {len(binding_sites)} structures")