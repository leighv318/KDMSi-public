"""
Created on Tue Feb 22 23:51:54 2022

@author: Haotian Teng
"""
import os
import pysam
import genomes
import pandas as pd
SUPPORTED_GENOME = ["GRCh38","GRCm39"]
GENOMES_FOLDER = os.path.dirname(genomes.__file__)

class Genome(object):
    def __init__(self, genome_id:str, data:str = 'dna'):
        """
        Construct the genome object.

        Parameters
        ----------
        genome_id : str
            The id of a genome.
        data : str, optional
            The type of genome, can be dna or rna. The default is 'dna'.

        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if not genome_id in SUPPORTED_GENOME:
            raise ValueError("Genome id %s not recognized, currently supported genome is %s"%(genome_id,','.join(SUPPORTED_GENOME)))
        if not os.path.exists(os.path.join(GENOMES_FOLDER,"%s_%s_ref"%(genome_id,data))):
            try:
                genomes.download(genome_id)
            except Exception as error:
                raise ValueError("Can not download genome %s: %s"%(genome_id,error))
        self.reference = {}
        ref_f = os.path.join(GENOMES_FOLDER,"%s_%s_ref"%(genome_id,data))
        with pysam.FastxFile(os.path.join(ref_f,"reference.fasta")) as f:
            for entry in f:
                self.reference[entry.name] = entry.sequence
        self.look_up_table = pd.read_csv(os.path.join(ref_f,"ref_dict.csv"),
                                         delimiter = "\t")
        
    def get_sequences_by_gene(self, gene_name:str):
        names = self.look_up_table[self.look_up_table['geneName'] == gene_name]['name']
        seqs = {}
        for name in names:
            hits = [x for x in self.reference.keys() if name in x ]
            for hit in hits:
                seqs[hit] = self.reference[hit]
        return seqs

'''
if __name__ == "__main__":
    g = Genome("GRCh38",data = "rna")
    output = "output_mrna"
    genes = ['COL6A2','ANK2','EDN1','LITAF','PDE4DIP','PERP','PMAIP1','SPTBN1','TRIM22','VCAN']
    for gene in genes:
        seqs = g.get_sequences_by_gene(gene)
        os.makedirs(output,exist_ok=True)
        with open(os.path.join(output,'%s.fasta'%(gene)),'w+') as f:
            for n in seqs.keys():
                f.write(">"+n+"\n")
                f.write(seqs[n]+"\n")
'''