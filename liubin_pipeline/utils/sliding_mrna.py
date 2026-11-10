import pdb
import os
import pandas as pd
import numpy as np

def read_by_windows():
    os.makedirs('output/sirna', exist_ok=True)

    # get all mrna sequences
    # the key in all_fasta is the gene name, the value cur_fasta is a dict composed of its mrnas
    # the key in cur_fasta is the NM_ID, the value is a string of its sequence
    all_fasta = {}
    for root, dirs, files in os.walk('./output/mrna', topdown=True):
        for name in files:
            if name.endswith('.fasta'):
                mrna_path = os.path.join(root, name)
                cur_fasta = {}
                for nm_id, seq in readFa(mrna_path):
                    cur_fasta[nm_id] = seq.replace('T', 'U')
                all_fasta[name.split('.')[0]] = cur_fasta

    # for each gene, get and save its all possible sirnas in one file by sliding windows
    # and collect its all AS into the input folder of RNA-FM-ENCODING
    for cur_gene_name in all_fasta.keys():
        cur_slide_path = os.path.join('./output/sirna', str(cur_gene_name) + '.csv')
        # cur_rnafm_sirna_path = os.path.join('./RNAFM_ENCODING/redevelop/data/examples', str(cur_gene_name) + '_sirna.fasta')
        # cur_rnafm_mrna_path = os.path.join('./RNAFM_ENCODING/redevelop/data/examples', str(cur_gene_name) + '_mrna.fasta')
        if not os.path.exists(cur_slide_path):
            print('reading {} by sliding windows'.format(cur_gene_name))
            cur_fasta = all_fasta[cur_gene_name]
            all_as, all_ss, all_mrna, all_gene_name, all_nm_id, all_pos, all_rnafm_ind, all_as_energy, all_ss_energy = [], [], [], [], [], [], [], [], []
            mrna_length, ss_gc_content, ss_gu_content, as_immune_subseq, ss_start_aa, consecutive_4_same_nts = [], [], [], [], [], []
            rnafm_index = 0
            for cur_nm_id in cur_fasta.keys():
                pos_record = 23  # start position for as is 21, but for ss is 23
                cur_gene_sequence = cur_fasta[cur_nm_id]
                cur_mrna_len = len(cur_gene_sequence)
                for seq in readSeqByWindow(sequence=cur_gene_sequence, winSize=59, stepSize=1):
                    ss_seq = seq[20:41]
                    all_ss.append(ss_seq[2:]) # remove 2 nts from its 5'end
                    as_seq = convert_from_ss_to_as(ss_seq)  # 21-nt as
                    all_as.append(as_seq)
                    mrna_length.append(cur_mrna_len)
                    all_mrna.append(seq)
                    all_gene_name.append(cur_gene_name)
                    all_pos.append(pos_record)
                    all_nm_id.append(cur_nm_id)
                    # all_rnafm_ind.append(rnafm_index)
                    ss_gu_content.append(round((ss_seq[2:].count('G')+ss_seq[2:].count('U')) / 19 * 100, 3))
                    ss_gc_content.append(round((ss_seq[2:].count('G') + ss_seq[2:].count('C')) / 19 * 100, 3))
                    as_immune_subseq.append('N' if as_seq.find('GUCCUUCAA')==-1 else 'Y')
                    as_immune_subseq[-1]+='|N' if as_seq.find('UGUGU')==-1 else '|Y'
                    ss_start_aa.append('Y' if ss_seq[2:].startswith('AA') else 'N')
                    consecutive_4_same_nts.append('Y' if as_seq.find('AAAA')!=-1 or as_seq.find('UUUU')!=-1 or as_seq.find('CCCC')!=-1  or as_seq.find('GGGG')!=-1  else 'N')
                    all_as_energy.append(get_energy(as_seq))
                    all_ss_energy.append(get_energy(ss_seq[2:]))
                    pos_record += 1
                    # rnafm_index += 1
                    # with open(cur_rnafm_sirna_path, 'a') as f:
                    #     f.write('>'+str(rnafm_index).zfill(5)+'\n'+as_seq+'\n')
                    # with open(cur_rnafm_mrna_path, 'a') as f:
                    #     f.write('>'+str(rnafm_index).zfill(5)+'\n'+seq+'\n')

            df = pd.DataFrame(
                list(zip(all_gene_name, all_nm_id, all_as, all_ss, all_mrna, mrna_length,
                         all_pos, ss_gc_content, ss_gu_content, as_immune_subseq, ss_start_aa,
                         consecutive_4_same_nts)),
                columns=['gene', 'NM_id', 'AS', 'SS', 'local_mrna', 'mrna_len', 'ss_pos(_prefer_middle)',
                         'ss_GC%(_prefer_30-60)', 'ss_GU%(_prefer_30-65)', 'as_immune_GUCCUUCAA|UGUGU(_prefer_N)', 'ss_start_AA(_prefer_Y)',
                         '4_same_nts(_prefer_N)'])
            df.to_csv(cur_slide_path, index=0)


def readFa(fa):
    with open(fa,'r') as FA:
        seqName,seq='',''
        while 1:
            line=FA.readline()
            line=line.strip('\n')
            if (line.startswith('>') or not line) and seqName:
                yield ((seqName, seq))
            if line.startswith('>'):
                seqName = line[1:16]
                seq = ''
            else:
                seq += line
            if not line:break

def convert_from_ss_to_as(seq):
    ss_seq = list(seq)
    dict_t = {'A':'U', 'U':'A', 'G':'C', 'C':'G'}
    as_seq = [dict_t[ele] for ele in ss_seq]
    as_seq.reverse()
    return "".join(as_seq)

def readSeqByWindow(sequence,winSize,stepSize):
    if stepSize<=0: return False
    now = 0
    seqLen = len(sequence)
    while(now+winSize-stepSize<seqLen):
        yield sequence[now:now+winSize]
        now+=stepSize

def get_energy(seq):
    result = 0
    energy_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'table': np.array(
        [[-0.93, -2.24, -2.08, -1.1],
         [-2.11, -3.26, -2.36, -2.08],
         [-2.35, -3.42, -3.26, -2.24],
         [-1.33, -2.35, -2.11, -0.93]])}

    for i in range(len(seq)-1):
        index_1 = energy_dict.get(seq[i])
        index_2 = energy_dict.get(seq[i + 1])
        result += energy_dict['table'][index_1, index_2]

    return result







