import torch
from . import alignments as alg

def compute_eba(similarity_matrix, extensive_output=False, gap_open_penalty=0.0, gap_extend_penalty=0.0):
    """Computes the embedding-based alignment score (EBA) for a pair of sequences.

        :similarity_matrix: matrix containing paiwise similarity scores
        :param extensive_output: if True returns also the alignment
        :param gap_open_penalty: open gap penality for the global alignment
        :param gap_extend_penalty: extend gap penality for the global alignment

        :type similarity_matrix: pytorch tensor
        :type gap_open_penalty: float
        :type gap_extend_penalty: float

    """
    aln_1, aln_2, EBA_raw = alg.dtw_align(similarity_matrix.cpu().numpy(), 
                                            gap_open_penalty=gap_open_penalty, 
                                            gap_extend_penalty=gap_extend_penalty)

    l_min = min(similarity_matrix.shape[0], similarity_matrix.shape[1])
    l_max = max(similarity_matrix.shape[0], similarity_matrix.shape[1])

    if extensive_output:
        return {'EBA_raw': EBA_raw, 'EBA_min': EBA_raw/l_max, 'EBA_max': EBA_raw/l_min,
                'aln_1':aln_1, 'aln_2':aln_2}
    else:
        return {'EBA_raw': EBA_raw, 'EBA_min': EBA_raw/l_max, 'EBA_max': EBA_raw/l_min}


def compute_eba_local(similarity_matrix, gap_penalty=0.0):
    """Computes the embedding-based alignment score (EBA) for a pair of sequences.

        :similarity_matrix: matrix containing paiwise similarity scores
        :param gap_penalty: gap penality for the local alignment

        :type similarity_matrix: pytorch tensor
        :type gap_penalty: float

    """
    return alg.smith_waterman(similarity_matrix.cpu().numpy(), gap=gap_penalty)


def compute_eba_dumb(similarity_matrix):
    """Computes a score based on the EBA similarity matrix.
    
        :similarity_matrix: matrix containing paiwise similarity scores

        :type similarity_matrix: pytorch tensor

    """
    values_row,idxs_row = torch.max(similarity_matrix,dim=1)
    values_clm,idxs_clm = torch.max(similarity_matrix,dim=0)
    
    eba_row = values_row.sum()/similarity_matrix.shape[0]
    eba_clm = values_clm.sum()/similarity_matrix.shape[1]

    return {'EBA_min': min(eba_row, eba_clm), 'EBA_max': max(eba_row, eba_clm)}