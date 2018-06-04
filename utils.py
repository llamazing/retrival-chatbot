import sys


filters = [".", ",", "-", "?", "!", "'", '"']



def get_attn_padding_mask(seq_q, seq_k, pad_id=0):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(pad_id).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask

import logging
def logger_set(gLogger, log_path=None):
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    gLogger.addHandler(handler)

    if log_path is not None:
        fhandler = logging.FileHandler(log_path)
        fhandler.setFormatter(formatter)
        gLogger.addHandler(fhandler)
    gLogger.setLevel(logging.INFO)
