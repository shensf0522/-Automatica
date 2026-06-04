import os
import torch
from models import FAT,AMD,FAT_CAFI,GAP_new,GAP_codex,FAT_resi,FAT_resi_fft,FAT_resi_struct_explict_add,FAT_resi_fft_att_gated_detach,FAT_resi_fft_unify_encoder
from models import FAT_resi_struct_explict_add_cross_attention,FAT_resi_cross_attention_res,FAT_resi_memory_bank,FAT_ablation,FAT_resi_ROPE
from models import FAT_resi_struct_explict_add_cross_attention_dfbp,FAT_interPDN,FAT_interPDN_v2,FAT_res_trend_gate_new
class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
                           'FAT':FAT,
                           'AMD':AMD,
                           'FAT_CAFI':FAT_CAFI,
                           'GAP':GAP_new,
                           'GAP_codex':GAP_codex,
                           'FAT_resi':FAT_resi,
                           'FAT_resi_fft':FAT_resi_fft,
                           'FAT_struct_gate':FAT_resi_struct_explict_add,
                           'FAT_gate_detach':FAT_resi_fft_att_gated_detach,
                           'FAT_unify_encoder':FAT_resi_fft_unify_encoder,
                           'FAT_VLM_Cross_att':FAT_resi_struct_explict_add_cross_attention,
                           'FAT_VLM_Cross_att_resi':FAT_resi_cross_attention_res,
                           'FAT_VLM_Cross_att_mem_bank':FAT_resi_memory_bank,
                           'FAT_abl':FAT_ablation,
                           'FAT_resi_ROPE':FAT_resi_ROPE,
                           'FAT_VLM_Cross_att_dfbp':FAT_resi_struct_explict_add_cross_attention_dfbp,
                           'FAT_interPDN':FAT_interPDN,
                           'FAT_interPDN_v2':FAT_interPDN_v2,
                           'FAT_res_trend_gate_new':FAT_res_trend_gate_new
                           }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            if self.args.task_name == "finetune" and self.args.task_type == 'c':
                device = torch.device('cuda:0')
            else:
                device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
