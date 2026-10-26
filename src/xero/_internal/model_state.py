"""Causal checkpoint adapter; historical format IDs retain compatibility."""
from history_policy import with_policy, state_policy
from ficompress.models.causal_pixel_predictor import CausalPixelPredictor
from seeded_spatial_codec import SeededSpatialCodec


def model_profile(state):
    if state.get('format') not in ('fiarc_multirecord_teacher_v1', 'fiarc_multirecord_teacher_v2'):
        raise ValueError('Unsupported checkpoint format')
    if 'bidirectional' in state or 'context_future' in state:
        raise ValueError('This release supports causal XERO checkpoints only')
    return 'causal'


def export_state(state):
    model_profile(state)
    result = dict(state)
    result.pop('optimizer', None)
    return with_policy(result, 'repeat_first_decoded')


def build_causal(state, device):
    state_policy(state)
    b = state['base']
    base = CausalPixelPredictor(b['history'], b['width'], input_mode=b['input_mode']).to(device).eval()
    base.load_state_dict(b['model'])
    codec = SeededSpatialCodec(**state['seed_config']).to(device).eval()
    codec.load_state_dict(state['seed_model'])
    codec.seed_entropy_offsets = codec.seed_entropy.weight.detach()
    codec.final_entropy_offsets = codec.final_entropy.weight.detach()
    return base, codec
