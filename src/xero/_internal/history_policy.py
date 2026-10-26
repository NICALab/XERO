"""Explicit outer-state versioning for decoder-causal history initialization."""
import torch

V1='fiarc_multirecord_teacher_v1'
V2='fiarc_multirecord_teacher_v2'
LEGACY_AHRENS='ahrens_fixed_relative_v1'
POLICIES=('mean','repeat_first_decoded')


def state_policy(state):
    version=state.get('format')
    if version in (V1,LEGACY_AHRENS):
        if state.get('history_initialization','mean')!='mean':raise ValueError('V1 requires legacy mean initialization')
        return 'mean'
    if version==V2 and state.get('history_initialization') in POLICIES:return state['history_initialization']
    raise ValueError('Unknown or incomplete outer codec state/profile')


def with_policy(state,policy):
    state_policy(state)
    if policy not in POLICIES:raise ValueError('Unknown history initialization')
    result=dict(state)
    if policy=='mean':
        result['format']=V1;result.pop('history_initialization',None)
    else:result.update(format=V2,history_initialization=policy)
    return result


def advance(history,reconstruction,first,policy):
    if policy not in POLICIES or not isinstance(first,bool):raise ValueError('Explicit supported policy and frame flag required')
    if first and policy=='repeat_first_decoded':
        # Only already decoded pixels; supports a multi-frame first chunk too.
        history=reconstruction[:,:1].expand(-1,history.shape[1],-1,-1)
    return torch.cat([history,reconstruction],1)[:,-history.shape[1]:]
