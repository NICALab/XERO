import argparse,hashlib,io,json,math,os,sys,time
from pathlib import Path
import h5py,numpy as np,torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from release_utils import write_json
from fiarc_bank import TeacherBank,BankTileCache as OriginalCache
from boundary_training import make_cache_class,history_for_target as boundary_history
BankTileCache=make_cache_class(OriginalCache)
from history_policy import state_policy,advance
from training_stability import learning_rate,distortion
from seeded_spatial_codec import SeededSpatialCodec
from sequence_codec import prepare_frame,encode_group,GroupDecoder
from xero._primitives.models.causal_pixel_predictor import CausalPixelPredictor
from xero._primitives.models.chunk_forecast import forecast_chunk,corrected_history
from xero._primitives.models.grouped_spatial_codec import bounded_symbols

def snapshot(base,codec,mean,std,step,width,opt=None):
    state=dict(format='fiarc_multirecord_teacher_v1',step=step,tolerance=codec.tolerance,
        base=dict(history=base.history,width=width,input_mode='relative_latest',mean=mean,std=std,model=base.state_dict()),
        seed_config=codec.config,seed_model=codec.state_dict())
    if opt is not None:state['optimizer']=opt.state_dict()
    return state


def build(state,device):
    state_policy(state)
    b=state['base'];base=CausalPixelPredictor(b['history'],b['width'],input_mode=b['input_mode']).to(device).eval();base.load_state_dict(b['model'])
    codec=SeededSpatialCodec(**state['seed_config']).to(device).eval();codec.load_state_dict(state['seed_model'])
    codec.seed_entropy_offsets=codec.seed_entropy.weight.detach();codec.final_entropy_offsets=codec.final_entropy.weight.detach()
    return base,codec


@torch.no_grad()
def byte_audit(state,movie,out,device,frames=8,side=128,odd=False):
    """Reinstantiate from serialized weights; decoder never receives source pixels."""
    out.mkdir(parents=True,exist_ok=False);b=io.BytesIO();torch.save(state,b);weights=b.getvalue()
    state=torch.load(io.BytesIO(weights),map_location='cpu',weights_only=False)
    with h5py.File(movie['path']) as f:
        h,w=movie['shape'][1:];sy=min(side,h);sx=min(side,w)
        if odd:sy-=int(sy%2==0);sx-=int(sx%2==0)
        y=(h-sy)//2;x=(w-sx)//2;source=np.asarray(f['data'][:frames,y:y+sy,x:x+sx],dtype='f4')
    base,codec=build(state,device);mean=state['base']['mean'];std=state['base']['std'];tol=state['tolerance'];policy=state_policy(state)
    hist=torch.full((1,base.history,sy,sx),mean,device=device);prepared=[];recons=[]
    for i,s in enumerate(source):
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=forecast_chunk(base,hist,mean,std,1,i==0)
        packet,rec=prepare_frame(codec,torch.from_numpy(s).to(device)[None,None],pred,mean,std)
        prepared.append(packet);recons.append(rec);hist=advance(hist,rec,i==0,policy)
    payload=encode_group(prepared,streams=64);(out/'packet.bin').write_bytes(payload);(out/'weights.pt').write_bytes(weights)
    expected=torch.cat(recons)[:,0].cpu();del base,codec,hist,prepared,recons
    base,codec=build(state,device);hist=torch.full((1,base.history,sy,sx),mean,device=device)
    reader=GroupDecoder((out/'packet.bin').read_bytes(),device);decoded=[]
    for i in range(len(source)):
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=forecast_chunk(base,hist,mean,std,1,i==0)
        rec=reader.decode_frame(codec,pred,mean,std);decoded.append(rec);hist=advance(hist,rec,i==0,policy)
    result=torch.cat(decoded)[:,0].cpu();err=(torch.from_numpy(source).double()-result.double()).abs();bad=err>tol*torch.from_numpy(source).double().abs()
    if not torch.equal(expected,result) or bool(bad.any()):raise RuntimeError('Independent model byte reconstruction/bound failure')
    report=dict(step=state['step'],source=movie['path'],source_sha256=movie.get('sha256'),role=movie['role'],shape=list(source.shape),tolerance=tol,streams=64,history_initialization=policy,outer_format=state['format'],
        payload_bytes=len(payload),model_bytes=len(weights),payload_bpv=8*len(payload)/source.size,total_bpv=8*(len(payload)+len(weights))/source.size,
        violations=int(bad.sum()),rmse=float(torch.mean(err.square()).sqrt()),bitwise_encoder_decoder_equal=True,
        encoder_sha256=hashlib.sha256(expected.numpy().tobytes()).hexdigest(),
        note='Small byte audit, same process/fresh models. Final benchmark requires separate-process verification and full representative movies; no speed claim.')
    write_json(out/'report.json',report);return report


def plots(out,rows,target,result,step):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for a,keys in zip(ax,[['rate_bpv','seed_rate','final_rate'],['temporal_failure','proposal_failure','final_nonzero'],['step_ms','cache_wait_ms']]):
        for key in keys:a.plot([r['step'] for r in rows],[r[key] for r in rows],label=key,alpha=.7)
        a.legend();a.grid(alpha=.2);a.set_xlabel('step')
    fig.savefig(out/'curves.png',dpi=120)
    for a in ax:
        values=np.concatenate([v.get_ydata() for v in a.lines]);lo,hi=np.percentile(values,[1,99]);pad=max((hi-lo)*.1,1e-6);a.set_ylim(lo-pad,hi+pad)
    fig.savefig(out/'curves_robust.png',dpi=120);plt.close(fig)
    x=target[0,0].detach().float().cpu().numpy();p=result['post'][0,0].detach().float().cpu().numpy();r=result['rec'][0,0].detach().float().cpu().numpy()
    fig,ax=plt.subplots(1,4,figsize=(14,4),constrained_layout=True);lo,hi=np.percentile(x,[1,99.5])
    for a,v,label in zip(ax,[x,p,r,x-r],['SUPPORT target','prediction','corrected','target-corrected']):
        if label=='target-corrected':lim=max(float(abs(v).max()),1e-6);a.imshow(v,cmap='coolwarm',vmin=-lim,vmax=lim)
        else:a.imshow(v,cmap='gray',vmin=lo,vmax=hi)
        a.set_title(label);a.axis('off')
    fig.suptitle(f'FiARC {step} — original-intensity target');fig.savefig(out/'preview.png',dpi=120);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--steps',type=int,default=100000);p.add_argument('--tolerance',type=float,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--history',type=int,default=64);p.add_argument('--rollout',type=int,default=5);p.add_argument('--batch',type=int,default=32)
    p.add_argument('--width',type=int,default=32);p.add_argument('--refiner-width',type=int,default=16);p.add_argument('--patch',type=int,default=128)
    p.add_argument('--tiles',type=int,default=32);p.add_argument('--refresh',type=int,default=1000);p.add_argument('--seed',type=int,default=20260909)
    p.add_argument('--audit-every',type=int,default=5000);p.add_argument('--audit-frames',type=int,default=8);p.add_argument('--compile',action='store_true')
    p.add_argument('--init-checkpoint',type=Path)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--min-learning-rate',type=float,default=2e-5)
    p.add_argument('--decay-start',type=int,default=50000)
    p.add_argument('--first-distortion-weight',type=float,default=1.)
    p.add_argument('--cache-generation-offset',type=int,default=0)
    p.add_argument('--max-median-rate-bpv',type=float)
    p.add_argument('--stop-at',type=int,help='Checkpointed segment boundary; does not redefine the total LR schedule')
    p.add_argument('--lr-scale',type=float,default=1.,help='Explicit external plateau-controller LR multiplier')
    p.add_argument('--skip-short-audits',action='store_true',help='Use external deployment-profile calibration instead')
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.resume and a.init_checkpoint:p.error('Resume and fresh-optimizer initialization are distinct')
    learning_rate(1,a.steps,a.learning_rate,a.min_learning_rate,a.decay_start)
    if not math.isfinite(a.lr_scale) or not 0<a.lr_scale<=1 or (a.stop_at is not None and not 1<=a.stop_at<=a.steps):
        p.error('Invalid adaptive segment boundary/LR multiplier')
    end=a.stop_at if a.stop_at is not None else a.steps
    if not math.isfinite(a.first_distortion_weight) or not 0<a.first_distortion_weight<=1:
        p.error('First-chunk distortion weight must be in(0,1]')
    if a.cache_generation_offset<0 or (a.max_median_rate_bpv is not None and
            (not math.isfinite(a.max_median_rate_bpv) or a.max_median_rate_bpv<=0)):
        p.error('Invalid cache offset or rate health guard')
    bank=TeacherBank(a.bank,allow_smoke=a.smoke);mean,std=bank.normalization();tol=a.tolerance
    a.out.mkdir(parents=True,exist_ok=a.resume)
    torch.set_num_threads(2);torch.cuda.set_device(a.device);torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    base=CausalPixelPredictor(a.history,a.width,input_mode='relative_latest').to(a.device)
    codec=SeededSpatialCodec(width=a.refiner_width,grid_stride=2,conditioning='relative',selection='lattice',initial_seed_error_percent=1.2,
        seed_tolerance=tol,tolerance=tol,conditional_final_entropy=True,conditional_seed_entropy=True,even_refinement_gain=1.,
        even_refinement_weights=[.25]*4,learn_even_refinement=True,seed_gradient='detached').to(a.device)
    initialization=None
    if a.init_checkpoint:
        origin=json.loads((a.init_checkpoint.parent/'configuration.json').read_text())
        cp=torch.load(a.init_checkpoint,map_location=a.device,weights_only=False)
        if (origin['bank_plan']!=bank.manifest['plan'] or cp['tolerance']!=tol or
                cp['base']['history']!=a.history or cp['base']['width']!=a.width or
                (cp['base']['mean'],cp['base']['std'])!=(mean,std) or cp['seed_config']!=codec.config):
            raise ValueError('Initialization differs from target bank, bound, or architecture')
        base.load_state_dict(cp['base']['model']);codec.load_state_dict(cp['seed_model'])
        initialization=dict(checkpoint=str(a.init_checkpoint),checkpoint_sha256=hashlib.sha256(a.init_checkpoint.read_bytes()).hexdigest(),
            source_step=cp['step'],optimizer='fresh AdamW, no inherited moments',step_count='additional local steps, not from-scratch training')
        del cp
    params=list(base.parameters())+list(codec.parameters());opt=torch.optim.AdamW(params,lr=a.learning_rate,weight_decay=1e-5);start=0
    config=dict(arguments=vars(a),bank_plan=bank.manifest['plan'],mean=mean,std=std,parameters=sum(v.numel() for v in params),
        loss='rate +16*post Huber +0.4*temporal Huber, Huber scaled by tolerance*abs(target)',
        history='64 by default, same-window source quantized context +5 detached current-network corrected rollouts; fully recursive byte audits',
        bound='original float32 SUPPORT target; zero absolute floor; 64 entropy streams',normalization='training bank only',
        selection='none from heldout test; calibration audits available; best_estimated is only a surrogate candidate',
        initialization=initialization,first_distortion_weight=a.first_distortion_weight,
        first_weight_note='Only post/temporal Huber terms reweighted for sampled first chunks; rate, sampling probability, hard symbols/epsilon and runtime are unchanged. Unweighted distortion is also logged.')
    if a.resume:
        old=json.loads((a.out/'configuration.json').read_text());assert old['bank_plan']==config['bank_plan'] and old['arguments']['tolerance']==tol
        for key,default in [('learning_rate',1e-4),('min_learning_rate',2e-5),('decay_start',50000),('first_distortion_weight',1.)]:
            if old['arguments'].get(key,default)!=getattr(a,key):raise ValueError('Resume changes a frozen training control')
        initialization=old.get('initialization')
        cp=torch.load(a.out/'latest.pt',map_location=a.device,weights_only=False);base.load_state_dict(cp['base']['model']);codec.load_state_dict(cp['seed_model']);opt.load_state_dict(cp['optimizer']);start=cp['step']
    else:write_json(a.out/'configuration.json',config)
    if end<=start:raise ValueError('Segment boundary must exceed the loaded checkpoint step')
    write_json(a.out/'active_segment.json',dict(start=start,end=end,total_schedule_steps=a.steps,
        lr_scale=a.lr_scale,skip_short_audits=a.skip_short_audits))
    net=torch.compile(codec) if a.compile else codec;predictor=torch.compile(base) if a.compile else base
    cache=BankTileCache(bank,a.device,a.seed,a.patch,a.tiles,a.refresh,a.batch,a.history,a.rollout,generation_offset=a.cache_generation_offset,start_step=start)
    writer=SummaryWriter(a.out/'tensorboard',purge_step=start+1);rows=[];best=float('inf');waiting=0.;count=0;tick=time.monotonic()
    if a.resume:rows=[json.loads(l) for l in (a.out/'metrics.jsonl').read_text().splitlines() if json.loads(l)['step']<=start]
    try:
        for step in range(start+1,end+1):
            seq,wait,refreshed=cache.next(step);waiting+=wait;count+=1
            if refreshed:
                with (a.out/'cache.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,generation=cache.generation-1,meta=cache.meta,load_seconds=cache.load_seconds,wait_seconds=wait))+'\n')
            hist,target,first=boundary_history(seq,cache.boundary_target,a.history,a.rollout,mean,std,tol,
                predictor,net,codec,forecast_chunk,bounded_symbols,corrected_history,
                lambda:torch.autocast('cuda',dtype=torch.bfloat16))
            opt.zero_grad(set_to_none=True)
            lr=learning_rate(step,a.steps,a.learning_rate,a.min_learning_rate,a.decay_start)*a.lr_scale
            for g in opt.param_groups:g['lr']=lr
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=forecast_chunk(predictor,hist,mean,std,1,first);result=net(target,pred,mean,std,'learned')
                scale=(tol*target.abs()).clamp_min(1e-6)
                post,post_unweighted=distortion(result['post'],target,scale,first,a.first_distortion_weight)
                temporal,temporal_unweighted=distortion(pred,target,scale,first,a.first_distortion_weight)
                loss=result['loss']+16*post+.4*temporal
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True);opt.step()
            if step%25==0 or step==end:
                bad=(target.double()-result['rec'].detach().double()).abs()>tol*target.double().abs()
                values=torch.stack([loss.detach(),result['loss'].detach(),result['seed_rate'].detach(),result['final_rate'].detach(),
                    ((pred.detach().double()-target.double()).abs()>tol*target.double().abs()).float().mean(),
                    ((result['post'].detach().double()-target.double()).abs()>tol*target.double().abs()).float().mean(),
                    ((result['final_q']!=0)&~result['mask'].detach().bool()).float().mean(),post.detach(),temporal.detach(),norm.detach(),post_unweighted.detach(),temporal_unweighted.detach(),first.float().mean(),bad.sum().float()]).cpu().tolist()
                if not np.isfinite(values).all() or values[-1]:raise RuntimeError('Nonfinite training or hard-bound failure')
                row=dict(zip(['loss','rate_bpv','seed_rate','final_rate','temporal_failure','proposal_failure','final_nonzero','post_huber','temporal_huber','grad_norm','post_huber_unweighted','temporal_huber_unweighted','first_fraction','violations'],values))
                row.update(step=step,lr=lr,step_ms=1000*(time.monotonic()-tick)/count,cache_wait_ms=1000*waiting/count,wall_time=time.time())
                rows.append(row)
                with (a.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                write_json(a.out/'status.json',dict(state='training',**row));print(json.dumps(row),flush=True)
                for k,v in row.items():writer.add_scalar('train/'+k,v,step)
                if a.max_median_rate_bpv is not None and len(rows)>=8:
                    recent=[r['rate_bpv'] for r in rows[-8:]]
                    if float(np.median(recent))>a.max_median_rate_bpv:
                        alarm=dict(state='stopped_unhealthy',step=step,recent_logged_rates=recent,
                            median_rate=float(np.median(recent)),limit=a.max_median_rate_bpv,
                            note='Training surrogate-rate health guard, not a fidelity threshold or final compression verdict.')
                        write_json(a.out/'unhealthy.json',alarm);write_json(a.out/'status.json',alarm)
                        alarm_state=snapshot(base,codec,mean,std,step,a.width,opt)
                        if initialization:alarm_state['training_initialization']=initialization
                        torch.save(alarm_state,a.out/'unhealthy.pt')
                        raise RuntimeError('Persistent training-rate deterioration; inspect unhealthy.json')
                writer.flush();tick=time.monotonic();count=0;waiting=0.
            if step%500==0 or step==end:plots(a.out,rows,target,result,step)
            if step%2500==0 or step==end:
                state=snapshot(base,codec,mean,std,step,a.width,opt)
                if initialization:state['training_initialization']=initialization
                tmp=a.out/'latest.tmp';torch.save(state,tmp);tmp.replace(a.out/'latest.pt')
                torch.save(state,a.out/f'checkpoint_{step:07d}.pt')
                if rows[-1]['rate_bpv']<best:best=rows[-1]['rate_bpv'];torch.save(state,a.out/'best_estimated.pt')
            if not a.skip_short_audits and (step%a.audit_every==0 or step==end):
                state=snapshot(base,codec,mean,std,step,a.width)
                if initialization:state['training_initialization']=initialization
                for role,movie in [('train',bank.train[0]),('calibration',bank.calibration[0])]:
                    report=byte_audit(state,movie,a.out/'audits'/f'{step:07d}_{role}',a.device,a.audit_frames,a.patch,odd=role=='calibration' and a.smoke)
                    for k in ['payload_bpv','total_bpv','violations']:writer.add_scalar(role+'_audit/'+k,report[k],step)
                    print('BYTE_AUDIT',json.dumps(report),flush=True)
                tick=time.monotonic()
        marker='complete.json' if end==a.steps else 'segment_complete.json'
        write_json(a.out/marker,dict(step=step,smoke=a.smoke,lr_scale=a.lr_scale,
            note='Training boundary reached; not final benchmark/quality selection'))
    finally:cache.close();writer.close()
