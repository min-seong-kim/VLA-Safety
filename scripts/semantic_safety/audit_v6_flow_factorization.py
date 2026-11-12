import sys,json,torch
from pathlib import Path
sys.path.insert(0,'scripts/semantic_safety')
from motor_lora import load_policy,tensor_batch
from v6_scoped_lora import install,loss_matrix
from train_v6_scoped_lora import Demonstrations
from rollout_common import write_json
p,_=load_policy();m=p._model;arch=install(m,'all');d=Demonstrations(Path('data/semantic_motor_v6_lora_v1/pilot16/calibration_manifest.jsonl'),p,7)
o,a=tensor_batch(d.batch(2));torch.manual_seed(772);noise=m.sample_noise(a.shape,'cuda');time=torch.tensor([.25,.75],device='cuda')
pre=m._preprocess_observation
m._preprocess_observation=lambda observation,train=False:pre(observation,train=False)
selected=[(n,p) for n,p in m.named_parameters() if n.endswith('lora_b') and any(s in n for s in ['vision_model.encoder.layers.0.self_attn.q_proj','language_model.layers.0.self_attn.k_proj','gemma_expert.model.layers.0.self_attn.q_proj'])]
assert len(selected)==3, [n for n,p in selected]
joint=m(o,a,noise=noise,time=time);jg=torch.autograd.grad(joint.mean(),[p for n,p in selected])
cached=loss_matrix(m,o,a,'all',noise=noise,time=time);cg=torch.autograd.grad(cached.mean(),[p for n,p in selected])
result={'joint_loss':joint.mean().item(),'cached_loss':cached.mean().item(),'max_element_loss_error':(joint-cached).abs().max().item(),'gradient_relative_errors':{n:((x-y).norm()/(x.norm()+1e-12)).item() for (n,p),x,y in zip(selected,jg,cg)}}
result['passed']=abs(result['joint_loss']-result['cached_loss'])<.001 and max(result['gradient_relative_errors'].values())<.1
write_json(Path('runs/semantic_motor_v6_lora_v1/flow_equivalence_audit.json'),result);print(result,flush=True);assert result['passed']
# Remove BF16 rounding to test the causal factorization itself.
torch.set_float32_matmul_precision("highest");torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
m.float()
joint=m(o,a,noise=noise,time=time);jg=torch.autograd.grad(joint.mean(),[p for n,p in selected])
cached=loss_matrix(m,o,a,'all',noise=noise,time=time);cg=torch.autograd.grad(cached.mean(),[p for n,p in selected])
f32={'joint_loss':joint.mean().item(),'cached_loss':cached.mean().item(),'max_element_loss_error':(joint-cached).abs().max().item(),'gradient_relative_errors':{n:((x-y).norm()/(x.norm()+1e-12)).item() for (n,p),x,y in zip(selected,jg,cg)}}
f32['passed']=f32['max_element_loss_error']<.0001 and max(f32['gradient_relative_errors'].values())<.001
write_json(Path('runs/semantic_motor_v6_lora_v1/flow_equivalence_float32_audit.json'),f32);print(f32,flush=True);assert f32['passed']
