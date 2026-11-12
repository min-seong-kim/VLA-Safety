"""Extract reusable object-relative grasp poses from prior development rollouts.

Only grasp transforms, not evaluation images/actions/states, become teacher parameters.
All resulting training trajectories are newly simulated and independently replayed.
"""
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from rollout_common import ROOT,bootstrap,write_json
from semantic_pairs_v6 import make_environment
from semantic_v6_diagnostics import items,RUN
from v6_lora_data import DATA


def main():
 bootstrap();result={};lookup={i['item_id']:i for i in items()}
 for asset in ['wallet','spoon','toy','foil','knife','kettle','wood_bowl','battery']:
  profiles=[]
  candidates=[]
  for arm in ['pickup_position','pickup_name','transport_name','original']:
   root=RUN/arm/'episodes'
   if not root.exists():
    root=RUN/'arms'/arm/'episodes'
   for path in root.glob('*/summary.json'):
    row=json.loads(path.read_text());item=lookup[row['item_id']]
    if item['actor_asset']==asset and row['ever_lifted_4cm']:candidates.append((path.parent,item,'policy'))
  for path in (RUN/'grasp_prefixes').glob('*/summary.json'):
   row=json.loads(path.read_text())
   if row['actor_asset']==asset and row['passed']:candidates.append((path.parent,lookup[row['item_id']],'teacher'))
  for folder,item,source in candidates:
   if source=='teacher':
    with np.load(folder/'prefix.npz') as f:simstate=f['sim_states'][-1];state=f['state']
   else:
    ev=[json.loads(x) for x in (folder/'events.jsonl').read_text().splitlines()]
    stable=[];run=0
    for e in ev:
     run=run+1 if e['lifted_4cm'] and e['grasped'] else 0
     if run>=8:stable.append(e['control_step'])
    if not stable:continue
    step=stable[0]
    with h5py.File(folder/'trajectory.h5') as f:simstate=f['observations/sim_state'][step];state=f['observations/state'][step]
   wrapper,obs,oracle=make_environment(item,64);env=wrapper.env
   env.sim.set_state_from_flattened(simstate);env.sim.forward()
   actor_r=env.sim.data.body_xmat[oracle.actor].reshape(3,3);actor_p=env.sim.data.body_xpos[oracle.actor]
   local_p=actor_r.T@(state[:3]-actor_p);local_r=actor_r.T@Rotation.from_rotvec(state[3:6]).as_matrix()
   profiles.append(dict(mode='object_pose',position_local=local_p.tolist(),rotation_local=local_r.tolist(),
                        source_type=source,development_source=str(folder.relative_to(ROOT)),
                        evaluation_frames_or_actions_used_in_training=False))
   wrapper.close()
  result[asset]=profiles;print(asset,len(profiles),flush=True)
 write_json(DATA/'development_grasp_geometry.json',result)

if __name__=='__main__':main()
