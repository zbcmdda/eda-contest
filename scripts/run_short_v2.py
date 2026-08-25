#!/usr/bin/env python3
"""Run the gated A0/A1/A2 short-specialist development matrix.

Only the first public 800k rows and group-split pilot *dev* labels are read.
Sealed pilot labels and public rows 800k+ are deliberately out of scope.
"""
from __future__ import annotations

import argparse, hashlib, json, subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from short_features import NAMES, TWO_HOP_NAMES
from train_model import ArchitectureFeatures, official_metrics

PARAMS = {"objective":"regression", "metric":"l1", "num_leaves":31, "max_depth":6,
          "learning_rate":0.08, "min_data_in_leaf":20, "feature_fraction":0.95,
          "bagging_fraction":0.9, "bagging_freq":1, "lambda_l2":1.0, "verbosity":-1,
          "seed":20260823, "num_threads":12, "force_col_wise":True}

def args():
 p=argparse.ArgumentParser(); p.add_argument("--estimate",type=Path,default=Path("build/estimate")); p.add_argument("--arch",type=Path,default=Path("arch")); p.add_argument("--answers",type=Path,default=Path("data/delay_estimate_ans.csv")); p.add_argument("--pilot-metadata",type=Path,default=Path("refine-logs/short_v2_pilot/dev.metadata.csv")); p.add_argument("--pilot-labels",type=Path,default=Path("refine-logs/short_v2_pilot/labels/dev_exact.csv")); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--public-training-rows",type=int,default=800000); return p.parse_args()
def endpoint_dist(frame):
 s=frame.From.str.extract(r"SRB_(\d+)_(\d+)/").astype(int); t=frame.To.str.extract(r"SRB_(\d+)_(\d+)/").astype(int); return (s[0]-t[0]).abs()+(s[1]-t[1]).abs()
def bucket(key): return int.from_bytes(hashlib.sha256(("20260823|"+key).encode()).digest()[:8],"big")%5
def row_scores(y,p):
 m=y>0; result=np.zeros(len(y)); result[m]=(1-np.tanh(4*np.abs(p[m]-y[m])/y[m]))*100; return result
def ci_diff(y,p,base,groups):
 unique=np.asarray(sorted(groups.unique())); rng=np.random.default_rng(20260823); diff=[]
 for _ in range(1000):
  take=rng.choice(unique,len(unique),replace=True); idx=np.concatenate([np.flatnonzero(groups.to_numpy()==g) for g in take]); diff.append(float((row_scores(y[idx],p[idx])-row_scores(y[idx],base[idx])).mean()))
 return [float(np.quantile(diff,.025)),float(np.quantile(diff,.975))]
def dump(a, request, output):
 request[["From","To"]].to_csv(output.with_suffix(".request.csv"),index=False)
 subprocess.run([str(a.estimate),"--dump-short-features","-in",str(output.with_suffix(".request.csv")),"-out",str(output),"--arch",str(a.arch),"--limit",str(len(request))],check=True)
 return pd.read_csv(output,dtype={"From":"string","To":"string"})[NAMES].astype(np.int32)
def main():
 a=args(); a.output_dir.mkdir(parents=True,exist_ok=True)
 public=pd.read_csv(a.answers,nrows=a.public_training_rows,dtype={"From":"string","To":"string","delay":np.int32}); public=public.loc[endpoint_dist(public)<=32].reset_index(drop=True)
 meta=pd.read_csv(a.pilot_metadata,dtype="string"); labels=pd.read_csv(a.pilot_labels,dtype={"From":"string","To":"string","delay":np.int32})
 if not meta[["From","To"]].equals(labels[["From","To"]]): raise ValueError("pilot labels do not align")
 pilot=meta.join(labels[["delay"]]); pilot["dev_fold"]=pilot.group_key.map(bucket); pilot["delay"]=pilot.delay.astype(np.int32)
 public_short=dump(a,public,a.output_dir/"public_short_features.csv"); pilot_short=dump(a,pilot,a.output_dir/"pilot_short_features.csv")
 combined=pd.concat([public[["From","To","delay"]],pilot[["From","To","delay"]]],ignore_index=True)
 base_features=ArchitectureFeatures(a.arch).build(combined)
 base_public=base_features.iloc[:len(public)].reset_index(drop=True); base_pilot=base_features.iloc[len(public):].reset_index(drop=True)
 full_public=pd.concat([base_public.reset_index(drop=True),public_short.reset_index(drop=True)],axis=1)
 full_pilot=pd.concat([base_pilot.reset_index(drop=True),pilot_short.reset_index(drop=True)],axis=1)
 eval_rows=pilot.loc[pilot.dev_fold==0].reset_index(drop=True); eval_x=full_pilot.loc[pilot.dev_fold==0].reset_index(drop=True)
 eval_rows[["From","To"]].to_csv(a.output_dir/"pilot_eval_request.csv",index=False)
 subprocess.run([str(a.estimate),"-in",str(a.output_dir/"pilot_eval_request.csv"),"-out",str(a.output_dir/"pilot_eval_base.csv"),"--arch",str(a.arch)],check=True)
 baseline=pd.read_csv(a.output_dir/"pilot_eval_base.csv")["delay"].to_numpy(float)
 runs=[]; gate_a1=True; gate_a2=True
 for segment, low, high in (("0-16",0,16),("17-32",17,32)):
  pm=endpoint_dist(public).between(low,high).to_numpy(); train_fold=(pilot.dev_fold!=0).to_numpy(); qm=endpoint_dist(pilot).between(low,high).to_numpy(); em=endpoint_dist(eval_rows).between(low,high).to_numpy()
  tx_base=pd.concat([base_public.loc[pm],base_pilot.loc[train_fold & qm]],ignore_index=True); tx_full=pd.concat([full_public.loc[pm],full_pilot.loc[train_fold & qm]],ignore_index=True)
  y=np.concatenate([public.delay.to_numpy(float)[pm],pilot.delay.to_numpy(float)[train_fold & qm]])
  pilot_weights=1/pilot.loc[train_fold & qm].group_key.map(pilot.loc[train_fold & qm].group_key.value_counts()).to_numpy(float); weights=np.concatenate([np.ones(pm.sum()),pilot_weights])
  categorical=[c for c in tx_full if isinstance(tx_full[c].dtype,pd.CategoricalDtype)]
  for variant, columns in (("A0",list(base_public.columns)),("A1",list(base_public.columns)+NAMES[:17]),("A2",list(base_public.columns)+TWO_HOP_NAMES)):
   if variant=="A2" and not gate_a1: continue
   train=tx_full[columns] if variant!="A0" else tx_base
   cats=[c for c in columns if c in categorical]
   model=lgb.train(PARAMS,lgb.Dataset(train,label=np.log1p(y),weight=weights,categorical_feature=cats,free_raw_data=False),num_boost_round=48)
   pred=np.rint(np.maximum(0,np.expm1(model.predict(eval_x.loc[em,columns])))); truth=eval_rows.delay.to_numpy(float)[em]; base=baseline[em]; groups=eval_rows.group_key.loc[em]
   score=official_metrics(truth,pred); bscore=official_metrics(truth,base); delta=score["official_score"]-bscore["official_score"]; ci=ci_diff(truth,pred,base,groups)
   run={"segment":segment,"variant":variant,"rows":int(em.sum()),"training_rows":int(len(train)),"score":score,"base_score":bscore,"delta":delta,"group_bootstrap_delta_ci95":ci}
   runs.append(run); print(json.dumps(run,sort_keys=True),flush=True)
   if variant=="A1": gate_a1 &= delta >= (3 if segment=="0-16" else 1) and ci[0]>0
   if variant=="A2": gate_a2 &= delta-(next(x["delta"] for x in runs if x["segment"]==segment and x["variant"]=="A1")) >= (1 if segment=="0-16" else .5) and ci[0]>0
 if not gate_a1:
  gate_a2 = False
 report={"protocol":{"public_training_rows":a.public_training_rows,"pilot":"dev only, group fold 0 held out","seed":20260823,"model":"48 trees/depth6/leaves31/log1p L2","sealed_not_read":True,"public_rows_800k_plus_not_read":True},"runs":runs,"a1_gate_pass":gate_a1,"a2_gate_pass":gate_a2,"a2_skipped":not gate_a1,"a3_skipped":True,"stop_reason":None if gate_a1 else "A1 did not meet both segment bootstrap-lower-bound gates; B2K4/B3K2 were not run."}
 (a.output_dir/"results.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__": main()
