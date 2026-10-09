import re, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
U="/root/.claude/uploads/560f0475-9b0a-5e3c-bad8-d41623133c01/"
V2=open(U+"0e84c320-attachment.txt").read()
V1=open(U+"7a140564-attachment.txt").read()
PRE1=open(U+"454e4fbc-attachment.txt").read()
OUT="/home/claude/readme/figures/"
INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; AXIS="#c3c2b7"; BG="#fcfcfb"
C={"full":"#2a78d6","no_memory":"#eb6834","no_pretrain":"#1baf7a","heuristic":"#898781","expert":"#4a3aa7"}
LBL={"full":"Full (memory 8 + pretraining)","no_memory":"No memory (memory 1)","no_pretrain":"No pretraining (random frozen encoder)"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.edgecolor":AXIS,"axes.labelcolor":INK2,
 "xtick.color":MUTED,"ytick.color":MUTED,"axes.titlecolor":INK,"axes.titlesize":11,"axes.titleweight":"bold",
 "figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,"axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,
 "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"legend.labelcolor":INK2})
EP=re.compile(r"^Episode (\d+) \| Reward\s+(\S+) \| Steps\s+(\d+) \| Success (\w+) \| Collision (\w+) \| Buffer\s+\d+ \| C1\s+(\S+) \| C2\s+(\S+) \| Actor\s+(\S+) \| Alpha\s+(\S+) \| SuccessRate\(50\)\s+(\S+)",re.M)
EV=re.compile(r"Held-out eval @ ep (\d+)\] SuccessRate (\S+) \| MeanReward (\S+)")
def parse(text):
    e=np.array([[float(x) if x not in("True","False") else float(x=="True") for x in m] for m in EP.findall(text)])
    v=np.array([[float(x) for x in m] for m in EV.findall(text)])
    return e,v
runs={}
for part in V2.split("SAC TRAINING (UNITY v2) - run: ")[1:]:
    name=part.split()[0]; runs[name]=parse(part)
easy=parse(V1)
for k,(e,v) in runs.items(): print(k,len(e),len(v))
def smooth(y,w=50):
    y=np.asarray(y,float); out=np.full_like(y,np.nan)
    for i in range(len(y)):
        s=y[max(0,i-w+1):i+1]; s=s[np.isfinite(s)]
        out[i]=s.mean() if len(s) else np.nan
    return out
# ---------------- Figure 1: SAC training curves ----------------
fig,ax=plt.subplots(2,3,figsize=(15,8.2))
panels=[("Training success rate (rolling 50 episodes)",9,None,False,(0,1.02)),
        ("Episode reward (rolling mean, 50 episodes)",1,None,False,None),
        ("Episode length in steps (rolling mean, 50)",2,None,False,None),
        ("Critic loss (mean of Q1, Q2; rolling 50)","critic",None,True,None),
        ("Actor loss (rolling 50)",7,None,False,None),
        ("Entropy temperature α",8,None,True,None)]
for a,(title,col,_,logy,ylim) in zip(ax.flat,panels):
    for k in ["full","no_memory","no_pretrain"]:
        e,_=runs[k]
        y=(e[:,5]+e[:,6])/2 if col=="critic" else e[:,col]
        y=y if col==9 else smooth(y)
        a.plot(e[:,0],y,color=C[k],lw=2,label=LBL[k])
    a.set_title(title,loc="left"); a.set_xlabel("Training episode")
    if logy: a.set_yscale("log")
    if ylim: a.set_ylim(*ylim)
h,l=ax[0,0].get_legend_handles_labels(); fig.legend(h,l,loc="upper right",ncol=3,bbox_to_anchor=(0.995,0.975))
fig.suptitle("SAC training in Unity (harder random layouts)",x=0.01,ha="left",fontsize=14,fontweight="bold",color=INK)
fig.text(0.01,0.935,"No-pretraining run was stopped at episode ~1300 after it collapsed.",color=INK2,fontsize=10)
fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig(OUT+"sac_training_curves.png",dpi=150); plt.close(fig)
# ---------------- Figure 2: held-out evaluation during training ----------------
fig,a=plt.subplots(figsize=(10,4.6))
for k in ["full","no_memory","no_pretrain"]:
    _,v=runs[k]; a.plot(v[:,0],v[:,1],color=C[k],lw=2,marker="o",ms=4,label=LBL[k])
a.set_ylim(-0.03,1.05); a.set_xlabel("Training episode"); a.set_ylabel("Success rate (40 held-out layouts)")
a.set_title("Held-out success rate during training (deterministic policy, every 100 episodes)",loc="left")
a.legend(loc="center right"); fig.tight_layout(); fig.savefig(OUT+"heldout_success_during_training.png",dpi=150); plt.close(fig)
# ---------------- Figure 3: pretraining curves ----------------
PE=re.compile(r"^\s*Epoch\s+(\d+) \| train (\S+) \| val (\S+) \(angle err ~(\S+) deg, dist RMSE ~(\S+) m\)",re.M)
pre2=np.array(PE.findall(V2.split("pretrain_unity_waypoints.py")[1].split("train_sac_unity.py")[0]),float)
pre1=np.array(PE.findall(PRE1),float)
fig,ax=plt.subplots(1,3,figsize=(15,4.4))
for p,name,col in [(pre1,"Easy layouts (v1)","#898781"),(pre2,"Harder layouts (v2, used for final results)","#2a78d6")]:
    ax[0].plot(p[:,0],p[:,1],color=col,lw=2,ls="--",label=f"{name}: train")
    ax[0].plot(p[:,0],p[:,2],color=col,lw=2,label=f"{name}: validation")
    ax[1].plot(p[:,0],p[:,3],color=col,lw=2,label=name)
    ax[2].plot(p[:,0],p[:,4],color=col,lw=2,label=name)
ax[0].set_title("Waypoint loss",loc="left"); ax[1].set_title("Validation angle error (degrees)",loc="left"); ax[2].set_title("Validation distance RMSE (m)",loc="left")
for a in ax: a.set_xlabel("Epoch")
ax[1].set_ylim(0,None); ax[2].set_ylim(0,None)
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=4,fontsize=9); fig.suptitle("Waypoint-prediction pretraining of encoder + Transformer memory",x=0.01,ha="left",fontsize=14,fontweight="bold",color=INK)
fig.tight_layout(rect=(0,0.08,1,0.92)); fig.savefig(OUT+"pretraining_curves.png",dpi=150); plt.close(fig)
# ---------------- Figure 4: final results ----------------
methods=["heuristic","full","no_memory","no_pretrain","expert"]
names=["Goal-angle\nheuristic","Full model","No memory","No pretraining","Map expert\n(upper bound)"]
succ=[0.43,0.98,0.99,0.47,1.00]; spl=[0.429,0.923,0.869,0.416,0.971]; coll=[13.42,1.08,0.45,13.61,0.0]
fig,ax=plt.subplots(1,2,figsize=(14,4.8),gridspec_kw={"width_ratios":[1.6,1]})
x=np.arange(len(methods)); w=0.36
b1=ax[0].bar(x-w/2-0.01,succ,w,color="#2a78d6",label="Success rate")
b2=ax[0].bar(x+w/2+0.01,spl,w,color="#eb6834",label="SPL")
for b,vv in list(zip(b1,succ))+list(zip(b2,spl)): ax[0].text(b.get_x()+b.get_width()/2,vv+0.015,f"{vv:.2f}",ha="center",color=INK2,fontsize=9)
ax[0].set_xticks(x,names); ax[0].set_ylim(0,1.12); ax[0].set_title("Success rate and SPL (100 held-out layouts)",loc="left"); ax[0].legend(loc="upper left",ncol=2)
b=ax[1].bar(x,coll,0.6,color="#e34948")
for bb,vv in zip(b,coll): ax[1].text(bb.get_x()+bb.get_width()/2,vv+0.25,f"{vv:.2f}",ha="center",color=INK2,fontsize=9)
ax[1].set_xticks(x,names,fontsize=8.5); ax[1].set_title("Collision steps per episode (lower is better)",loc="left")
for a in ax: a.grid(axis="x",visible=False)
fig.tight_layout(); fig.savefig(OUT+"unity_results.png",dpi=150); plt.close(fig)
# ---------------- Figure 5: easy env v1 training ----------------
e,v=easy
fig,ax=plt.subplots(1,2,figsize=(13,4.2))
ax[0].plot(e[:,0],e[:,9],color="#2a78d6",lw=2,label="Training success (rolling 50)")
ax[0].plot(v[:,0],v[:,1],color="#eb6834",lw=2,marker="o",ms=4,label="Held-out success (40 layouts)")
ax[0].set_ylim(-0.03,1.05); ax[0].legend(loc="lower right"); ax[0].set_title("Success rate",loc="left")
ax[1].plot(e[:,0],smooth((e[:,5]+e[:,6])/2),color="#2a78d6",lw=2); ax[1].set_yscale("log"); ax[1].set_title("Critic loss (rolling 50)",loc="left")
for a in ax: a.set_xlabel("Training episode")
fig.suptitle("Sanity check: SAC on the easy Unity layout (fixed corners)",x=0.01,ha="left",fontsize=14,fontweight="bold",color=INK)
fig.tight_layout(rect=(0,0,1,0.92)); fig.savefig(OUT+"unity_easy_layout_training.png",dpi=150); plt.close(fig)
print("pre",pre1.shape,pre2.shape)
