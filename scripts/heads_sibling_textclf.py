"""Leave-one-doc-out TF-IDF classifier: does explanation text (rl/sft/gold) or the readouts identify which head of a KV group wrote it?"""
import pandas as pd, numpy as np, glob, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
R='/home/syvb/l24_docs/hf/l24/std_rl/doc_viewer'
S=pd.concat([pd.read_parquet(d+'/sites.parquet').assign(doc=os.path.basename(d)[:2]) for d in sorted(glob.glob(R+'/0*'))])
S=S[S.pos>0]
def un(x): return x/np.linalg.norm(x,axis=-1,keepdims=True)
# sibling write similarity
cs=[]
for d in sorted(glob.glob(R+'/0*')):
    w=un(np.load(d+'/writes.npz')['v'][1:,16:20].astype(np.float32))
    cs.append(np.einsum('tad,tbd->tab',w,w).mean(0))
print('mean cos between sibling writes at the same site:\n',np.mean(cs,0).round(2))
def clf(groups,col,analyzer='word'):
    accs=[];coefs=None
    sub=S[S['head'].isin(groups)]
    for d in sorted(sub.doc.unique()):
        tr,te=sub[sub.doc!=d],sub[sub.doc==d]
        v=TfidfVectorizer(analyzer=analyzer,ngram_range=(1,2),min_df=3,sublinear_tf=True)
        X=v.fit_transform(tr[col]); m=LogisticRegression(C=3,max_iter=2000).fit(X,tr['head'])
        accs.append((m.predict(v.transform(te[col]))==te['head']).mean())
    return np.mean(accs),accs,v,m
for groups in [[16,17,18,19],[0,1,2,3],[24,25,26,27]]:
    print(f'\nheads {groups} — leave-one-doc-out 4-way accuracy (chance 0.25)')
    for col in ['rl_text','sft_text','gold_text','lens','avg16']:
        a,accs,v,m=clf(groups,col)
        print(f'  {col:<10} {a:.2f}  per doc {np.round(accs,2)}')
a,_,v,m=clf([16,17,18,19],'rl_text')
fn=np.array(v.get_feature_names_out())
for i,h in enumerate(m.classes_):
    print(f'H{h} top RL-text features:',list(fn[np.argsort(-m.coef_[i])[:15]]))
a,_,v,m=clf([16,17,18,19],'gold_text')
fn=np.array(v.get_feature_names_out())
for i,h in enumerate(m.classes_):
    print(f'H{h} top gold features:',list(fn[np.argsort(-m.coef_[i])[:15]]))
