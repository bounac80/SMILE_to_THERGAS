
#
# Conversion notation SMILE <-> NLF (Thergas)
#
# Roda - Version du 16 avril 2026
#

# Correction sur les polycyles, les métagroupes
# Ajout des conversions SMILES -> NLF et NLF-> SMILES
#

#!/usr/bin/env python3
"""
Convertisseur bidirectionnel SMILES ↔ NLF (Nancy Linear Form)
=================================================================
Règles :  Côme & Muller (1984) · Blurock et al. (2012) · Rapport LRGP
"""

#
# Bibliothèques python standard
#
import streamlit as st
from streamlit_ketcher import st_ketcher

import pandas as pd
import numpy as np

#
# Graphes
#
import matplotlib
import matplotlib.pyplot as plt #importer la lib graphique 
from matplotlib.pyplot import plot # tracage de courbe / evite d'ecrire plt.plot()
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

#
# Bibliothèques python pour conversion
#
import re
from rdkit import Chem
from rdkit.Chem import RWMol, Atom, BondType
from rdkit.Chem import Draw, AllChem
import sys
from io import BytesIO
from collections import deque

# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#  Configuration de la page
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Convertisseur SMILES ↔ NLF",
    page_icon="🧪",
    #layout="wide",
)

# ── Style CSS personnalisé ──────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-top: 0;
    }
    .result-box {
        background-color: #f0f2f6;
        border-left: 4px solid #1E88E5;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        word-break: break-all;
        margin: 10px 0;
    }
    .nlf-box {
        border-left-color: #7B1FA2;
    }
    .smiles-box {
        border-left-color: #2E7D32;
    }
    .error-box {
        background-color: #fce4ec;
        border-left: 4px solid #c62828;
        padding: 15px;
        border-radius: 0 8px 8px 0;
    }
    .legend-title {
        font-weight: 600;
        color: #1E3A5F;
        margin-bottom: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

st.image('Gemini_Generated_Image_ve3u4zve3u4zve3u.png',width=850)

st.markdown('<h1 class="main-title">🧪 Convertisseur SMILES ↔ NLF (Thergas notation)</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Nancy Linear Form — LRGP, Université de Lorraine<br>'
            '<em>Côme & Muller (1984) · Muller (1987) . Blurock et al. (2012) . etc ...</em></p>', unsafe_allow_html=True)

st.markdown("<hr style='height: 2px; background-color: #333;'>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#  Introduction
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

st.markdown("***Université de Lorraine, CNRS, LRGP, F-54000 Nancy, France***")
st.write("----------------------------------------------------------")
st.write("")
st.markdown('<a href="mailto:roda.bounaceur@univ-lorraine.fr"> If you have any problems please Contact us !</a>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#  Utilitaires
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

# Traduction automatique des groupes fonctionnels
# super groupe, etc...

def _find_meta_groups(mol):
    """
    Détecte tous les groupes fonctionnels multi-atomes → méta-atomes NLF.
    
    Groupes détectés :
      'co'  — carbonyle   C=O  (O terminal, degré 1, 0H)
      #Roda - 'cs'  — thiocarbonyle C=S (S terminal, degré 1, 0H)
      #Roda - 'cn'  — nitrile     C≡N  (N terminal, degré 1, 0H)
      'no2' — nitro       N(-O)₂ (N lié à exactement 2 O terminaux)
      'so2' — sulfonyle   S(=O)₂ (S lié à exactement 2 O terminaux)
      'so'  — sulfinyle   S=O    (S lié à exactement 1 O terminal, pas SO₂)
    
    Retourne :
      meta_map  — dict {atom_idx: "'co'", ...}  (fragment NLF du centre)
      centers   — set des indices d'atomes centres (pour _vnb)
      absorbed  — set des indices d'atomes absorbés (pour _vnb)
    """
    meta_map = {}     # idx → "'co'" / "'no2'" / ...
    centers = set()   # atomes centres de méta-atomes
    absorbed = set()  # atomes absorbés par un méta-atome

    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        idx = a.GetIdx()

        # ── CO : carbonyle C=O ──────────────────────────────────────
        if sym == 'C':
            for b in a.GetBonds():
                o = b.GetOtherAtom(a)
                if (o.GetSymbol() == 'O' and b.GetBondTypeAsDouble() == 2
                        and o.GetDegree() == 1 and o.GetTotalNumHs() == 0):
                    meta_map[idx] = "co"
                    centers.add(idx)
                    absorbed.add(o.GetIdx())

            # ── CS : thiocarbonyle C=S ──────────────────────────────
            #if idx not in centers:  # pas déjà CO
            #    for b in a.GetBonds():
            #        o = b.GetOtherAtom(a)
            #        if (o.GetSymbol() == 'S' and b.GetBondTypeAsDouble() == 2
            #                and o.GetDegree() == 1 and o.GetTotalNumHs() == 0):
            #            meta_map[idx] = "cs" # Roda ---
            #            centers.add(idx)
            #            absorbed.add(o.GetIdx())
#
            # ── CN : nitrile C≡N ────────────────────────────────────
            #if idx not in centers:
            #    for b in a.GetBonds():
            #        o = b.GetOtherAtom(a)
            #        if (o.GetSymbol() == 'N' and b.GetBondTypeAsDouble() == 3
            #                and o.GetDegree() == 1 and o.GetTotalNumHs() == 0):
            #            meta_map[idx] = "'cn'"
            #            centers.add(idx)
            #            absorbed.add(o.GetIdx())
#
        # ── NO2 : nitro N(-O)₂ ──────────────────────────────────────
        elif sym == 'N':
            o_term = [b.GetOtherAtom(a).GetIdx() for b in a.GetBonds()
                      if b.GetOtherAtom(a).GetSymbol() == 'O'
                      and b.GetOtherAtom(a).GetDegree() == 1]
            if len(o_term) == 2:
                meta_map[idx] = "'no2'"
                centers.add(idx)
                for oi in o_term:
                    absorbed.add(oi)

        # ── SO2 / SO : sulfonyle S(=O)₂ ou sulfinyle S=O ───────────
        elif sym == 'S':
            o_term = [b.GetOtherAtom(a).GetIdx() for b in a.GetBonds()
                      if b.GetOtherAtom(a).GetSymbol() == 'O'
                      and b.GetOtherAtom(a).GetDegree() == 1]
            if len(o_term) == 2:
                meta_map[idx] = "'so2'"
                centers.add(idx)
                for oi in o_term:
                    absorbed.add(oi)
            elif len(o_term) == 1:
                meta_map[idx] = "so"
                centers.add(idx)
                absorbed.add(o_term[0])

    return meta_map, centers, absorbed


def _atom_frag(mol, idx, meta_map):
    """Fragment NLF d'un atome. Si l'atome est un centre de méta-atome,
    retourne directement le nom du méta-atome (ex: \"'co'\", \"'no2'\")."""
    if idx in meta_map:
        return meta_map[idx]
    at = mol.GetAtomWithIdx(idx)
    s, h, r = at.GetSymbol(), at.GetTotalNumHs(), at.GetNumRadicalElectrons()
    def _r(f): return f + '(.)' if r else f
    if s == 'F': return _r('f')
    if s == 'I': return _r('i')
    if s == 'Br': return _r("'br'")
    if s == 'Cl': return _r("'cl'")
    if s == 'B': return _r('b')
    if s == 'O': return _r('oh' if h==1 else ('o' if h==0 else f"oh{h}"))
    if s == 'S': return _r("sh" if h==1 else ('s' if h==0 else 's'+'h'*h))
    if s == 'N': return _r("nh2" if h==2 else ('nh' if h==1 else 'n'))
    if s == 'C': return _r({0:'c',1:'ch',2:'ch2',3:'ch3',4:'ch4'}.get(h,f'ch{h}'))
    if s == 'P': return "p"
    f = s.lower() + (('h' if h==1 else f'h{h}') if h>0 else '')
    return _r(f)

def _vnb(mol, idx, cc, co):
    out = []
    for b in mol.GetAtomWithIdx(idx).GetBonds():
        o = b.GetOtherAtomIdx(idx)
        if o in co and idx in cc: continue
        if idx in co: continue
        out.append(o)
    return out


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#  SMILES → NLF
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

def smiles_to_nlf(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: raise ValueError(f"SMILES invalide : {smiles}")
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    n = mol.GetNumAtoms()
    if n == 0: return ""
    Chem.Kekulize(mol, clearAromaticFlags=False)
    meta_map, co_c, co_o = _find_meta_groups(mol)

    ri = mol.GetRingInfo()
    atom_rings = [list(r) for r in ri.AtomRings()]
    bond_rings = [list(r) for r in ri.BondRings()]
    ring_atoms_all = set()
    for r in atom_rings: ring_atoms_all.update(r)

    arom6_bonds = set()
    for idx, ring in enumerate(atom_rings):
        if len(ring) == 6 and all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            for bi in bond_rings[idx]: arom6_bonds.add(bi)

    def bsep(a1, a2):
        b = mol.GetBondBetweenAtoms(a1, a2)
        if not b: return '/'
        if b.GetIdx() in arom6_bonds: return '&'
        t = b.GetBondTypeAsDouble()
        return '//' if t==2 else ('///' if t==3 else '/')

    if not atom_rings:
        nlf = _adfs(mol, next((i for i in range(n) if i not in co_o), 0),
                      set(), co_c, co_o, bsep, meta_map)
        nlf = re.sub(r"o/s\(//o\)\(o\)//o", "'so4'", nlf)
        nlf = re.sub(r"s\(//o\)\(oh\)//o",  "'so3h'", nlf)
        nlf = re.sub(r"s\(//o\)\(o\)//o",   "'so3'", nlf)
        nlf = re.sub(r"s\(//o\)\(//o\)",     "'so2'", nlf)
        nlf = re.sub(r"s\(//o\)",           "so", nlf)
        return nlf

    a2r = {}
    for i, r in enumerate(atom_rings):
        for a in r: a2r.setdefault(a, set()).add(i)
    junctions = {a for a, rs in a2r.items() if len(rs) > 1}

    radj = {}
    for i in range(len(atom_rings)):
        for j in range(i+1, len(atom_rings)):
            if set(atom_rings[i]) & set(atom_rings[j]):
                radj.setdefault(i,[]).append(j); radj.setdefault(j,[]).append(i)

    rv = [False]*len(atom_rings); ring_systems = []
    for i in range(len(atom_rings)):
        if not rv[i]:
            comp=[]; q=deque([i])
            while q:
                r=q.popleft()
                if rv[r]: continue
                rv[r]=True; comp.append(r)
                for nr in radj.get(r,[]):
                    if not rv[nr]: q.append(nr)
            ring_systems.append(comp)

    lc=[0]; labels={}
    def _lbl(a):
        if a not in labels: lc[0]+=1; labels[a]=lc[0]
        return labels[a]

    def _rwalk(ring, start, avoid=None):
        rs=set(ring); adj={a:[] for a in ring}
        for a in ring:
            for b in mol.GetAtomWithIdx(a).GetBonds():
                o=b.GetOtherAtomIdx(a)
                if o in rs: adj[a].append(o)
        order=[start]
        for _ in range(len(ring)-1):
            cur=order[-1]; prev=order[-2] if len(order)>1 else None
            nbs=[nb for nb in adj[cur] if nb not in order and nb!=prev]
            if avoid is not None and len(order)==1 and len(nbs)>1:
                nbs=[nb for nb in nbs if nb!=avoid] or nbs
            if not nbs: break
            order.append(nbs[0])
        return order

    def _path_to_ring(start, from_a):
        q=deque([(start,[start])]); vis={start,from_a}
        while q:
            cur,path=q.popleft()
            for b in mol.GetAtomWithIdx(cur).GetBonds():
                o=b.GetOtherAtomIdx(cur)
                if o in vis or o in co_o: continue
                if o in ring_atoms_all: return path, o
                vis.add(o); q.append((o,path+[o]))
        return None, None

    def _leads_to_ring(nb, fr):
        p,_=_path_to_ring(nb,fr); return p is not None

    described = set()

    def _sub_dfs(root, excl):
        described.add(root); frag=_atom_frag(mol,root,meta_map)
        kids=[nb for nb in _vnb(mol,root,co_c,co_o)
              if nb not in excl and nb not in described
              and nb not in ring_atoms_all and nb not in co_o]
        if not kids: return frag
        main=kids[0]
        for nb in kids[1:]:
            sub=_sub_dfs(nb,excl); sp=bsep(root,nb)
            frag+=f'({sp}{sub})' if sp!='/' else f'({sub})'
        frag+=bsep(root,main)+_sub_dfs(main,excl)
        return frag

    def _isubs(a, rset):
        s=""; deferred=[]
        for nb in _vnb(mol,a,co_c,co_o):
            if nb in rset or nb in described or nb in co_o: continue
            # Si le voisin est dans un cycle, vérifier s'il est déjà traité ou non
            if nb in ring_atoms_all:
                nb_rings = a2r.get(nb, set())
                if all(ri in processed_ri for ri in nb_rings):
                    continue  # cycle déjà traité → ignorer
                else:
                    deferred.append((a, nb))  # cycle pas encore traité → différer
                continue
            if _leads_to_ring(nb,a): deferred.append((a,nb))
            else:
                sub=_sub_dfs(nb,described|ring_atoms_all); sp=bsep(a,nb)
                s+=f'({sp}{sub})' if sp!='/' else f'({sub})'
        return s, deferred

    def _emit(a, rset, dfc):
        described.add(a); frag=_atom_frag(mol,a,meta_map)
        if a in labels: frag+=f'(#{labels[a]})'
        subs,dlist=_isubs(a,rset)
        for att,fnb in dlist:
            if att not in labels: _lbl(att); frag+=f'(#{labels[att]})'
            dfc.append((att,fnb))
        frag+=subs; return frag

    # ── Génération ─────────────────────────────────────────────────
    chains=[]; deferred_chains=[]; processed_ri=set()

    def _proc_fused(rsys_comp):
        rorder=[]; rvis=set(); rq=deque([rsys_comp[0]])
        while rq:
            r=rq.popleft()
            if r in rvis: continue
            rvis.add(r); rorder.append(r)
            for nr in radj.get(r,[]):
                if nr in set(rsys_comp) and nr not in rvis: rq.append(nr)
        processed=set()
        
        # Roda
        # CORRECTIF : inclure les atomes des cycles déjà traités de ce système
        for ri in rorder:
            if ri in processed_ri:
                processed.update(atom_rings[ri])
        
        for rp, ri in enumerate(rorder):
            if ri in processed_ri: continue
            processed_ri.add(ri)
            ring=atom_rings[ri]; rset=set(ring)
            if rp==0:
                start=ring[0]
                for a in ring:
                    if a in junctions: start=a; break
                _lbl(start)
                for a in ring:
                    if a in junctions: _lbl(a)
                order=_rwalk(ring,start); ch=""
                for i,a in enumerate(order):
                    if i>0: ch+=bsep(order[i-1],a)
                    ch+=_emit(a,rset,deferred_chains)
                ch+=bsep(order[-1],order[0])+str(labels[order[0]])
                chains.append(ch); processed.update(order)
            else:
                shared=rset&processed
                for a in shared: _lbl(a)
                sl=sorted(shared,key=lambda a:labels.get(a,9999))
                start=sl[0]; order=_rwalk(ring,start)
                def _fp(od):
                    for i in range(1,len(od)):
                        if od[i] in shared: return od[:i+1]
                    return od
                p1=_fp(order); p2=_fp([order[0]]+list(reversed(order[1:])))
                n1=sum(1 for a in p1 if a not in processed)
                n2=sum(1 for a in p2 if a not in processed)
                path=p1 if n1>=n2 else p2
                ch=str(labels[start])
                for i in range(1,len(path)):
                    a=path[i]; ch+=bsep(path[i-1],a)
                    if a in shared and a==path[-1]: ch+=str(labels[a])
                    else: ch+=_emit(a,rset,deferred_chains)
                chains.append(ch); processed.update(path)

    ring_systems.sort(key=lambda c:-sum(len(atom_rings[r]) for r in c))
    _proc_fused(ring_systems[0])

    while deferred_chains:
        attach,first_nb=deferred_chains.pop(0)
        if first_nb in described: continue
        path_atoms=[]; cur,prev=first_nb,attach
        while cur not in ring_atoms_all:
            path_atoms.append(cur); described.add(cur)
            nbs=[nb for nb in _vnb(mol,cur,co_c,co_o)
                 if nb!=prev and nb not in described and nb not in co_o]
            if not nbs: break
            nxt=nbs[0]
            for nb in nbs:
                if nb in ring_atoms_all or _leads_to_ring(nb,cur): nxt=nb; break
            prev,cur=cur,nxt
        ch=str(labels[attach]); pv=attach
        for a in path_atoms:
            ch+=bsep(pv,a); frag=_atom_frag(mol,a,meta_map)
            for nb in _vnb(mol,a,co_c,co_o):
                if nb!=pv and nb not in described and nb not in ring_atoms_all and nb not in co_o:
                    if not _leads_to_ring(nb,a):
                        sub=_sub_dfs(nb,described|ring_atoms_all); sp=bsep(a,nb)
                        frag+=f'({sp}{sub})' if sp!='/' else f'({sub})'
            ch+=frag; pv=a
        if cur in ring_atoms_all and cur not in described:
            dest_ri=None
            for i,r in enumerate(atom_rings):
                if cur in r and i not in processed_ri: dest_ri=i; break
            if dest_ri is not None:
                ring=atom_rings[dest_ri]; rset=set(ring)
                processed_ri.add(dest_ri); _lbl(cur)
                ch+=bsep(pv,cur); order=_rwalk(ring,cur)
                for i,a in enumerate(order):
                    if i>0: ch+=bsep(order[i-1],a)
                    ch+=_emit(a,rset,deferred_chains)
                ch+=bsep(order[-1],order[0])+str(labels[order[0]])
                
                # Roda
                # CORRECTIF : ajouter CETTE chaîne (qui ouvre les labels #n) 
                # AVANT d'appeler _proc_fused
                chains.append(ch); ch=None
                
                for rsys in ring_systems:
                    if dest_ri in rsys:
                        rem=[r for r in rsys if r not in processed_ri]
                        if rem: _proc_fused(rsys)
                        break
        # CORRECTIF - Roda                
        if ch is not None: chains.append(ch)
        #chains.append(ch)

    for rsys in ring_systems:
        rem=[r for r in rsys if r not in processed_ri]
        if rem: _proc_fused(rsys)

    for a in range(n):
        if a not in described and a not in co_o:
            sub=_sub_dfs(a,described)
            if sub: chains.append(sub)

    nlf = ','.join(chains)
    # Post-traitement regex pour les groupes soufrés complexes
    nlf = re.sub(r"o/s\(//o\)\(o\)//o", "'so4'", nlf)
    nlf = re.sub(r"s\(//o\)\(oh\)//o",  "'so3h'", nlf)
    nlf = re.sub(r"s\(//o\)\(o\)//o",   "'so3'", nlf)
    nlf = re.sub(r"s\(//o\)\(//o\)",     "'so2'", nlf)
    nlf = re.sub(r"s\(//o\)",           "'so'", nlf)
    return nlf

def _adfs(mol, start, excl, cc, co, bsep_fn, meta_map=None):
    desc=set(excl)
    def _d(idx):
        desc.add(idx); f=_atom_frag(mol,idx,meta_map or {})
        ch=[(nb,bsep_fn(idx,nb)) for nb in _vnb(mol,idx,cc,co) if nb not in desc and nb not in co]
        if not ch: return f
        mn,ms=ch[0]
        for nb,sp in ch[1:]:
            sub=_d(nb); f+=f'({sp}{sub})' if sp!='/' else f'({sub})'
        f+=ms+_d(mn); return f
    return _d(start)

# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#  NLF → SMILES  
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════


_BT={'/':BondType.SINGLE,'//':BondType.DOUBLE,'///':BondType.TRIPLE,'&':BondType.AROMATIC}

def _split_chains(nlf):
    chains,d,cur=[],0,""
    for c in nlf:
        if c=='(': d+=1
        elif c==')': d-=1
        if c==',' and d==0: chains.append(cur); cur=""
        else: cur+=c
    if cur: chains.append(cur)
    return chains

# Méta-atomes reconnus SANS apostrophes dans le NLF
_BARE_META = {'co', 'oh', 'f', 'i', 'b', 'p' , 'so' }

def _tok(nlf):
    tokens,i,n=[],0,len(nlf)
    while i<n:
        c=nlf[i]
        if c=="'":
            j=nlf.index("'",i+1); tokens.append(('meta',nlf[i+1:j])); i=j+1
        elif c=='(':
            if i+1<n and nlf[i+1]=='#':
                j=nlf.index(')',i); tokens.append(('ropen',nlf[i+2:j])); i=j+1
            elif i+2<n and nlf[i+1]=='.' and nlf[i+2]==')':
                tokens.append(('rad','.')); i+=3
            else: tokens.append(('bo','(')); i+=1
        elif c==')':
            tokens.append(('bc',')')); i+=1
            if i<n and nlf[i].isdigit():
                j=i
                while j<n and nlf[j].isdigit(): j+=1
                tokens.append(('mult',int(nlf[i:j]))); i=j
        elif c=='/':
            if i+2<n and nlf[i+1]=='/' and nlf[i+2]=='/':
                i+=3
                if i<n and nlf[i].isdigit():
                    j=i
                    while j<n and nlf[j].isdigit(): j+=1
                    tokens.append(('rcl',nlf[i:j],'///')); i=j
                else: tokens.append(('bond','///'))
            elif i+1<n and nlf[i+1]=='/':
                i+=2
                if i<n and nlf[i].isdigit():
                    j=i
                    while j<n and nlf[j].isdigit(): j+=1
                    tokens.append(('rcl',nlf[i:j],'//')); i=j
                else: tokens.append(('bond','//'))
            else:
                i+=1
                if i<n and nlf[i].isdigit():
                    j=i
                    while j<n and nlf[j].isdigit(): j+=1
                    tokens.append(('rcl',nlf[i:j],'/')); i=j
                else: tokens.append(('bond','/'))
        elif c=='&':
            i+=1
            if i<n and nlf[i].isdigit():
                j=i
                while j<n and nlf[j].isdigit(): j+=1
                tokens.append(('rcl',nlf[i:j],'&')); i=j
            else: tokens.append(('bond','&'))
        elif c.isalpha():
            j=i
            while j<n and nlf[j].isalpha(): j+=1
            while j<n and nlf[j].isdigit(): j+=1
            frag=nlf[i:j].lower()
            # Reconnaître les méta-atomes écrits sans apostrophes
            if frag in _BARE_META:
                tokens.append(('meta',frag))
            else:
                tokens.append(('atom',frag))
            i=j
        else: i+=1
    return tokens

def _add_atom(rw,tt,val):
    if tt=='meta':
        nm=val.lower()
        if nm=='co':
            ci=rw.AddAtom(Atom(6)); oi=rw.AddAtom(Atom(8))
            rw.AddBond(ci,oi,BondType.DOUBLE); return ci
        #if nm=='cs':
        #    ci=rw.AddAtom(Atom(6)); si=rw.AddAtom(Atom(16))
        #    rw.AddBond(ci,si,BondType.DOUBLE); return ci
        #if nm=='cn':
        #    ci=rw.AddAtom(Atom(6)); ni=rw.AddAtom(Atom(7))
        #    rw.AddBond(ci,ni,BondType.TRIPLE); return ci
        if nm=='no2':
            ni=rw.AddAtom(Atom(7))
            rw.GetAtomWithIdx(ni).SetFormalCharge(1)
            o1=rw.AddAtom(Atom(8))
            o2=rw.AddAtom(Atom(8)); rw.GetAtomWithIdx(o2).SetFormalCharge(-1)
            rw.AddBond(ni,o1,BondType.DOUBLE)
            rw.AddBond(ni,o2,BondType.SINGLE)
            return ni
        if nm=='so2':
            si=rw.AddAtom(Atom(16))
            o1=rw.AddAtom(Atom(8)); o2=rw.AddAtom(Atom(8))
            rw.AddBond(si,o1,BondType.DOUBLE)
            rw.AddBond(si,o2,BondType.DOUBLE)
            return si
        
        if nm=='so':
            si=rw.AddAtom(Atom(16)); oi=rw.AddAtom(Atom(8))
            rw.AddBond(si,oi,BondType.DOUBLE); return si
        
        el={'oh':8,'cl':17,'br':35,'b':5,'f':9,'i':53,'sh':16,'nh2':7,'p':15}.get(nm)
        
        if el:
            idx=rw.AddAtom(Atom(el))
            return idx
        return rw.AddAtom(Atom(6))
    f=val.lower()
    if f.startswith('c') and not f.startswith('cl'): return rw.AddAtom(Atom(6))
    if f.startswith('o'): return rw.AddAtom(Atom(8))
    if f.startswith('s'): return rw.AddAtom(Atom(16))
    if f.startswith('n'): return rw.AddAtom(Atom(7))
    return rw.AddAtom(Atom(6))

def _link(rw,a1,a2,bt):
    if a1 is None or a2 is None or a1==a2: return
    if rw.GetBondBetweenAtoms(int(a1),int(a2)) is not None: return
    bi=rw.AddBond(int(a1),int(a2),bt)
    if bt==BondType.AROMATIC:
        b=rw.GetBondWithIdx(bi-1); b.SetIsAromatic(True)
        rw.GetAtomWithIdx(int(a1)).SetIsAromatic(True)
        rw.GetAtomWithIdx(int(a2)).SetIsAromatic(True)

def nlf_to_smiles(nlf):
    rw=RWMol(); lab={}
    for ci,raw in enumerate(_split_chains(nlf.strip())):
        raw=raw.strip()
        if not raw: continue
        sl=None; rest=raw
        if ci>0 and raw[0].isdigit():
            j=0
            while j<len(raw) and raw[j].isdigit(): j+=1
            sl=raw[:j]; rest=raw[j:]
        tokens=_tok(rest)
        cur=lab.get(sl) if sl else None
        pbt=BondType.SINGLE; stk=[]
        for tok in tokens:
            tt=tok[0]
            if tt=='bond': pbt=_BT.get(tok[1],BondType.SINGLE); continue
            if tt=='bo': stk.append(cur); continue
            if tt=='bc': cur=stk.pop() if stk else cur; continue
            if tt=='mult': continue
            if tt=='ropen': lab[tok[1]]=cur; continue
            if tt=='rcl':
                tgt=lab.get(tok[1]); bt=_BT.get(tok[2],BondType.SINGLE)
                _link(rw,cur,tgt,bt); continue
            if tt=='rad':
                if cur is not None: rw.GetAtomWithIdx(int(cur)).SetNumRadicalElectrons(1)
                continue
            if tt in ('atom','meta'):
                idx=_add_atom(rw,tt,tok[1])
                if cur is not None: _link(rw,cur,idx,pbt)
                cur=idx; pbt=BondType.SINGLE
    try:
        m=rw.GetMol(); Chem.SanitizeMol(m); return Chem.MolToSmiles(m)
    except:
        try: return Chem.MolToSmiles(rw.GetMol())
        except: return '???'


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#  Début du code Streamlit de conversion
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

#
# Partie Ketcher
#
SMILES_Molecules = 'C1C=CC=C(OC)C=1' # Molécule par défaut

DEFAULT_MOL = SMILES_Molecules
molecule = st.text_input("Insert the SMILE notation of the molecule or draw it", DEFAULT_MOL)
SMILES_Molecules = st_ketcher(molecule)
st.markdown(f"Smile notation: ``{SMILES_Molecules}``")
st.write('')
#
# Partie Conversion
#

#molecule = 'CCCCC'
#molecule = r"C1C(CC([O-])=O)C(C/C=C\CCOS([O-])(=O)=O)C(=O)C1"

nlf = smiles_to_nlf(SMILES_Molecules)

nlf = re.sub(r"o/s\(//o\)\(o\)//o", "'so4'", nlf)
nlf = re.sub(r"s\(//o\)\(oh\)//o", "'so3h'", nlf)
nlf = re.sub(r"s\(//o\)\(o\)//o",  "'so3'", nlf)
nlf = re.sub(r"s\(//o\)\(//o\)",   "'so2'", nlf)

st.markdown(f"Thergas notation: ``{nlf}``")
st.write('')

# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#  Partie pré-enregistrée
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

def draw_molecule(smiles, size=(350, 250)):
    """Génère une image PNG de la molécule à partir du SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf
    
# ── Exemples ────────────────────────────────────────────────────────
EXEMPLES_SMILES = {
    "Butane":            "CCCC",
    "Benzène":           "c1ccccc1",
    "Toluène":           "Cc1ccccc1",
    "Naphtalène":        "c1ccc2ccccc2c1",
    "Aspirine":          "CC(=O)Oc1ccccc1C(=O)O",
    "2-Butanone":        "CC(=O)CC",
    "Acide acétique":    "CC(=O)O",
    "Acétonitrile":      "CC#N",
    "TNT":               "CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
    "Cyclohexane":       "C1CCCCC1",
    "Furane":            "c1ccoc1",
    "DMSO":              "CS(C)=O",
    "1-Propanol":        "CCCO",
    "Propyne":           "C#CC",
    "Acetone":           "CC(=O)C",
    "roda1":             "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)F",
    "roda2":             "C1=CC=CC2=C1S(=O)(=O)NC2(=O)",
    "roda3":             "C/C=C/c1cc(OC)c(c(c1)OC)O",
    "roda4":             "OB(O)C1=CC(Cl)=CN=C1F",
    "roda5":             "CP(C)(=O)CO",
    "roda6":             "COc1ccc(P(Cl)Cl)cc1",
    "roda7":             "N#CNN",
    "roda8":             "c1ccc2cc(Cc3ccsc3)ccc2c1",
    "roda9":             "Cc1cc(Cl)c2cc(I)ccc2c1",  
    "roda radical1":     "C=C[CH]",
    "roda radical 2":    "[CH2]c1ccccc1",
    "roda tricycle":    "C1CCC2CC(C3CCCCC3)CCC2C1",
    "roda decaline":    "C12CCCCC1CCCC2",
    "roda polycyles":    "C1CCC2CC(C3CCC4CCCCC4C3)CCC2C1",
    
    
    
    
}

EXEMPLES_NLF = {
    "Butane":            "ch3/ch2/ch2/ch3",
    "Benzène":           "ch(#1)&ch&ch&ch&ch&ch&1",
    "Naphtalène":        "c(#1)&c(#2)&ch&ch&ch&ch&1,1&ch&ch&ch&ch&2",
    "2-Butanone":        "ch3/co/ch2/ch3",
    "Acide acétique":    "ch3/co/oh",
    "Éthanol":           "ch3/ch2/oh",
    "Acétonitrile":      "ch3/'cn'",
    "TNT":               "c(#1)(ch3)&c('no2')&ch&c('no2')&ch&c('no2')&1",
    "Cyclohexane":       "ch2(#1)/ch2/ch2/ch2/ch2/ch2/1",
    "Bromoéthane":       "ch3/ch2/'br'",
    "Fluoroéthane":      "ch3/ch2/f",
    "Propyne":           "ch3/c///ch",
    "Roda 1":           "ch(#1)/ch2/ch2/ch2/ch2/ch(#2)/1,1/ch2/ch2/ch(#3)/ch2/2,3/ch(#4)/ch2/ch2/ch(#5)/ch(#6)/ch2/4,5/ch2/ch2/ch2/ch2/6",
    "Roda 2":           "ch(#1)/ch2/ch2/ch(#3)/ch2/ch(#2)/1,1/ch2/ch2/ch2/ch2/2,3/ch(#4)/ch2/ch2/ch2/ch2/ch2/4",
    "Roda 3":           "ch(#1)/co/nh/co/n//c(#2)/1,1/nh/nh/nh/2",
    "Roda 4":           "c(#1)(nh/co/ch(nh2)/ch2/ch2/ch2/n//c(nh/'no2')/nh2)&ch&ch&c('no2')&ch&ch&1",
    "Roda 5":           "c(#1)(c///n)&ch&ch&ch&ch&ch&1",
}


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#  Partie 1 - SMILES -> NLF
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

st.markdown("<hr style='height: 2px; background-color: #333;'>", unsafe_allow_html=True)

col_input, col_result = st.columns([1, 1])

with col_input:
    st.subheader("Entrée SMILES")

    # Callback : quand le selectbox change, on met à jour le text_input
    def _on_select_s2n():
        choix = st.session_state.ex_s2n
        if choix != "— Saisie libre —":
            st.session_state.smi_input = EXEMPLES_SMILES[choix]

    choix = st.selectbox(
        "Choisir un exemple :",
        ["— Saisie libre —"] + list(EXEMPLES_SMILES.keys()),
        key="ex_s2n",
        on_change=_on_select_s2n
    )

    smiles_input = st.text_input(
        "Notation SMILES :",
        placeholder="Ex: c1ccccc1  ou  CC(=O)O",
        key="smi_input"
    )

    convert_s2n = st.button("🔄 Convertir en NLF", type="primary", key="btn_s2n")

with col_result:
    st.subheader("Résultat")

    if convert_s2n and smiles_input.strip():
        smi = smiles_input.strip()
        try:
            # Canonisation
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError("SMILES invalide")
            can = Chem.MolToSmiles(mol)

            # Conversion
            nlf_result = smiles_to_nlf(smi)

            st.markdown("**SMILES canonique :**")
            st.markdown(f'<div class="result-box smiles-box">{can}</div>',
                        unsafe_allow_html=True)

            st.markdown("**Notation NLF :**")
            st.markdown(f'<div class="result-box nlf-box">{nlf_result}</div>',
                        unsafe_allow_html=True)

            # Image de la molécule
            img = draw_molecule(can)
            if img:
                st.image(img, caption=f"Structure 2D", width=300)

        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Erreur : {e}</div>',
                        unsafe_allow_html=True)

    elif convert_s2n:
        st.warning("Veuillez entrer une notation SMILES.")


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#  Partie 2 - NLF -> SMILES
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

st.markdown("<hr style='height: 2px; background-color: #333;'>", unsafe_allow_html=True)

col_input2, col_result2 = st.columns([1, 1])

with col_input2:
    st.subheader("Entrée NLF")

    def _on_select_n2s():
        choix2 = st.session_state.ex_n2s
        if choix2 != "— Saisie libre —":
            st.session_state.nlf_input = EXEMPLES_NLF[choix2]

    choix2 = st.selectbox(
        "Choisir un exemple :",
        ["— Saisie libre —"] + list(EXEMPLES_NLF.keys()),
        key="ex_n2s",
        on_change=_on_select_n2s
    )

    nlf_input = st.text_input(
        "Notation NLF :",
        placeholder="Ex: ch3/ch2/ch3  ou  ch(#1)&ch&ch&ch&ch&ch&1",
        key="nlf_input"
    )

    convert_n2s = st.button("🔄 Convertir en SMILES", type="primary", key="btn_n2s")

with col_result2:
    st.subheader("Résultat")

    if convert_n2s and nlf_input.strip():
        nlf = nlf_input.strip()
        try:
            smi_result = nlf_to_smiles(nlf)
            if smi_result == '???':
                raise ValueError("Impossible de construire la molécule à partir de cette notation NLF")

            st.markdown("**SMILES canonique :**")
            st.markdown(f'<div class="result-box smiles-box">{smi_result}</div>',
                        unsafe_allow_html=True)

            # Formule brute
            mol = Chem.MolFromSmiles(smi_result)
            if mol:
                from rdkit.Chem import Descriptors
                formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
                mw = round(Descriptors.ExactMolWt(mol), 2)
                st.markdown(f"**Formule brute :** {formula} &nbsp;—&nbsp; **Masse :** {mw} g/mol")

            # Image
            img = draw_molecule(smi_result)
            if img:
                st.image(img, caption="Structure 2D", width=300)

        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Erreur : {e}</div>',
                        unsafe_allow_html=True)

    elif convert_n2s:
        st.warning("Veuillez entrer une notation NLF.")

# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#  Partie 3 - Aide & règles
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

st.markdown("<hr style='height: 2px; background-color: #333;'>", unsafe_allow_html=True)

st.subheader("Règles de conversion SMILES ↔ NLF")

st.markdown("""
La notation **NLF (Nancy Linear Form)** est une représentation linéaire développée par le LRGP
(Université de Lorraine) pour les logiciels **THERGAS** et **EXGAS** de calcul de propriétés
thermodynamiques par la méthode d'additivité de groupes de Benson.
""")

st.markdown("#### Méta-atomes et autres ....")
st.markdown("""
| Groupe | NLF | Description |
|:---:|:---:|:---|
| OH | `oh` | Alcool |
| C=O | `co` | Carbonyl |
| NO₂ | `'no2'` | Nitro |
| C≡N | `cn` | Nitrile |
| SO | `so` | Sulfinyle |
| SO₂ | `'so2'` | Sulfonyle |
| SO3 | `'so3'` | S2 |
| SO4 | `'so4'` | S3 |
| SO3H | `'so3h'` | S4 |
| C=S | `cs` | Thiocarbonyle |
| F | `f` | Fluor |
| Cl | `'cl'` | Chlore |
| Br | `'br'` | Brome |
| I | `i` | Iode |
| B | `b` | Bore |
| P | `p` | Phosphore |
""")
