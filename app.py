from __future__ import annotations

import struct
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sanger_engine import run_analysis

st.set_page_config(page_title="Sanger AB1 Analysis", page_icon="🧬", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container{max-width:1450px;padding-top:1.2rem;padding-bottom:3rem}
[data-testid="stMetric"]{background:#f5f8fc;border:1px solid #e6ebf2;padding:12px 15px;border-radius:12px}
.sequence{font-family:ui-monospace,monospace;font-size:1.05rem;letter-spacing:.08em;line-height:2.0;word-break:break-all}
.small{color:#69768b;font-size:.88rem}
h2{padding-top:1.2rem;border-top:1px solid #e8edf3}
</style>
""",unsafe_allow_html=True)

BASES="ACGT"
BASE_COLOR={"A":"#16845b","C":"#2773b8","G":"#343a40","T":"#d84b5e","N":"#8993a3","?":"#d39b13","-":"#bdc6d2"}


def q_conf(q):
    return 100*(1-10**(-int(q)/10))


def sequence_display(seq):
    return "".join("?" if b=="N" else b for b in str(seq))


def show_sequence(seq):
    spans=[]
    for i,b in enumerate(str(seq),1):
        shown="?" if b=="N" else b
        color=BASE_COLOR.get(shown,"#18253b")
        spans.append(f'<span title="position {i}" style="color:{color};font-weight:750">{shown}</span>')
        if i%10==0: spans.append("&nbsp;&nbsp;")
    st.markdown('<div class="sequence">'+"".join(spans)+'</div>',unsafe_allow_html=True)


def read_base_table(read):
    rows=[]
    for pos,(base,q,peak) in enumerate(zip(read.seq,read.q,read.peaks),1):
        sig=[max(0,float(ch[int(peak)])) for ch in read.trace]
        total=sum(sig)
        rows.append({
            "Position":pos,"Call":"?" if base=="N" else base,"Q":int(q),
            "Q confidence (%)":round(q_conf(q),2),"Peak sample":int(peak),
            "Peak channel":BASES[int(np.argmax(sig))] if sig else "—",
            "Call channel share (%)":round(100*sig[BASES.index(base)]/total,2) if total and base in BASES else None,
            "A signal":round(sig[0],1),"C signal":round(sig[1],1),"G signal":round(sig[2],1),"T signal":round(sig[3],1),
        })
    return pd.DataFrame(rows)


def trace_figure(read):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.76,.24],vertical_spacing=.05)
    palette={"A":"#16845b","C":"#2773b8","G":"#343a40","T":"#d84b5e"}
    xs=np.arange(read.trace.shape[1])
    labels=np.array(["?" if b=="N" else b for b in read.seq])
    for b,channel in zip(BASES,read.trace):
        fig.add_trace(go.Scattergl(x=xs,y=channel,mode="lines",name=b,line=dict(color=palette[b],width=1.2),hovertemplate=f"{b} signal: %{{y:.0f}}<extra></extra>"),row=1,col=1)
    fig.add_trace(go.Bar(x=read.peaks,y=read.q,name="Q",customdata=labels,marker=dict(color=read.q,colorscale="Blues",showscale=False),hovertemplate="sample %{x}<br>base %{customdata}<br>Q %{y}<extra></extra>"),row=2,col=1)
    fig.update_layout(height=360,margin=dict(l=8,r=8,t=8,b=8),legend=dict(orientation="h",y=1.08),hovermode="x unified",plot_bgcolor="white",paper_bgcolor="white")
    fig.update_yaxes(title="signal",row=1,col=1,gridcolor="#edf1f6")
    fig.update_yaxes(title="Q",row=2,col=1,gridcolor="#edf1f6")
    fig.update_xaxes(title="trace sample",row=2,col=1,rangeslider=dict(visible=True,thickness=.07),gridcolor="#edf1f6")
    return fig


def alignment_figure(alignment,reads):
    cols=alignment["cols"]
    letters=[]
    for j,read in enumerate(reads):
        letters.append([("?" if read.seq[pair[j]]=="N" else read.seq[pair[j]]) if pair[j] is not None else "-" for pair in cols])
    status=[]
    for a,b in zip(*letters):
        if a=="?" or b=="?": status.append(3)
        elif a=="-" or b=="-": status.append(1)
        elif a==b: status.append(0)
        else: status.append(2)
    n=len(cols)
    fig=go.Figure(go.Heatmap(
        x=list(range(1,n+1)),y=["OF","OR_rc"],z=[status,status],text=letters,texttemplate="%{text}",
        textfont=dict(family="ui-monospace,monospace",size=13,color="#17243a"),
        colorscale=[[0,"#d9f0e3"],[.24,"#d9f0e3"],[.25,"#e4edf9"],[.49,"#e4edf9"],[.5,"#f8d9dc"],[.74,"#f8d9dc"],[.75,"#ffedc2"],[1,"#ffedc2"]],
        zmin=0,zmax=3,showscale=False,xgap=2,ygap=5,hovertemplate="alignment column %{x}<br>%{y}: %{text}<extra></extra>"))
    fig.update_layout(height=155,margin=dict(l=10,r=10,t=8,b=8),plot_bgcolor="white",paper_bgcolor="white")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="5′→3′ alignment column",dtick=max(1,n//30))
    return fig


def method_code(row):
    src=str(row.get("sources","")); cid=str(row.get("candidate_id",""))
    if cid.startswith("M") or "Combined N" in src: return "D"
    if cid.startswith("R") or "Peak rescue" in src: return "C"
    if "Phred + peak" in src: return "B"
    return "A"


def candidate_positions(row,env):
    seq=str(row.sequence_5to3)
    geom=next((a for a in env["alignments"] if a["alignment_id"]==row.alignment_id),env["alignments"][0])
    cols=list(geom["cols"])
    events=env.get("df_peak_events",pd.DataFrame())
    event_value=row.get("rescue_event_ids","") if hasattr(row,"get") else getattr(row,"rescue_event_ids","")
    event_ids=str(event_value or "").split(" | ")
    extra={}
    if not events.empty and "event_id" in events:
        for e in events.loc[events.event_id.isin(event_ids)&events.event_type.eq("insertion")].to_dict("records"):
            extra.setdefault(int(e.get("insert_after_seed_nt",0)),[]).append(e)
    expanded=[]
    for i,col in enumerate(cols,1):
        expanded.append((col,None))
        expanded.extend([((None,None),e) for e in extra.get(i,[])])
    if len(expanded)!=len(seq):
        expanded=[(c,None) for c in cols[:len(seq)]]+[((None,None),None)]*max(0,len(seq)-len(cols))
    return expanded


def candidate_detail(row,env):
    seq=str(row.sequence_5to3); reads=env["reads"]; expanded=candidate_positions(row,env)
    weights={"Q confidence":.45,"Peak share":.35,"Read agreement":.20}
    rows=[]; site_scores=[]; parts={k:[] for k in weights}; evidence_sites=0
    for pos,(base,(col,insert_event)) in enumerate(zip(seq,expanded),1):
        if base=="N":
            rows.append({"Position":pos,"Base":"?","OF":"—","OF Q":None,"OR_rc":"—","OR_rc Q":None,"Q confidence":None,"Peak share":None,"Read agreement":None,"Site score":None,"Decision":"Unresolved"})
            continue
        calls=[]; q_values=[]; peak_values=[]
        for j,idx in enumerate(col):
            if idx is None: continue
            r=reads[j]; call=r.seq[idx]; q=int(r.q[idx]); calls.append(call)
            if call==base: q_values.append(q_conf(q))
            peak=int(r.peaks[idx]); sig=[max(0,float(ch[peak])) for ch in r.trace]; total=sum(sig)
            if total: peak_values.append(100*sig[BASES.index(base)]/total)
        qv=float(np.mean(q_values)) if q_values else None
        pv=float(np.mean(peak_values)) if peak_values else None
        if insert_event is not None: pv=max(pv or 0,100*float(insert_event.get("peak_evidence",0)))
        if len(calls)==2: av=100.0 if calls[0]==base and calls[1]==base else 50.0 if base in calls else 0.0
        else: av=None
        available=[]
        for label,val in (("Q confidence",qv),("Peak share",pv),("Read agreement",av)):
            if val is not None:
                available.append((weights[label],val)); parts[label].append(val)
        sv=sum(w*v for w,v in available)/sum(w for w,_ in available) if available else 0.0
        if available: evidence_sites+=1; site_scores.append(sv)
        oi,ri=col
        def basecall(read,idx):
            if idx is None:return "—"
            return "?" if read.seq[idx]=="N" else read.seq[idx]
        if len(calls)==2 and calls[0]==base and calls[1]==base: decision="OF + OR_rc agree"
        elif len(calls)==2: decision="Reads differ"
        elif len(calls)==1: decision="Single-read support"
        elif insert_event is not None: decision="Trace insertion"
        else: decision="No direct call"
        rows.append({"Position":pos,"Base":"?" if base=="N" else base,"OF":basecall(reads[0],oi),"OF Q":int(reads[0].q[oi]) if oi is not None else None,
                     "OR_rc":basecall(reads[1],ri),"OR_rc Q":int(reads[1].q[ri]) if ri is not None else None,
                     "Q confidence":round(qv,1) if qv is not None else None,"Peak share":round(pv,1) if pv is not None else None,
                     "Read agreement":round(av,1) if av is not None else None,"Site score":round(sv,1) if available else None,"Decision":decision})
    length=max(1,len(seq)); index=sum(site_scores)/length if site_scores else 0.0
    coverage=100*evidence_sites/length
    summary={k:(float(np.mean(v)) if v else 0.0) for k,v in parts.items()}
    return index,coverage,summary,pd.DataFrame(rows)


def build_candidate_scores(env):
    all_df=env["df_all_candidates"].copy()
    if all_df.empty:return all_df
    rows=[]
    for cand in all_df.itertuples(index=False):
        index,coverage,summary,_=candidate_detail(cand,env)
        rows.append({"candidate_id":cand.candidate_id,"Method":method_code(cand._asdict()),"AB1 index":round(index,2),"Evidence coverage (%)":round(coverage,1),
                     "Mean Q support (%)":round(summary["Q confidence"],1),"Mean peak share (%)":round(summary["Peak share"],1),"Mean read agreement (%)":round(summary["Read agreement"],1),"Unresolved positions":str(cand.sequence_5to3).count("N")})
    return all_df.merge(pd.DataFrame(rows),on="candidate_id",how="left",validate="one_to_one")


def consensus_audit(candidate,env):
    alignment=next((a for a in env["alignments"] if a["alignment_id"]==candidate.alignment_id),env["alignments"][0])
    reads=env["reads"]; seq=str(candidate.sequence_5to3); cols=alignment["cols"]
    mats=env["probabilities"]["phred"]
    rows=[]
    for p,col in enumerate(cols):
        base=seq[p] if p<len(seq) else "N"
        calls=[]; evidence=[]; qvals=[]
        for j,idx in enumerate(col):
            if idx is not None and reads[j].seq[idx] in BASES:
                calls.append(reads[j].seq[idx]); evidence.append(mats[j][idx]); qvals.append((j,int(reads[j].q[idx])))
            else: qvals.append((j,None))
        if evidence:
            post=np.prod(evidence,axis=0); post=post/post.sum()
            confidence=100*post[BASES.index(base)] if base in BASES else None
        else: confidence=None
        if base=="N": decision="No confident base → ?"
        elif len(calls)==2 and calls[0]==calls[1]==base: decision="Both reads agree"
        elif len(calls)==2: decision="Combined Phred likelihood"
        elif len(calls)==1: decision=f"Single read: {reads[next(j for j,i in enumerate(col) if i is not None)].role}"
        else: decision="No base call"
        def call(j):
            idx=col[j]
            if idx is None:return "—"
            b=reads[j].seq[idx]
            return "?" if b=="N" else b
        def qual(j):
            idx=col[j]
            return int(reads[j].q[idx]) if idx is not None else None
        rows.append({"Position":p+1,"OF":call(0),"OF Q":qual(0),"OR_rc":call(1),"OR_rc Q":qual(1),"C001 base":"?" if base=="N" else base,"Phred likelihood (%)":round(confidence,2) if confidence is not None else None,"Selection":decision})
    return pd.DataFrame(rows)


def render_input():
    st.title("Sanger AB1 analysis")
    st.caption("Upload OF and OR. Results appear below in order.")
    c1,c2=st.columns(2)
    of_file=c1.file_uploader("OF · forward AB1",type=["ab1","abi"],key="of")
    or_file=c2.file_uploader("OR · reverse AB1",type=["ab1","abi"],key="or")
    with st.expander("Primer settings",expanded=False):
        p1,p2=st.columns(2)
        primer5=p1.text_input("5′ primer","ACACGACGCTCTTCCGATCT")
        primer3=p2.text_input("3′ primer","AGATCGGAAGAGCACACGTCT")
    if st.button("Analyze reads",type="primary",disabled=of_file is None or or_file is None):
        for key in ["env","candidate_scores","ref_ready","reference_result"]: st.session_state.pop(key,None)
        try:
            with st.spinner("Reading AB1 and building candidates…"):
                env=run_analysis(of_file.getvalue(),or_file.getvalue(),primer5=primer5,primer3=primer3)
            st.session_state["env"]=env
            st.session_state["names"]=(of_file.name,or_file.name)
            st.session_state["raw_bytes"]=(of_file.getvalue(),or_file.getvalue())
            st.session_state["candidate_scores"]=build_candidate_scores(env)
            st.success("Analysis ready.")
        except Exception as e: st.error(f"Could not read the AB1 files: {e}")


def render_read_and_alignment(env):
    st.header("1 · Reads and full-strand alignment")
    reads=env["reads"]
    c1,c2=st.columns(2)
    for c,read,label in ((c1,reads[0],"OF · forward"),(c2,reads[1],"OR_rc · reverse-complement")):
        with c:
            st.markdown(f"**{label}** · {len(read.seq)} calls · mean Q {np.mean(read.q):.1f}")
            show_sequence(read.seq)
            st.plotly_chart(trace_figure(read),use_container_width=True,config={"displaylogo":False,"scrollZoom":True})
            with st.expander(f"{label} · per-base data"):
                st.dataframe(read_base_table(read),use_container_width=True,hide_index=True,height=300)
    raw_or=env["read_ab1"](st.session_state["raw_bytes"][1],"OR",reverse=False)
    with st.expander("OR file orientation"):
        st.markdown(f"**OR · original direction** · {len(raw_or.seq)} calls · mean Q {np.mean(raw_or.q):.1f}")
        show_sequence(raw_or.seq)
        st.plotly_chart(trace_figure(raw_or),use_container_width=True,config={"displaylogo":False,"scrollZoom":True})
    if not env["df_reads"].empty:
        st.dataframe(env["df_reads"].drop(columns=[x for x in ["raw_Q20","raw_N"] if x in env["df_reads"]]),use_container_width=True,hide_index=True)
    best=env["alignments"][0]
    m=best["metrics"]
    st.markdown(f"**Chosen order:** {best['order']}　·　**Overlap:** {int(m.get('overlap_nt',0))} nt　·　**Match:** {int(m.get('overlap_matches',0))}　·　**Difference:** {int(m.get('overlap_mismatches',0))}　·　**Gap:** {int(m.get('overlap_gap_nt',0))}")
    st.plotly_chart(alignment_figure(best,reads),use_container_width=True,config={"displaylogo":False})
    st.markdown("🟩 agree　　🟥 differ　　🟦 one-read coverage　　🟨 ambiguous call")
    if not env["df_alignments"].empty:
        st.dataframe(env["df_alignments"][[c for c in ["alignment_id","order","overlap_nt","overlap_matches","overlap_mismatches","overlap_gap_nt","overlap_identity_pct"] if c in env["df_alignments"]]],use_container_width=True,hide_index=True)


def candidate_delta(row,baseline):
    for key in ("rescue_edits","combination_edits"):
        value=row.get(key,None)
        if value is not None and pd.notna(value) and str(value).strip():
            return str(value).replace("N>","?>")
    a=str(baseline.sequence_5to3); b=str(row.sequence_5to3)
    if len(a)!=len(b): return f"length {len(a)}→{len(b)} nt"
    changes=[f"{i}:{'?' if x=='N' else x}>{y}" for i,(x,y) in enumerate(zip(a,b),1) if x!=y]
    return ", ".join(changes) if changes else "same as A1"


def method_section(code,title,short,scored,baseline):
    rows=scored.loc[scored.Method.eq(code)].sort_values(["AB1 index","candidate_id"],ascending=[False,True]).copy()
    st.markdown(f"#### {code} · {title}")
    st.caption(short)
    if rows.empty:
        st.write("No candidates.")
        return
    rows["Sequence"] = rows["sequence_5to3"].map(sequence_display)
    rows["Change vs consensus"] = rows.apply(lambda r:candidate_delta(r,baseline),axis=1)
    cols=["candidate_id","Sequence","length_nt","Change vs consensus","AB1 index","Evidence coverage (%)","Mean Q support (%)","Mean peak share (%)","Mean read agreement (%)"]
    st.dataframe(rows[[c for c in cols if c in rows]],use_container_width=True,hide_index=True,height=min(420,100+35*len(rows)))


def render_candidates(env):
    st.header("2 · Candidate methods A–D")
    st.caption("Same AB1 Evidence Index for every method; method inputs and edits are shown in each group.")
    scored=st.session_state["candidate_scores"]
    if scored.empty or env["df_all_candidates"].empty:
        st.warning("No usable candidates were generated. Check the read quality and primer settings.")
        return
    final=env["df_final"]
    baseline_rows=final.loc[final.role.eq("Phred consensus")] if not final.empty and "role" in final else pd.DataFrame()
    if baseline_rows.empty:
        base_row=env["df_all_candidates"].iloc[0]
    else: base_row=baseline_rows.iloc[0]
    st.markdown(f"### {base_row.candidate_id} · Method A · Consensus")
    show_sequence(base_row.sequence_5to3)
    idx,coverage,summary,audit=candidate_detail(base_row,env)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("AB1 index",f"{idx:.1f}/100")
    c2.metric("Evidence coverage",f"{coverage:.1f}%")
    c3.metric("Q contribution",f"{summary['Q confidence']:.1f}")
    c4.metric("Peak / agreement",f"{summary['Peak share']:.1f} / {summary['Read agreement']:.1f}")
    st.markdown(f"**How {base_row.candidate_id} was built**")
    st.caption("At each alignment column, AB1 Phred likelihoods from the covered read(s) are multiplied and normalized. The base with the highest likelihood among called bases is selected; a low-quality unsupported column remains ?.")
    with st.expander(f"Every nucleotide decision for {base_row.candidate_id}",expanded=True):
        st.dataframe(audit,use_container_width=True,hide_index=True,height=460)
    st.markdown("**AB1 Evidence Index**")
    st.caption("Per-position score = 45% Q-derived confidence + 35% selected-channel share + 20% read agreement. If a component is unavailable, remaining weights are rescaled. The index is the mean of position scores; positions without evidence count as zero. It ranks evidence and is not whole-sequence accuracy.")
    st.markdown("#### Method groups")
    method_section("A","Phred consensus","Joint likelihood from base calls and Phred values; this is the baseline.",scored,base_row)
    method_section("B","Peak alternatives","Retains low-Q alternatives when a competing peak reaches 60% of the called-base peak.",scored,base_row)
    method_section("C","Peak / indel rescue","Searches single-read regions for a strong peak inside a wide interval between calls.",scored,base_row)
    method_section("D","Combined hypotheses","Combines supported per-site alternatives; each candidate has its own index.",scored,base_row)
    st.markdown("#### Overall AB1 ranking")
    ranked=scored.sort_values(["AB1 index","Evidence coverage (%)","candidate_id"],ascending=[False,False,True]).reset_index(drop=True)
    ranked.insert(0,"Rank",np.arange(1,len(ranked)+1))
    ranked["Sequence"]=ranked.sequence_5to3.map(sequence_display)
    st.dataframe(ranked[[c for c in ["Rank","candidate_id","Method","Sequence","AB1 index","Evidence coverage (%)","Mean Q support (%)","Mean peak share (%)","Mean read agreement (%)","Unresolved positions"] if c in ranked]],use_container_width=True,hide_index=True,height=420)
    export=ranked.drop(columns=[c for c in ranked.columns if c.lower().startswith("q20") or c=="sequence_5to3"],errors="ignore")
    st.download_button("Download candidate scores",export.to_csv(index=False).encode("utf-8-sig"),"sanger_candidates_AB1_ranked.csv","text/csv")


def render_reference(env):
    st.header("3 · Reference evaluation")
    st.caption("Reference rank is separate; it does not change AB1 scores or candidate sequences.")
    ref_file=st.file_uploader("Reference FASTA/TXT · optional",type=["fa","fasta","fna","txt"],key="reference_file")
    text=st.text_area("Or paste full reference · 5′→3′",height=90,key="reference_text")
    reference=text.strip() or (ref_file.getvalue().decode("utf-8",errors="replace") if ref_file is not None else "")
    target_only=st.checkbox("Input contains only the target between primers",value=False)
    with st.expander("Manual target coordinates"):
        manual=st.checkbox("Use coordinates in full reference",value=False)
        a,b=st.columns(2)
        start=a.number_input("Start · 1-based",min_value=1,value=1)
        end=b.number_input("End · inclusive",min_value=1,value=1)
    mode="manual_coordinates" if manual else "already_between_primers" if target_only else "auto_primers"
    if st.button("Compare all candidates",type="primary",disabled=not reference):
        try:
            full,lo,hi=env["reference_region"](reference,mode,int(start),int(end),env["ab1_run_primers"])
            all_ref=env["add_reference_scores"](env["df_all_candidates"],full[lo:hi],lo)
            scores=st.session_state["candidate_scores"]
            result=scores.merge(all_ref[["candidate_id","ref_rank","ref_match_nt","ref_match_pct_full_region","ref_paired_coverage_pct","ref_called_coverage_pct","ref_sub_nt","ref_ins_nt","ref_del_nt","ref_N_paired_nt","ref_uncovered_end_nt","ref_alignment_tied","ref_sub_positions","ref_ins_positions","ref_del_positions","ref_N_positions"]],on="candidate_id",how="left",validate="one_to_one")
            st.session_state["reference_result"]=(result,full,lo,hi,all_ref)
        except Exception as e: st.error(f"Reference comparison failed: {e}")
    if "reference_result" in st.session_state:
        result,full,lo,hi,all_ref=st.session_state["reference_result"]
        ranked=result.sort_values(["ref_rank","AB1 index"],ascending=[True,False]).reset_index(drop=True)
        ranked.insert(0,"Reference order",np.arange(1,len(ranked)+1))
        st.markdown(f"**Target:** {hi-lo} nt · reference coordinates {lo+1}–{hi}")
        ranked["Sequence"]=ranked.sequence_5to3.map(sequence_display)
        cols=["Reference order","candidate_id","Method","Sequence","AB1 index","ref_rank","ref_match_nt","ref_match_pct_full_region","ref_paired_coverage_pct","ref_called_coverage_pct","ref_sub_nt","ref_ins_nt","ref_del_nt","ref_alignment_tied"]
        view=ranked[cols].rename(columns={"ref_rank":"Reference rank","ref_match_nt":"Matches","ref_match_pct_full_region":"Match (%)","ref_paired_coverage_pct":"Paired coverage (%)","ref_called_coverage_pct":"Called coverage (%)","ref_sub_nt":"Substitutions","ref_ins_nt":"Insertions","ref_del_nt":"Deletions","ref_alignment_tied":"Tie"})
        st.dataframe(view,use_container_width=True,hide_index=True,height=400)
        st.download_button("Download reference ranking",ranked.drop(columns=[c for c in ranked if c.lower().startswith("q20") or c=="sequence_5to3"],errors="ignore").to_csv(index=False).encode("utf-8-sig"),"sanger_candidates_with_reference.csv","text/csv")
        best=ranked.iloc[0]
        st.markdown(f"**Best match · {best.candidate_id}**")
        c1,c2,c3,c4=st.columns(4)
        c1.write("Substitutions: "+(best.ref_sub_positions or "None"))
        c2.write("Insertions: "+(best.ref_ins_positions or "None"))
        c3.write("Deletions: "+(best.ref_del_positions or "None"))
        c4.write("Alignment tie: "+("Yes" if best.ref_alignment_tied else "No"))


render_input()
env=st.session_state.get("env")
if env is not None:
    st.divider()
    render_read_and_alignment(env)
    st.divider()
    render_candidates(env)
    st.divider()
    render_reference(env)
