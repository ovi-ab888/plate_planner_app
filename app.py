# app.py (Lightweight deploy-safe version)
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
from io import BytesIO
from collections import Counter
from math import ceil
import string

st.set_page_config(page_title="Pre-Press Auto Planner", page_icon="🖨️", layout="wide")

# ---------- Helper ----------
def plate_name(n):
    n -= 1
    out = ""
    chars = string.ascii_uppercase
    while True:
        out = chars[n % 26] + out
        n = n // 26 - 1
        if n < 0:
            break
    return out

def auto_plan(demand, cap, max_plates=20):
    remain = demand.copy()
    plates, safe = [], 1000
    while any(v > 0 for v in remain.values()) and len(plates) < max_plates and safe > 0:
        safe -= 1
        total = sum(remain.values())
        if total == 0: break
        ratio = {k: max(1, int(remain[k]*cap/total)) for k in remain if remain[k] > 0}
        used = sum(ratio.values())
        if used > cap:
            over = used - cap
            for k in sorted(ratio, key=ratio.get):
                take = min(ratio[k], over)
                ratio[k] -= take
                over -= take
                if over <= 0: break
        layout = {k:v for k,v in ratio.items() if v>0}
        if not layout: break
        limit = min([ceil(remain[k]/v) for k,v in layout.items()])
        for k,v in layout.items():
            remain[k] = max(0, remain[k]-v*limit)
        plates.append({"name": plate_name(len(plates)+1),"layout":layout,"sheets":limit})
    prod = Counter()
    for p in plates:
        for k,v in p["layout"].items():
            prod[k]+=v*p["sheets"]
    return plates, dict(prod)

# ---------- UI ----------
st.title("🖨️ Auto Multi-Plate Planner (Fast Build)")
col1,col2,col3=st.columns(3)
n=col1.number_input("কতটি Tag",1,50,6)
cap=col2.number_input("Plate capacity",1,64,12)
maxp=col3.number_input("সর্বোচ্চ Plate",1,200,20)

st.markdown("---")
st.subheader("📦 Tag QTY দিন")
l,r=st.columns(2)
tags,qty=[],[]
for i in range(n):
    name=l.text_input(f"Tag {i+1}",f"Tag {i+1}",key=f"t{i}")
    q=r.number_input(f"{name} Qty",0,step=10,key=f"q{i}")
    tags.append(name); qty.append(q)
demand={t:int(q) for t,q in zip(tags,qty) if q>0}

if st.button("🚀 Generate Plan"):
    if not demand: st.error("Tag QTY দিন"); st.stop()
    progress=st.progress(0)
    plates,prod=auto_plan(demand,cap,maxp)
    progress.progress(100)
    if not plates: st.warning("পরিকল্পনা তৈরি হয়নি"); st.stop()
    cols=["Plate"]+list(demand.keys())+["Sheets"]
    rows=[]
    for p in plates:
        row={"Plate":p["name"],"Sheets":p["sheets"]}
        for t in demand.keys(): row[t]=p["layout"].get(t,0)
        rows.append(row)
    df=pd.DataFrame(rows,columns=cols)
    st.dataframe(df,use_container_width=True)
    total=sum(p["sheets"] for p in plates)
    st.success(f"✅ মোট শিট: {total}")
    bio=BytesIO()
    with pd.ExcelWriter(bio,engine="openpyxl") as w:
        df.to_excel(w,sheet_name="Plates",index=False)
        pd.DataFrame([{"Tag":k,"Demand":demand[k],"Produced":prod.get(k,0)} for k in demand])\
            .to_excel(w,sheet_name="Summary",index=False)
    bio.seek(0)
    st.download_button("⬇️ Excel",data=bio,file_name="plan.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
