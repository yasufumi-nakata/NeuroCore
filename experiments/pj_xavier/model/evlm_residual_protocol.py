"""EVLM residual diagnostic: deterministic visible-only interpolation, absolute-uV evaluation."""
import copy
import json
from evlm_prepare_residual_candidate import transform

def residual_row(row):
    if row.get("split") not in ("train", "val"):
        raise ValueError("residual protocol denies heldout splits")
    # The existing candidate transform accepts only train. For fixed dev evaluation
    # use the same mathematical mapping on a copy, retaining original split/id.
    z = transform({**row, "split": "train"})
    z["split"] = row["split"]
    return z

def absolute_row(row):
    z = copy.deepcopy(row)
    z["messages"][1]["content"] = json.dumps(row["absolute_target"], separators=(",", ":"))
    z["task_kind"] = "quantized_eeg_reconstruction"
    return z

def absolute_output(output, row):
    try:
        v = json.loads(output.strip())
        b = row["deterministic_baseline"]
        if not isinstance(v, list) or len(v) != len(b) or any(type(x) is not int or not -200 <= x <= 200 for x in v):
            return "INVALID"
        y = [a+c for a,c in zip(v,b)]
        if any(not -100 <= x <= 100 for x in y):
            return "INVALID"
        return json.dumps(y, separators=(",", ":"))
    except (ValueError, TypeError, AttributeError):
        return "INVALID"

def absolute_inputs(outputs, rows):
    if len(outputs) != len(rows) or not rows:
        raise ValueError("unpaired residual evaluation")
    return [absolute_output(o,r) for o,r in zip(outputs,rows)], [absolute_row(r) for r in rows]

def controls(rows):
    # Fixed rounded linear and cubic Hermite, no fitted dev parameters.
    sums = {"rounded_linear": 0., "rounded_hermite": 0.}
    n = 0
    for row in rows:
        rec = row["reconstruction"]; obs = rec["observed"]; ix = rec["masked_indices"]; width = rec["geometry"]["shape"][1]
        l,r = ix[0]-1,ix[-1]+1
        a,b = obs[l],obs[r]; secant = (b-a)/(r-l)
        ml = a-obs[l-1] if l%width>0 else secant
        mr = obs[r+1]-b if r%width<width-1 else secant
        pred=[]
        for i in ix:
            t=(i-l)/(r-l)
            pred.append(round((2*t**3-3*t*t+1)*a+(t**3-2*t*t+t)*(r-l)*ml+(-2*t**3+3*t*t)*b+(t**3-t*t)*(r-l)*mr))
        # Unclipped interpolation controls; report their error as-is.
        for name,vals in (("rounded_linear",row["deterministic_baseline"]),("rounded_hermite",pred)):
            sums[name]+=sum((x-y)**2 for x,y in zip(vals,row["absolute_target"]))
        n+=len(ix)
    return {k+"_baseline_mse_all_targets":v/n for k,v in sums.items()}

def selftest():
    from evlm_prepare_residual_candidate import selftest as old
    assert old()
    r={"split":"train","task_kind":"quantized_eeg_reconstruction","id":"s","reconstruction":{"observed":[1,None,None,None,None,6,7],"masked_indices":[1,2,3,4],"geometry":{"shape":[1,7]}},"messages":[{"role":"user","content":"Return only a JSON integer array in masked-index order."},{"role":"assistant","content":"[2,4,4,5]"}]}
    z=residual_row(r)
    assert absolute_output("[0,1,0,0]",z)=="[2,4,4,5]"
    assert absolute_output("[0,1,0,0]",residual_row({**r,"split":"val"}))=="[2,4,4,5]"
    assert residual_row({**r,"messages":[r["messages"][0],{"role":"assistant","content":"[9,9,9,9]"}]})["messages"][0]==z["messages"][0]
    for out in ("[0,0,0]","[0,true,0,0]","[0,1.0,0,0]","[200,0,0,0]","not JSON"):
        assert absolute_output(out,z)=="INVALID"
    for split in ("test","review"):
        try: residual_row({**r,"split":split})
        except ValueError: pass
        else: raise AssertionError("heldout accepted")
    assert controls([z])["rounded_linear_baseline_mse_all_targets"]==0.25
    print("residual synthetic tests PASS")
if __name__=="__main__": selftest()
