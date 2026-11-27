"""Read-only local GUI/API checks. Does not invoke EEG inference or actions."""
import argparse
import json
from urllib import parse,request
p=argparse.ArgumentParser()
p.add_argument("--url",default="http://127.0.0.1:8788")
u=p.parse_args().url.rstrip("/")
split=parse.urlsplit(u)
if split.scheme!="http" or split.hostname not in ("127.0.0.1","localhost") or split.port is None:
    raise SystemExit("Loopback URL only")
opener=request.build_opener(request.ProxyHandler({}))
result={"schema":"neurocore-product-readonly-selfcheck-v1"}
for path in ("/","/api/health","/api/xavier/status","/api/xavier/models","/api/workspace/status"):
    with opener.open(u+path,timeout=12) as resp:
        data=resp.read(1024*1024)
        result[path]={"http":resp.status}
        if path=="/":result[path]["ui_present"]=b"NeuroCore" in data
        else:
            obj=json.loads(data)
            result[path]["fields"]=list(obj)
            if path=="/api/xavier/status":result[path]["model_status"]=obj.get("status")
assert result["/"]["ui_present"]
assert all(val["http"]==200 for key,val in result.items() if key.startswith("/"))
print(json.dumps(result,ensure_ascii=False,indent=2))
