"""Run the five W3 preparation checks; this is not W3 full-model acceptance."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from probes.w3_common import new_output_dir, source_evidence, sha256_file, write_reports
from probes.w3_qwen3_full.resources import spawn_owned, terminate_process_tree, wait_bounded

GATES = ("medium","subgraphs","attention","full-preflight","layer-diff")
IDENTITIES = {"medium":"w3-medium-host","subgraphs":"w3_qwen3_subgraphs",
              "attention":"unit:attention-backend","full-preflight":"preflight:w3-full-assets",
              "layer-diff":"w3_layer_diff"}
LEVELS = ("none", "basic", "all")
GRAPH_KINDS = ("normal", "diagnostic")
MEDIUM_CASES = (("full_seed_0",256),("full_seed_42",256),("one_token",1),
                ("short_17",17),("short_255",255),("changed_future",256),("changed_padding",17))
MEDIUM_CHECKPOINTS = ("token_embedding",) + tuple(
    f"layer_{layer}.{name}" for layer in range(6) for name in
    ("input_norm","q_norm","k_norm","v_proj","rope_q","rope_k","attn_probs","attn_context",
     "attn_output","attn_residual","post_attention_norm","mlp","residual")) + ("final_norm","logits")
SUBGRAPH_CASES = ("projection_q","projection_k","projection_v","projection_o","rmsnorm_hidden",
                  "rmsnorm_q","rope_q","rmsnorm_k","rope_k","gqa_full","gqa_padding",
                  "gqa_changed_padding","gqa_changed_future","swiglu")
ATTENTION_CASES = (("full17",17),("padding17",5),("future17",17),
                   ("padding_changed17",5),("single17",1),("boundary256",255))
MEMORY_FIELDS = ("input_logical_bytes","initializer_logical_bytes","peak_live_logical_bytes",
                 "peak_numpy_storage_bytes","retained_values","retained_logical_bytes",
                 "retained_numpy_storage_bytes","return_logical_bytes")

def require(condition, message):
    if not condition:
        raise ValueError(message)

def check_numeric(value, atol):
    if isinstance(value,dict):
        if "passed" in value:
            require(value["passed"] is True, "A nested required comparison failed")
        for key,item in value.items():
            if key in ("max_abs","max_abs_error") and item is not None:
                require(type(item) in (int,float) and math.isfinite(item) and 0 <= item < atol,
                        f"Numeric error exceeds {atol}")
            check_numeric(item,atol)
    elif isinstance(value,list):
        for item in value:
            check_numeric(item,atol)
    elif isinstance(value,float):
        require(math.isfinite(value),"Nonfinite number in evidence")


def checked_keys(value, keys, label):
    require(isinstance(value,dict) and set(value)==set(keys),f"Missing/extra {label}")
    return value


def checked_rows(rows, identities, fields, label):
    require(isinstance(rows,list) and all(isinstance(row,dict) for row in rows),f"Missing {label}")
    actual = [tuple(row.get(field) for field in fields) for row in rows]
    require(len(actual)==len(identities) and set(actual)==set(identities),f"Missing/duplicate {label}")
    return rows


def checked_tensor(row, atol):
    require(isinstance(row,dict) and row.get("passed") is True,"Missing passing tensor comparison")
    require(row.get("finite") is True and row.get("shape_matches") is True,"Missing finite/shape proof")
    actual_shape = row.get("actual_shape",row.get("shape"))
    require(isinstance(actual_shape,list) and actual_shape==row.get("expected_shape"),"Tensor shape mismatch")
    require(row.get("actual_dtype",row.get("dtype"))==row.get("expected_dtype")=="float32","Missing FP32 proof")
    maximum = row.get("max_abs",row.get("max_abs_error"))
    require(type(maximum) in (int,float) and math.isfinite(maximum) and 0 <= maximum < atol,"Missing strict numeric proof")
    require(row.get("atol")==atol and row.get("rtol",0)==0,"Wrong comparison tolerance")
    require(row.get("empty",False) is False,"Empty tensor comparison")


def checked_comparison(comparison, names, atol, *, positions=False, padding=False):
    require(isinstance(comparison,dict) and comparison.get("passed") is True,"Missing passing comparison")
    rows = comparison.get("checkpoints")
    require(isinstance(rows,list) and tuple(row.get("name") for row in rows)==tuple(names),
            "Missing/duplicate/reordered comparison checkpoints")
    for row in rows:
        checked_tensor(row,atol)
        if positions:
            checked_tensor(row.get("valid_tokens"),atol)
            if padding:
                checked_tensor(row.get("padding_queries"),atol)


def checked_memory(execution):
    require(isinstance(execution,dict) and type(execution.get("executed_steps")) is int
            and execution["executed_steps"]>0,"Missing IR execution")
    memory = execution.get("memory_stats")
    require(isinstance(memory,dict),"Missing memory statistics")
    require(all(type(memory.get(key)) is int and memory[key]>=0 for key in MEMORY_FIELDS),
            "Missing/invalid memory statistic")
    require(isinstance(memory.get("scope"),str) and bool(memory["scope"]),"Missing memory scope")


def checked_file(folder, relative, digest=None):
    require(isinstance(relative,str) and bool(relative),"Missing evidence path")
    path = (folder/relative).resolve()
    require(path.is_relative_to(folder.resolve()),"Evidence path escapes output")
    require(path.is_file(),f"Missing evidence: {relative}")
    if digest is not None:
        require(isinstance(digest,str) and len(digest)==64 and sha256_file(path)==digest,
                f"Evidence hash mismatch: {relative}")
    return path


def checked_artifact(folder, artifact):
    require(isinstance(artifact,dict) and isinstance(artifact.get("sha256"),str),"Missing artifact hash")
    return checked_file(folder,artifact.get("path"),artifact["sha256"])


def validate_medium(report,folder):
    require(report.get("checkpoint_count")==81 and report.get("coverage_complete") is True,"Incomplete medium coverage")
    require(tuple(report.get("checkpoint_metadata",{}))==MEDIUM_CHECKPOINTS,"Wrong medium checkpoints")
    cases = checked_rows(report.get("cases"),MEDIUM_CASES,("name","valid_length"),"medium cases")
    kinds = (("pytorch","diagnostic"),("ort","normal"),("ort","diagnostic"),
             *((f"ir_{level}",kind) for level in LEVELS for kind in GRAPH_KINDS))
    identities = {(name,backend,kind) for name in ("causality","padding_isolation") for backend,kind in kinds}
    invariants = checked_rows(report.get("invariants"),identities,("name","backend","graph"),"medium invariants")
    trace_paths = {}
    for name,_ in MEDIUM_CASES:
        trace_paths[f"traces/{name}/reference.npz"] = (name,"pytorch","diagnostic",None)
        for kind in GRAPH_KINDS:
            trace_paths[f"traces/{name}/ort_{kind}.npz"] = (name,"ort",kind,None)
            for level in LEVELS:
                trace_paths[f"traces/{name}/ir_{kind}_{level}.npz"] = (name,"ir",kind,level)
    traces = checked_rows(report.get("trace_artifacts"),{(name,) for name in trace_paths},("path",),"medium traces")
    by_path = {row["path"]:row for row in traces}
    for trace in traces:
        require(tuple(trace.get(key) for key in ("case","backend","graph","optimization"))==trace_paths[trace["path"]],
                "Trace identity does not match its path")
        checked_artifact(folder,trace)
    for artifact in checked_keys(report.get("onnx"),GRAPH_KINDS,"medium ONNX graphs").values():
        checked_artifact(folder,artifact)
    checked_file(folder,"trace_schema.json")
    checked_file(folder,"logits_schema.json")
    for case in cases:
        require(case.get("passed") is True,"Failed medium case")
        checked_tensor(case.get("capture_preserves_logits"),1e-5)
        attention_names = {(f"layer_{layer}.{suffix}",) for layer in range(6)
                           for suffix in ("gqa_probabilities","gqa_context","blocked_attention")}
        for check in checked_rows(case.get("attention_checks"),attention_names,("name",),"medium attention checks"):
            checked_tensor(check,1e-5)
        reference = case.get("reference_trace",{})
        require(reference==by_path[f"traces/{case['name']}/reference.npz"],"Missing reference trace binding")
        ort = checked_keys(case.get("ort"),GRAPH_KINDS,"medium ORT graphs")
        ir = checked_keys(case.get("ir"),LEVELS,"medium IR levels")
        for kind in GRAPH_KINDS:
            names = MEDIUM_CHECKPOINTS if kind=="diagnostic" else ("logits",)
            require(ort[kind].get("passed") is True,"Missing successful medium ORT execution")
            checked_comparison(ort[kind].get("pytorch_comparison"),names,1e-5,positions=True,padding=case["valid_length"]<256)
            require(ort[kind].get("trace")==by_path[f"traces/{case['name']}/ort_{kind}.npz"],"Missing ORT trace binding")
            if kind=="diagnostic":
                checked_comparison(ort[kind].get("pack_layout"),names,1e-5)
            for level in LEVELS:
                execution = checked_keys(ir[level],GRAPH_KINDS,"medium IR graphs")[kind]
                require(execution.get("status")=="success" and execution.get("passed") is True,"Missing successful medium execution")
                checked_memory(execution)
                for comparison in ("pytorch_comparison","ort_comparison"):
                    checked_comparison(execution.get(comparison),names,1e-5,positions=True,padding=case["valid_length"]<256)
                require(execution.get("trace")==by_path[f"traces/{case['name']}/ir_{kind}_{level}.npz"],"Missing IR trace binding")
    for invariant in invariants:
        checked_comparison(invariant.get("comparison"),MEDIUM_CHECKPOINTS if invariant["graph"]=="diagnostic" else ("logits",),1e-5)


def validate_subgraphs(report,folder):
    require(report.get("sequence_length")==256 and report.get("coverage_complete") is True,"Not full L256 subgraph set")
    require(report.get("levels")==list(LEVELS),"Missing/duplicate subgraph levels")
    rows = checked_rows(report.get("cases"),{(name,) for name in SUBGRAPH_CASES},("name",),"subgraph cases")
    base = {"ort_pack_layout","torch_vs_ort","ordinary_ort_vs_torch","ordinary_ort_vs_diagnostic"}
    keys = base | {f"{prefix}_{level}_vs_{reference}" for level in LEVELS
                   for prefix,references in (("ir",("ort","torch")),("ordinary_ir",("ort","torch","diagnostic")))
                   for reference in references}
    for row in rows:
        name = row["name"]
        require(row.get("passed") is True,"Failed subgraph case")
        names = ("probabilities","y") if name.startswith("gqa_") else (("gate","up","hidden","y") if name=="swiglu" else ("y",))
        for key,comparison in checked_keys(row.get("comparisons"),keys,"subgraph comparisons").items():
            if key.startswith("ordinary_"):
                checked_tensor(comparison,1e-4)
            else:
                checked_comparison(comparison,names,1e-4)
        for executions in checked_keys(row.get("executions"),LEVELS,"subgraph IR levels").values():
            for execution in checked_keys(executions,("ordinary","diagnostic"),"subgraph IR graphs").values():
                checked_memory(execution)
        artifacts = row.get("artifacts",{})
        require(artifacts.get("directory")==name,"Wrong subgraph artifact directory")
        for filename,key in (("model.onnx","model_sha256"),("diagnostics.onnx","diagnostic_sha256")):
            checked_artifact(folder,{"path":f"{name}/{filename}","sha256":artifacts.get(key)})
        for filename in ("inputs.npz","torch.npz","ort.npz","ordinary_ort.npy","checkpoints.json",
                         *(f"ir_{level}.npz" for level in LEVELS),*(f"ordinary_ir_{level}.npy" for level in LEVELS)):
            checked_file(folder,f"{name}/{filename}")
        if name.startswith("gqa_"):
            for checks in checked_keys(row.get("semantic_checks"),("torch","ort","ir_none","ir_basic","ir_all"),"GQA backends").values():
                for comparison in checked_keys(checks,("blocked_probability","probability_row_sum"),"GQA semantics").values():
                    checked_tensor(comparison,1e-4)
    identities = {(name,backend) for name in ("padding_isolation","causality")
                  for backend in ("torch","ort","ir_none","ir_basic","ir_all")}
    for invariant in checked_rows(report.get("invariants"),identities,("name","backend"),"subgraph invariants"):
        require(invariant.get("query_prefix")== (256 if invariant["name"]=="padding_isolation" else 64),"Wrong invariant prefix")
        checked_comparison(invariant,("probabilities","y"),1e-4)


def validate_attention(report,folder):
    require(report.get("planned_executions")==report.get("passed_executions")==12,"Missing QEMU runs")
    rows = checked_rows(report.get("cases"),ATTENTION_CASES,("name","valid_length"),"Attention cases")
    for row in rows:
        name = row["name"]
        require(row.get("passed") is True,"Failed Attention case")
        checked_tensor(row.get("ort_vs_numpy"),1e-4)
        for filename,key in (("model.onnx","model_sha256"),("inputs.npz","input_sha256")):
            checked_artifact(folder,{"path":f"{name}/{filename}","sha256":row.get(key)})
        for execution in checked_rows(row.get("executions"),{("none",),("all",)},("optimization",),"Attention executions"):
            require(execution.get("status")=="success" and execution.get("passed") is True,"Missing successful target execution")
            for key in ("ir_vs_ort","ir_vs_numpy","qemu_vs_ort","qemu_vs_numpy"):
                checked_tensor(execution.get(key),1e-4)
            elapsed = execution.get("qemu_process_wall_seconds")
            require(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>=0,"Missing QEMU wall time")
            for key in ("command","compile_command"):
                command = execution.get(key)
                require(isinstance(command,list) and len(command)>1 and all(isinstance(x,str) and x for x in command),"Missing target command")
            level = execution["optimization"]
            checked_artifact(folder,{"path":f"{name}/{level}/build/model.elf","sha256":execution.get("elf_sha256")})
            for filename in ("qemu.npy","ir.npy"):
                checked_file(folder,f"{name}/{level}/{filename}")
    identities = {(name,reference,prefix,backend) for name,reference,prefix in
                  (("future17","full17",5),("padding_changed17","padding17",17))
                  for backend in ("ort","ir_none","ir_all","qemu_none","qemu_all")}
    for invariant in checked_rows(report.get("invariants"),identities,("case","reference","prefix","backend"),"Attention invariants"):
        checked_tensor(invariant.get("comparison"),1e-4)

def validate_report(name, folder, sources):
    for filename in ("report.json","report.md","report.html"):
        require((folder/filename).is_file(),f"Missing {filename}")
    report = json.loads((folder/"report.json").read_text(encoding="utf-8"))
    require(report.get("gate")==IDENTITIES[name],"Wrong gate identity")
    require(report.get("status")=="PASS" and report.get("passed") is True,"Gate not PASS")
    fingerprints = report.get("source_sha256") or report.get("provenance",{}).get("source_sha256")
    require(fingerprints == sources,"Source fingerprints differ from this invocation")
    check_numeric(report,1e-5 if name in ("medium","layer-diff") else 1e-4)
    if name == "medium":
        validate_medium(report,folder)
    elif name == "subgraphs":
        validate_subgraphs(report,folder)
    elif name == "attention":
        validate_attention(report,folder)
    elif name == "full-preflight":
        require(report.get("full_ir_executed") is False and report.get("w3_exit_accepted") is False,
                "Preflight must not claim complete-model numerical acceptance")
        require(report.get("node_count")==7847 and report.get("weights_file_bytes")==2384201728,"Unexpected fixed full assets")
    else:
        coverage = report.get("coverage",{})
        require(coverage.get("complete") is True and coverage.get("compared_checkpoints")==81,"Incomplete layer comparison")
    return {"report_sha256":sha256_file(folder/"report.json"),"gate":report["gate"],
            "child_seconds":report.get("seconds",report.get("elapsed_seconds")),
            "process_peak_rss_bytes":report.get("process_peak_rss_bytes")}


def wait_child(process,timeout,row):
    """Bound the child and also reclaim its owned process tree on cancellation."""
    if getattr(process,"_scratchv_job",None) is not None or getattr(process,"_scratchv_owned_group",False):
        try:
            # Share the full runner's ownership protocol, including children
            # surviving their launcher. Preparation has no memory threshold;
            # the added samples only describe the owned subprocess tree.
            return wait_bounded(process,timeout=timeout,max_memory_bytes=sys.maxsize,row=row)
        except TimeoutError:
            raise subprocess.TimeoutExpired(process.args,timeout) from None
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        if process.poll() is None:
            try:
                terminate_process_tree(process)
            except (OSError,subprocess.SubprocessError) as cleanup:
                row["cleanup_error"] = f"{type(cleanup).__name__}: {cleanup}"
        row["returncode"] = process.returncode
        raise

def build_commands(args, out):
    # Keep the selected venv symlink: resolving it runs the base interpreter.
    python = [str(Path(args.python).absolute()),"-B","-X","utf8"]
    tools = []
    if args.cc:
        tools += ["--cc",args.cc]
    if args.qemu:
        tools += ["--qemu",args.qemu]
    medium = out/"medium"
    return {
        "medium":python+["probes/w3_qwen3_medium/run.py","--output-dir",str(medium)],
        "subgraphs":python+["probes/w3_qwen3_subgraphs/run.py","--source-dir",str(args.source_dir.resolve()),
                            "--output-dir",str(out/"subgraphs")],
        "attention":python+["probes/w3_attention/run.py","--output-dir",str(out/"attention"),
                            "--timeout",str(args.qemu_timeout),*tools],
        "full-preflight":python+["probes/w3_full_preflight/run.py","--model-dir",str(args.model_dir.resolve()),
                                "--output-dir",str(out/"full-preflight")],
        "layer-diff":python+["probes/w3_layer_diff/run.py",
                            "--actual",str(medium/"traces/short_17/ir_diagnostic_all.npz"),
                            "--reference",str(medium/"traces/short_17/ort_diagnostic.npz"),
                            "--schema",str(medium/"trace_schema.json"),"--out",str(out/"layer-diff")],
    }

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python",default=sys.executable)
    parser.add_argument("--source-dir",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--cc")
    parser.add_argument("--qemu")
    parser.add_argument("--gate-timeout",type=float,default=1800)
    parser.add_argument("--qemu-timeout",type=float,default=180)
    args = parser.parse_args(argv)
    if any(not math.isfinite(x) or x<=0 for x in (args.gate_timeout,args.qemu_timeout)):
        parser.error("Timeouts must be finite and positive")
    out = new_output_dir(args.output_dir)
    (out/"logs").mkdir()
    started = time.perf_counter()
    report = {"gate":"preparation:w3","passed":False,"status":"FAIL","gates":[],
              "w3_exit_accepted":False,"full_ir_executed":False,
              "scope":"Five local preparation checks only; full IR, team sign-off and Nightly remain separate"}
    try:
        report.update(source_evidence())
        commands = build_commands(args,out)
        env = dict(os.environ,PYTHONIOENCODING="utf-8",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
        for name in GATES:
            row = {"name":name,"passed":False,"status":"FAIL","command":commands[name]}
            report["gates"].append(row)
            tick = time.perf_counter()
            try:
                if name=="layer-diff":
                    require(report["gates"][0]["passed"],"Layer-diff blocked by failed medium evidence")
                with (out/"logs"/(name+".log")).open("w",encoding="utf-8") as stream:
                    options = {"creationflags":subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=="nt" else {"start_new_session":True}
                    process = spawn_owned(commands[name],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env,**options)
                    try:
                        row["returncode"] = wait_child(process,args.gate_timeout,row)
                    except subprocess.TimeoutExpired:
                        raise TimeoutError(f"{name} exceeded {args.gate_timeout} seconds") from None
                require(row["returncode"]==0,f"{name} exited {row['returncode']}; see logs/{name}.log")
                row["validation"] = validate_report(name,out/name,report["source_sha256"])
                row.update(passed=True,status="PASS")
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["elapsed_seconds"] = time.perf_counter()-tick
            print(f"[{name}] {row['status']}: {row.get('error','verified')}",flush=True)
        require(source_evidence()["source_sha256"]==report["source_sha256"],"Source changed during execution")
        report["passed"] = len(report["gates"])==len(GATES) and all(r["passed"] for r in report["gates"])
    except KeyboardInterrupt:
        report.update(passed=False,interrupted=True,error="KeyboardInterrupt: preparation cancelled")
        if report["gates"]:
            report["gates"][-1].update(passed=False,status="FAIL",interrupted=True)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report.update(status="PASS" if report["passed"] else "FAIL",elapsed_seconds=time.perf_counter()-started)
    write_reports(out,report)
    return 130 if report.get("interrupted") else (0 if report["passed"] else 1)

if __name__=="__main__":
    raise SystemExit(main())
