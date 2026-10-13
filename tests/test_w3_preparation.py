import json
from pathlib import Path
import pytest
from onnx import helper, TensorProto
from probes.w3_full_preflight.run import size_accounting
from scripts.run_w3_preparation import validate_report, check_numeric

def test_size_accounting_is_logical_not_alias_storage():
    graph = helper.make_graph([], "sizes",
        [helper.make_tensor_value_info("x",TensorProto.FLOAT,[1,256,128])],
        [helper.make_tensor_value_info("y",TensorProto.FLOAT,[1,256,128])])
    result = size_accounting(helper.make_model(graph))
    assert result["sum_named_value_logical_bytes"] == 2*256*128*4
    assert "Not measured" in result["scope"]
    assert result["output_logical_bytes"]["y"] == 256*128*4

def test_dynamic_shape_is_not_guessed_as_zero():
    graph = helper.make_graph([], "dynamic", [],
        [helper.make_tensor_value_info("y",TensorProto.FLOAT,[1,"L",128])])
    result = size_accounting(helper.make_model(graph))
    assert result["unknown_size_values"] == ["y"]
    assert result["output_logical_bytes"]["y"] is None

def test_absent_rank_and_missing_value_info_are_explicit():
    graph = helper.make_graph([helper.make_node("Identity",["x"],["hidden"])], "missing",
        [helper.make_tensor_value_info("x",TensorProto.FLOAT,[1,4])],
        [helper.make_tensor_value_info("y",TensorProto.FLOAT,None)])
    result = size_accounting(helper.make_model(graph))
    assert result["output_logical_bytes"]["y"] is None
    assert "hidden" in result["values_without_persisted_static_size"]

@pytest.mark.parametrize("value",[
    {"passed":True,"comparisons":[{"passed":False}]},
    {"passed":True,"max_abs":1e-5},
    {"max_abs":float("nan")}, {"max_abs":float("inf")}, {"max_abs":-1},
])
def test_aggregate_rejects_bad_numeric_evidence(value):
    with pytest.raises(ValueError):
        check_numeric(value,1e-5)

def good_preflight(folder):
    data = {"gate":"preflight:w3-full-assets","status":"PASS","passed":True,
            "source_sha256":{"x.py":"abc"},"full_ir_executed":False,"w3_exit_accepted":False,
            "node_count":7847,"weights_file_bytes":2384201728}
    for name in ("report.md","report.html"):
        (folder/name).write_text("evidence",encoding="utf-8")
    return data

@pytest.mark.parametrize("mutation",["source","identity","full_claim","missing_view","skipped"])
def test_preflight_cannot_forge_full_acceptance(tmp_path,mutation):
    data = good_preflight(tmp_path)
    if mutation=="source":
        data["source_sha256"]={}
    elif mutation=="identity":
        data["gate"]="numeric:ir-full-qwen3"
    elif mutation=="full_claim":
        data["full_ir_executed"]=True
    elif mutation=="missing_view":
        (tmp_path/"report.html").unlink()
    else:
        data["status"]="SKIPPED"
    (tmp_path/"report.json").write_text(json.dumps(data),encoding="utf-8")
    with pytest.raises(ValueError):
        validate_report("full-preflight",tmp_path,{"x.py":"abc"})


# These are report-contract fixtures, not numerical or model-execution evidence.
def comparison_fixture(atol):
    return {"passed":True,"shape":[1],"expected_shape":[1],"dtype":"float32",
            "expected_dtype":"float32","finite":True,"shape_matches":True,
            "max_abs_error":0.0,"atol":atol,"rtol":0}


def named_comparison_fixture(names,atol,positions=False):
    rows=[]
    for name in names:
        row={"name":name,**comparison_fixture(atol)}
        if positions:
            row["valid_tokens"]=comparison_fixture(atol)
            row["padding_queries"]=comparison_fixture(atol)
        rows.append(row)
    return {"passed":True,"checkpoints":rows}


def memory_fixture():
    from scripts.run_w3_preparation import MEMORY_FIELDS
    return {**dict.fromkeys(MEMORY_FIELDS,4),"scope":"unit fixture"}


def artifact_fixture(folder,relative):
    import hashlib
    path=folder/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b"report-contract fixture")
    return {"path":relative,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def medium_report_fixture(folder):
    from scripts import run_w3_preparation as gate
    report={"checkpoint_count":81,"coverage_complete":True,
            "checkpoint_metadata":dict.fromkeys(gate.MEDIUM_CHECKPOINTS,{}),
            "cases":[],"invariants":[],"trace_artifacts":[],"onnx":{}}
    for kind,file in (("normal","model.onnx"),("diagnostic","diagnostics.onnx")):
        report["onnx"][kind]=artifact_fixture(folder,file)
    for file in ("trace_schema.json","logits_schema.json"):
        artifact_fixture(folder,file)
    for name,length in gate.MEDIUM_CASES:
        row={"name":name,"valid_length":length,"passed":True,"ort":{},"ir":{},
             "capture_preserves_logits":comparison_fixture(1e-5),"attention_checks":[]}
        report["cases"].append(row)
        for layer in range(6):
            for suffix in ("gqa_probabilities","gqa_context","blocked_attention"):
                row["attention_checks"].append({"name":f"layer_{layer}.{suffix}",**comparison_fixture(1e-5)})
        def trace(backend,kind,level=None):
            filename="reference" if backend=="pytorch" else (f"ort_{kind}" if backend=="ort" else f"ir_{kind}_{level}")
            artifact={**artifact_fixture(folder,f"traces/{name}/{filename}.npz"),
                      "case":name,"backend":backend,"graph":kind,"optimization":level}
            report["trace_artifacts"].append(artifact)
            return artifact
        row["reference_trace"]=trace("pytorch","diagnostic")
        for kind in gate.GRAPH_KINDS:
            names=gate.MEDIUM_CHECKPOINTS if kind=="diagnostic" else ("logits",)
            row["ort"][kind]={"passed":True,"trace":trace("ort",kind),
                "pytorch_comparison":named_comparison_fixture(names,1e-5,True)}
            if kind=="diagnostic":
                row["ort"][kind]["pack_layout"]=named_comparison_fixture(names,1e-5)
            for level in gate.LEVELS:
                row["ir"].setdefault(level,{})[kind]={"passed":True,"status":"success",
                    "executed_steps":1,"memory_stats":memory_fixture(),"trace":trace("ir",kind,level),
                    "pytorch_comparison":named_comparison_fixture(names,1e-5,True),
                    "ort_comparison":named_comparison_fixture(names,1e-5,True)}
    kinds=[("pytorch","diagnostic"),("ort","normal"),("ort","diagnostic")]
    kinds += [(f"ir_{level}",kind) for level in gate.LEVELS for kind in gate.GRAPH_KINDS]
    for name in ("causality","padding_isolation"):
        for backend,kind in kinds:
            report["invariants"].append({"name":name,"backend":backend,"graph":kind,"passed":True,
                "comparison":named_comparison_fixture(gate.MEDIUM_CHECKPOINTS if kind=="diagnostic" else ("logits",),1e-5)})
    return report


def subgraph_report_fixture(folder):
    from scripts import run_w3_preparation as gate
    report={"sequence_length":256,"coverage_complete":True,"levels":list(gate.LEVELS),"cases":[],"invariants":[]}
    for name in gate.SUBGRAPH_CASES:
        names=("probabilities","y") if name.startswith("gqa_") else (("gate","up","hidden","y") if name=="swiglu" else ("y",))
        row={"name":name,"passed":True,"comparisons":{},"executions":{},"semantic_checks":{},
             "artifacts":{"directory":name}}
        report["cases"].append(row)
        for key in ("ort_pack_layout","torch_vs_ort"):
            row["comparisons"][key]=named_comparison_fixture(names,1e-4)
        for key in ("ordinary_ort_vs_torch","ordinary_ort_vs_diagnostic"):
            row["comparisons"][key]=comparison_fixture(1e-4)
        for level in gate.LEVELS:
            row["executions"][level]={kind:{"executed_steps":1,"memory_stats":memory_fixture()}
                                       for kind in ("ordinary","diagnostic")}
            for reference in ("ort","torch"):
                row["comparisons"][f"ir_{level}_vs_{reference}"]=named_comparison_fixture(names,1e-4)
            for reference in ("ort","torch","diagnostic"):
                row["comparisons"][f"ordinary_ir_{level}_vs_{reference}"]=comparison_fixture(1e-4)
        for file,key in (("model.onnx","model_sha256"),("diagnostics.onnx","diagnostic_sha256")):
            row["artifacts"][key]=artifact_fixture(folder,f"{name}/{file}")["sha256"]
        for file in ("inputs.npz","torch.npz","ort.npz","ordinary_ort.npy","checkpoints.json",
                     *(f"ir_{level}.npz" for level in gate.LEVELS),*(f"ordinary_ir_{level}.npy" for level in gate.LEVELS)):
            artifact_fixture(folder,f"{name}/{file}")
        if name.startswith("gqa_"):
            row["semantic_checks"]={backend:{check:comparison_fixture(1e-4)
                for check in ("blocked_probability","probability_row_sum")}
                for backend in ("torch","ort","ir_none","ir_basic","ir_all")}
    for name in ("padding_isolation","causality"):
        for backend in ("torch","ort","ir_none","ir_basic","ir_all"):
            report["invariants"].append({"name":name,"backend":backend,"query_prefix":256 if name=="padding_isolation" else 64,
                                        **named_comparison_fixture(("probabilities","y"),1e-4)})
    return report


def attention_report_fixture(folder):
    from scripts import run_w3_preparation as gate
    report={"planned_executions":12,"passed_executions":12,"cases":[],"invariants":[]}
    for name,valid in gate.ATTENTION_CASES:
        row={"name":name,"valid_length":valid,"passed":True,"executions":[],"ort_vs_numpy":comparison_fixture(1e-4)}
        report["cases"].append(row)
        for file,key in (("model.onnx","model_sha256"),("inputs.npz","input_sha256")):
            row[key]=artifact_fixture(folder,f"{name}/{file}")["sha256"]
        for level in ("none","all"):
            execution={"optimization":level,"passed":True,"status":"success","qemu_process_wall_seconds":0.25,
                       "command":["qemu","-kernel","model.elf"],"compile_command":["cc","model.c"],
                       "elf_sha256":artifact_fixture(folder,f"{name}/{level}/build/model.elf")["sha256"]}
            for key in ("ir_vs_ort","ir_vs_numpy","qemu_vs_ort","qemu_vs_numpy"):
                execution[key]=comparison_fixture(1e-4)
            row["executions"].append(execution)
            for file in ("ir.npy","qemu.npy"):
                artifact_fixture(folder,f"{name}/{level}/{file}")
    for name,reference,prefix in (("future17","full17",5),("padding_changed17","padding17",17)):
        for backend in ("ort","ir_none","ir_all","qemu_none","qemu_all"):
            report["invariants"].append({"case":name,"reference":reference,"prefix":prefix,"backend":backend,
                                        "passed":True,"comparison":comparison_fixture(1e-4)})
    return report


def publish_fixture(folder,report,name):
    from scripts import run_w3_preparation as gate
    report.update(gate=gate.IDENTITIES[name],passed=True,status="PASS",source_sha256={"fixture.py":"fixed"})
    for filename in ("report.md","report.html"):
        (folder/filename).write_text("contract fixture",encoding="utf-8")
    (folder/"report.json").write_text(json.dumps(report),encoding="utf-8")


@pytest.mark.parametrize("name",["medium","subgraphs","attention"])
def test_complete_execution_matrices_are_accepted(tmp_path,name):
    builder={"medium":medium_report_fixture,"subgraphs":subgraph_report_fixture,"attention":attention_report_fixture}[name]
    report=builder(tmp_path)
    publish_fixture(tmp_path,report,name)
    assert validate_report(name,tmp_path,{"fixture.py":"fixed"})["gate"]==report["gate"]


@pytest.mark.parametrize("name,mutation",[
    ("medium","missing_ir_graph"),("medium","missing_ort_graph"),("medium","missing_comparison"),
    ("medium","missing_checkpoint"),("medium","missing_padding"),("medium","duplicate_case"),
    ("subgraphs","missing_all_comparisons"),("subgraphs","missing_comparison"),
    ("subgraphs","missing_memory"),("subgraphs","missing_ir_graph"),("subgraphs","duplicate_invariant"),
    ("attention","duplicate_execution"),("attention","missing_comparison"),
    ("attention","missing_command"),("attention","missing_wall_time"),("attention","duplicate_invariant"),
])
def test_pass_scalar_cannot_hide_missing_execution_evidence(tmp_path,name,mutation):
    builder={"medium":medium_report_fixture,"subgraphs":subgraph_report_fixture,"attention":attention_report_fixture}[name]
    report=builder(tmp_path)
    row=report["cases"][0]
    if mutation=="duplicate_case":
        report["cases"][1]=report["cases"][0]
    elif mutation=="duplicate_invariant":
        report["invariants"][1]=report["invariants"][0]
    elif name=="medium":
        if mutation=="missing_ir_graph": row["ir"]["basic"].pop("normal")
        elif mutation=="missing_ort_graph": row["ort"].pop("diagnostic")
        elif mutation=="missing_comparison": row["ir"]["all"]["diagnostic"].pop("ort_comparison")
        elif mutation=="missing_checkpoint": row["ir"]["none"]["diagnostic"]["ort_comparison"]["checkpoints"].pop()
        else: report["cases"][2]["ir"]["none"]["normal"]["ort_comparison"]["checkpoints"][0].pop("padding_queries")
    elif name=="subgraphs":
        if mutation=="missing_all_comparisons": row.pop("comparisons")
        elif mutation=="missing_comparison": row["comparisons"].pop("ir_all_vs_ort")
        elif mutation=="missing_memory": row["executions"]["basic"]["ordinary"].pop("memory_stats")
        else: row["executions"]["none"].pop("diagnostic")
    else:
        if mutation=="duplicate_execution": row["executions"][1]=row["executions"][0]
        elif mutation=="missing_comparison": row["executions"][0].pop("qemu_vs_numpy")
        elif mutation=="missing_command": row["executions"][0].pop("command")
        else: row["executions"][0].pop("qemu_process_wall_seconds")
    publish_fixture(tmp_path,report,name)
    with pytest.raises(ValueError):
        validate_report(name,tmp_path,{"fixture.py":"fixed"})


@pytest.mark.parametrize("name,path",[("medium","traces/full_seed_0/reference.npz"),
                                      ("subgraphs","projection_q/diagnostics.onnx"),
                                      ("attention","full17/none/build/model.elf")])
def test_existing_artifact_with_changed_bytes_is_rejected(tmp_path,name,path):
    builder={"medium":medium_report_fixture,"subgraphs":subgraph_report_fixture,"attention":attention_report_fixture}[name]
    report=builder(tmp_path)
    publish_fixture(tmp_path,report,name)
    (tmp_path/path).write_bytes(b"tampered")
    with pytest.raises(ValueError,match="hash mismatch"):
        validate_report(name,tmp_path,{"fixture.py":"fixed"})


@pytest.mark.parametrize("failure",[KeyboardInterrupt,RuntimeError])
def test_wait_child_cleans_live_process_without_swallowing_interrupt(monkeypatch,failure):
    from scripts import run_w3_preparation as gate
    class Child:
        returncode=None
        def wait(self,timeout): raise failure()
        def poll(self): return self.returncode
    child=Child()
    calls=[]
    def cleanup(process):
        calls.append(process)
        process.returncode=-9
    monkeypatch.setattr(gate,"terminate_process_tree",cleanup)
    row={}
    with pytest.raises(failure):
        gate.wait_child(child,1,row)
    assert calls==[child] and row["returncode"]==-9


def test_keyboard_interrupt_writes_fail_and_stops_subsequent_gates(tmp_path,monkeypatch):
    from scripts import run_w3_preparation as gate
    calls=[]
    class Child:
        returncode=None
        def wait(self,timeout): raise KeyboardInterrupt()
        def poll(self): return self.returncode
    def spawn(*args,**kwargs):
        calls.append(args[0])
        return Child()
    monkeypatch.setattr(gate,"spawn_owned",spawn)
    monkeypatch.setattr(gate,"terminate_process_tree",lambda child:setattr(child,"returncode",-9))
    monkeypatch.setattr(gate,"source_evidence",lambda:{"source_sha256":{"fixture.py":"fixed"}})
    output=tmp_path/"interrupted"
    code=gate.main(["--source-dir",str(tmp_path),"--model-dir",str(tmp_path),"--output-dir",str(output)])
    report=json.loads((output/"report.json").read_text(encoding="utf-8"))
    assert code==130 and len(calls)==1 and len(report["gates"])==1
    assert report["status"]=="FAIL" and report["passed"] is False and report["interrupted"] is True
    assert report["gates"][0]["returncode"]==-9
