"""
BridgeDEUX Core Framework
ONNX Graph Inspector - Research-Grade A/B Diagnostic Tool (v6.0 Final)
"""

import sys
import math
import json
from pathlib import Path
from collections import Counter
import onnx

TENSOR_TYPE_MAP = {
    1: "FLOAT (32-bit)", 2: "UINT8", 3: "INT8", 4: "UINT16",
    5: "INT16", 6: "INT32", 7: "INT64", 10: "FLOAT16",
    11: "DOUBLE (64-bit)", 16: "BFLOAT16"
}

def get_tensor_type_str(data_type_int: int) -> str:
    return TENSOR_TYPE_MAP.get(data_type_int, f"UNKNOWN ({data_type_int})")

def analyze_graph(model_path: Path):
    if not model_path.exists():
        return {"status": "error", "error_message": "File not found"}
    
    try:
        model = onnx.load(str(model_path), load_external_data=True)
        graph = model.graph
        
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        op_counts = Counter(node.op_type for node in graph.node)
        
        init_type_counts = Counter()
        init_element_counts = Counter()
        total_elements = 0
        quantized_elements = 0
        
        has_external_data = any(
            getattr(init, "data_location", 0) == onnx.TensorProto.EXTERNAL 
            for init in graph.initializer
        )
        
        for init in graph.initializer:
            t_type_str = get_tensor_type_str(init.data_type)
            init_type_counts[t_type_str] += 1
            
            elem_count = math.prod(init.dims) if init.dims else 1
            init_element_counts[t_type_str] += elem_count
            total_elements += elem_count
            
            if init.data_type in (2, 3):  # UINT8 or INT8
                quantized_elements += elem_count

        int8_ratio = (quantized_elements / total_elements * 100) if total_elements > 0 else 0.0
        
        return {
            "status": "success",
            "file_size_mb": round(file_size_mb, 2),
            "has_external_data": has_external_data,
            "ir_version": model.ir_version,
            "opsets": ", ".join([f"{op.domain or 'ai.onnx'} v{op.version}" for op in model.opset_import]),
            "total_nodes": len(graph.node),
            "total_initializers": len(graph.initializer),
            "total_elements": total_elements,
            "quantized_elements": quantized_elements,
            "int8_ratio": round(int8_ratio, 2),
            "op_counts": dict(op_counts),
            "init_type_counts": dict(init_type_counts),
            "init_element_counts": dict(init_element_counts)
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

def compare_models(component_name: str, fp32_path: Path, int8_path: Path) -> tuple[str, dict]:
    report = []
    report.append(f"\n{'='*75}")
    report.append(f" A/B Diagnostic Analysis: {component_name}")
    report.append(f" FP32 Source : {fp32_path}")
    report.append(f" INT8 Target : {int8_path}")
    report.append(f"{'='*75}")
    
    fp32_data = analyze_graph(fp32_path)
    int8_data = analyze_graph(int8_path)
    
    json_payload = {
        "component": component_name,
        "fp32_model": fp32_data,
        "int8_model": int8_data
    }
    
    if fp32_data.get("status") == "error" or int8_data.get("status") == "error":
        report.append("\n[ERROR] Analysis failed for one or both models.")
        if fp32_data.get("status") == "error":
            report.append(f"  FP32 Error: {fp32_data.get('error_message')}")
        if int8_data.get("status") == "error":
            report.append(f"  INT8 Error: {int8_data.get('error_message')}")
        return "\n".join(report), json_payload

    report.append("\n--- 1. File Size & Macro Structural Changes ---")
    report.append(f" File Size (MB)       : {fp32_data['file_size_mb']:.2f} MB -> {int8_data['file_size_mb']:.2f} MB "
                  f"({(int8_data['file_size_mb'] - fp32_data['file_size_mb']):+.2f} MB)")
    report.append(f" External Tensor Data : {'Yes' if fp32_data['has_external_data'] else 'No'} -> {'Yes' if int8_data['has_external_data'] else 'No'}")
    report.append(f" Total Nodes          : {fp32_data['total_nodes']} -> {int8_data['total_nodes']}")
    report.append(f" Total Initializers   : {fp32_data['total_initializers']} -> {int8_data['total_initializers']}")

    report.append("\n--- 2. Weight Element Quantization Distribution ---")
    report.append(f" Total Weight Elements: {fp32_data['total_elements']:,} -> {int8_data['total_elements']:,}")
    report.append(f" Observed INT8 Elements: {int8_data['quantized_elements']:,} / {int8_data['total_elements']:,}")
    report.append(f" Observed INT8 Ratio  : {int8_data['int8_ratio']:.2f}%")

    report.append("\n--- 3. Initializer Type & Element Breakdown ---")
    all_init_types = sorted(set(fp32_data['init_type_counts']) | set(int8_data['init_type_counts']))
    for t in all_init_types:
        fp_t_count = fp32_data['init_type_counts'].get(t, 0)
        int_t_count = int8_data['init_type_counts'].get(t, 0)
        fp_e_count = fp32_data['init_element_counts'].get(t, 0)
        int_e_count = int8_data['init_element_counts'].get(t, 0)
        report.append(f" {t:<16} | Tensors: {fp_t_count:<4} -> {int_t_count:<4} | Elements: {fp_e_count:>12,} -> {int_e_count:>12,}")

    report.append("\n--- 4. Operator Deltas (Quantization & Changed Ops Only) ---")
    all_ops = sorted(set(fp32_data['op_counts']) | set(int8_data['op_counts']))
    quant_ops = ["MatMulInteger", "DynamicQuantizeLinear", "QLinearMatMul", "QuantizeLinear", "DequantizeLinear", "QLinearConv"]
    
    report.append(" [Quantization Control Operators]")
    for op in quant_ops:
        fp_c = fp32_data['op_counts'].get(op, 0)
        int_c = int8_data['op_counts'].get(op, 0)
        report.append(f"   {op:<25}: {fp_c:<5} -> {int_c:<5} (Delta: {int_c - fp_c:+d})")

    report.append("\n [Graph Structural Changes (Suppressed Identical Ops)]")
    changed_found = False
    for op in all_ops:
        if op in quant_ops: continue
        fp_c = fp32_data['op_counts'].get(op, 0)
        int_c = int8_data['op_counts'].get(op, 0)
        if fp_c != int_c:
            changed_found = True
            report.append(f" * {op:<25}: {fp_c:<5} -> {int_c:<5} (Delta: {int_c - fp_c:+d})")
            
    if not changed_found:
        report.append("   (No standard operators were modified or replaced)")

    return "\n".join(report), json_payload

def main() -> None:
    base_dir_fp32 = Path("models/onnx/opus_mt_de_en_opt_extended")
    base_dir_int8 = Path("models/onnx/opus_mt_de_en_opt_extended_int8")
    analysis_dir = Path("analysis")
    
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_txt_file = analysis_dir / "onnx_quantization_diagnostic_final.txt"
    report_json_file = analysis_dir / "onnx_quantization_diagnostic_final.json"
    
    full_text_report = []
    full_json_payload = {}
    
    models_to_check = [
        ("Encoder", "encoder_model.onnx"),
        ("Decoder (Standard)", "decoder_model.onnx"),
        ("Decoder (With Past)", "decoder_with_past_model.onnx")
    ]
    
    for name, filename in models_to_check:
        try:
            txt, json_data = compare_models(
                name,
                base_dir_fp32 / filename,
                base_dir_int8 / filename
            )
            full_text_report.append(txt)
            full_json_payload[name] = json_data
            print(txt)
        except Exception as e:
            error = (
                f"\n{'='*75}\n"
                f" A/B Diagnostic Analysis: {name}\n"
                f"{'='*75}\n\n"
                f"[ERROR] {e}\n"
            )
            print(error)
            full_text_report.append(error)
            full_json_payload[name] = {"error": str(e)}
            
    with open(report_txt_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(full_text_report))
        
    with open(report_json_file, "w", encoding="utf-8") as f:
        json.dump(full_json_payload, f, indent=4)
    
    print(f"\n[INFO] Reports saved to:\n  - {report_txt_file}\n  - {report_json_file}")

if __name__ == "__main__":
    main()