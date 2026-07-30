
"""
BridgeDEUX Core Framework
ONNX Recursive Graph Topology Inspector (Baseline)
"""

import sys
from pathlib import Path
import onnx
from collections import Counter

def inspect_graph_recursively(graph, level=0, context_name="Root"):
    indent = "  " * (level * 2)
    
    # 1. Base Graph Info & I/O Signatures
    graph_name = graph.name if graph.name else "<unnamed>"
    print(f"\n{indent}--- Graph Level {level} [{context_name}] ---")
    print(f"{indent}Graph Name: {graph_name}")
    
    print(f"{indent}Inputs ({len(graph.input)}):")
    for value in graph.input:
        print(f"{indent}  - {value.name}")
        
    print(f"\n{indent}Outputs ({len(graph.output)}):")
    for value in graph.output:
        print(f"{indent}  - {value.name}")
        
    print(f"\n{indent}Total Nodes: {len(graph.node)}")
    
    # 2. Complete Node List
    print(f"\n{indent}Node List:")
    for i, node in enumerate(graph.node):
        node_name = node.name if node.name else "<unnamed>"
        domain = node.domain if node.domain else "ai.onnx"
        print(f"{indent}  [{i}] {domain}::{node.op_type} ({node_name})")

    # 3. Operator Histogram
    op_counts = Counter(node.op_type for node in graph.node)
    print(f"\n{indent}Operator Histogram:")
    for op, count in op_counts.most_common():
        print(f"{indent}  {op}: {count}")
        
    # 4. Identify and traverse control-flow / nested-graph nodes
    for i, node in enumerate(graph.node):
        has_subgraph = any(
            attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) 
            for attr in node.attribute
        )
        
        if node.op_type in ["If", "Loop", "Scan"] or has_subgraph:
            node_name = node.name if node.name else f"<unnamed_node_{i}>"
            print(f"\n{indent}[!] Found Nested Graph Container: Node {i}")
            print(f"{indent}    Type : {node.op_type}")
            print(f"{indent}    Name : {node_name}")
            print(f"{indent}    Attributes:")
            
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    print(f"{indent}      - {attr.name} (Type: GRAPH)")
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    print(f"{indent}      - {attr.name} (Type: GRAPHS, Count: {len(attr.graphs)})")
                else:
                    print(f"{indent}      - {attr.name} (Type: {attr.type})")
                    
            # Execute recursive descent
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    print(f"\n{indent}    -> Descending into: {node_name}.{attr.name}")
                    inspect_graph_recursively(attr.g, level + 1, context_name=f"{attr.name}")
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for idx, sub_g in enumerate(attr.graphs):
                        print(f"\n{indent}    -> Descending into: {node_name}.{attr.name}[{idx}]")
                        inspect_graph_recursively(sub_g, level + 1, context_name=f"{attr.name}[{idx}]")

def main():
    decoder_path = Path("models/onnx/opus_mt_de_en_opt_extended/decoder_model_merged.onnx")
    
    if not decoder_path.exists():
        print(f"[ERROR] File not found: {decoder_path}")
        sys.exit(1)
        
    print(f"Loading {decoder_path.name}...")
    model = onnx.load(str(decoder_path), load_external_data=False)
    
    inspect_graph_recursively(model.graph)

if __name__ == "__main__":
    main()