# Cleans up the traced ONNX graph before TensorRT parses it. Export runs without constant folding (folding would
# rename the weights), so the graph holds every constant computation the tracer saw, including the shape arithmetic
# around each reshape, which TensorRT would turn into layers (about half of SDXL's ~14,000 nodes).
import collections

import numpy as np
import onnx
from onnx import helper, numpy_helper

MAX_FOLDED = 1 << 16  # elements: larger constant results stay as nodes, keeping the .onnx file small
_FOLDABLE = {
    "Abs", "Add", "Cast", "Ceil", "Concat", "ConstantOfShape", "Cos", "Div", "Equal", "Exp", "Expand", "Flatten", "Floor",
    "Gather", "Greater", "Identity", "Less", "Log", "Max", "Min", "Mul", "Neg", "Not", "Pow", "Range", "Reciprocal",
    "Reshape", "Sin", "Slice", "Sqrt", "Squeeze", "Sub", "Tile", "Transpose", "Unsqueeze", "Where",
}


def tidy(model, weights) -> tuple:
    graph = model.graph
    start = len(graph.node)
    _unsqueezed_weights(graph, set(weights))
    for _ in range(4):  # each pass makes more shapes constant, which lets the next fold more
        before = len(graph.node)
        _fold(model, graph)
        _drop_unused(graph)
        if len(graph.node) == before:
            break
    del graph.value_info[:]
    return start, len(graph.node)


def _attr(node, name, default=None):
    return next((helper.get_attribute_value(a) for a in node.attribute if a.name == name), default)


def _rename_uses(graph, renames):
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in renames:
                node.input[i] = renames[name]


def _constants(graph) -> dict:
    return {n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in graph.node if n.op_type == "Constant" and n.attribute[0].name == "value"}


def _unsqueezed_weights(graph, weights):
    # A weight whose only use is an Unsqueeze (GroupNorm's scale and shift) becomes a graph input of the unsqueezed
    # shape instead. Same values in the same order, and TensorRT only checks a refitted weight's element count, so
    # the checkpoint's tensor still refits it as is.
    inputs = {i.name: i for i in graph.input}
    consts = _constants(graph)
    while True:
        users = collections.Counter(name for n in graph.node for name in n.input)
        gone, renames = [], {}
        for node in graph.node:
            name = node.input[0] if node.input else None
            if name not in weights or users[name] != 1:
                continue
            if node.op_type != "Unsqueeze" or len(node.input) != 2 or node.input[1] not in consts:
                continue
            dims = inputs[name].type.tensor_type.shape.dim
            shape = [d.dim_value for d in dims]
            rank = len(shape) + consts[node.input[1]].size
            axes = sorted(int(a) % rank for a in consts[node.input[1]].reshape(-1))
            new = list(shape)
            for a in axes:
                new.insert(a, 1)
            for d in list(dims):
                dims.remove(d)
            for n in new:
                dims.add().dim_value = n
            renames[node.output[0]] = name
            gone.append(node)
        if not gone:
            return
        _rename_uses(graph, renames)
        for node in gone:
            graph.node.remove(node)


def _dims(model) -> dict:
    try:
        inferred = onnx.shape_inference.infer_shapes(model, data_prop=True).graph
    except Exception:
        return {}
    out = {}
    for vi in (*inferred.input, *inferred.value_info, *inferred.output):
        t = vi.type.tensor_type
        if t.HasField("shape"):
            out[vi.name] = [d.dim_value if d.HasField("dim_value") else (d.dim_param or None) for d in t.shape.dim]
    return out


def _evaluate(node, values):
    from onnx.reference import ReferenceEvaluator

    return ReferenceEvaluator(node, opsets={"": 17}).run(None, {name: values[name] for name in node.input if name})


def _fold(model, graph):
    dims, values = _dims(model), _constants(graph)
    # shape_of: Shape output -> its input. scalar_size_of / size_of: a dynamic dim as a scalar / 1-element list ->
    # (tensor, axis). parts_of: Concat output -> its parts, ints or (tensor, axis).
    shape_of, scalar_size_of, size_of, parts_of = {}, {}, {}, {}
    folded, new_nodes = set(), []
    for node in graph.node:
        out = node.output[0] if node.output else None
        if node.op_type == "Constant" or out is None:
            continue
        if node.op_type == "Shape" and not node.attribute:
            d = dims.get(node.input[0])
            if d is not None and all(isinstance(x, int) for x in d):
                values[out] = np.array(d, dtype=np.int64)
                folded.add(out)
            else:
                shape_of[out] = node.input[0]
            continue
        if node.op_type == "Gather" and node.input[0] in shape_of and node.input[1] in values and not _attr(node, "axis", 0):
            source, index = shape_of[node.input[0]], values[node.input[1]]
            d = dims.get(source)
            if d and index.ndim <= 1 and all(-len(d) <= int(i) < len(d) for i in index.reshape(-1)):
                picked = [d[int(i)] for i in index.reshape(-1)]
                if all(isinstance(x, int) for x in picked):
                    values[out] = np.array(picked, dtype=np.int64).reshape(index.shape)
                    folded.add(out)
                elif index.ndim == 1 and index.size == 1:
                    size_of[out] = (source, int(index[0]) % len(d))
                elif index.ndim == 0:
                    scalar_size_of[out] = (source, int(index) % len(d))
            continue
        if node.op_type == "Unsqueeze" and node.input[0] in scalar_size_of and node.input[1] in values and values[node.input[1]].reshape(-1).tolist() == [0]:
            size_of[out] = scalar_size_of[node.input[0]]
            continue
        if node.op_type == "Concat" and _attr(node, "axis") == 0 and not all(i in values for i in node.input):
            parts = []
            for i in node.input:
                if i in values and values[i].ndim == 1:
                    parts.extend(int(v) for v in values[i])
                elif i in size_of:
                    parts.append(size_of[i])
                else:
                    break
            else:
                parts_of[out] = parts
            continue
        if node.op_type == "Reshape" and node.input[1] in parts_of and not _attr(node, "allowzero", 0):
            target = _keep_dynamic(parts_of[node.input[1]], dims.get(node.input[0]), dims)
            if target is not None:
                name = node.input[1] + "_folded"
                if name not in values:
                    values[name] = target
                    new_nodes.append(helper.make_node("Constant", [], [name], value=numpy_helper.from_array(target, name)))
                node.input[1] = name
            continue
        if node.op_type in _FOLDABLE and all(not i or i in values for i in node.input):
            try:
                results = _evaluate(node, values)
            except Exception:
                continue
            if all(np.asarray(r).size <= MAX_FOLDED for r in results):
                for name, r in zip(node.output, results):
                    values[name] = np.asarray(r)
                    folded.add(name)
    kept = []
    for node in graph.node:
        if node.op_type != "Constant" and node.output and all(o in folded for o in node.output):
            kept.extend(helper.make_node("Constant", [], [o], value=numpy_helper.from_array(values[o], o)) for o in node.output)
        else:
            kept.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes + kept)


def _keep_dynamic(parts, data_dims, dims):
    # A constant Reshape target, possible when each dynamic dim equals the input's dim at the same position: ONNX
    # Reshape reads 0 as "keep the input's dim".
    if data_dims is None:
        return None
    shape = []
    for position, part in enumerate(parts):
        if isinstance(part, int):
            shape.append(part)
            continue
        tensor, axis = part
        size = (dims.get(tensor) or [None] * (axis + 1))[axis]
        if size is None or position >= len(data_dims) or data_dims[position] != size:
            return None
        shape.append(0)
    return np.array(shape, dtype=np.int64)


def _drop_unused(graph):
    needed = {o.name for o in graph.output}
    kept = []
    for node in reversed(graph.node):
        if any(o in needed for o in node.output):
            kept.append(node)
            needed.update(i for i in node.input if i)
    del graph.node[:]
    graph.node.extend(reversed(kept))
