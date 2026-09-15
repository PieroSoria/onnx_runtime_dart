"""Small deterministic ONNX Runtime CPU oracles for standard transformer ops.

Requires onnx==1.22.0, onnxruntime==1.29.0, numpy. Does not download models.
"""
import json
import base64
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1] / 'test/fixtures'
rng = np.random.default_rng(731)
portable_cases = {}

def random(shape):
    return rng.normal(size=shape).astype(np.float32)

def emit(name, op, inputs, attributes=None, outputs=None, domain='', initializers=None, opset=23):
    outputs = outputs or ['Y']
    initializers = initializers or {}
    feeds = {k: v for k, v in inputs.items() if v is not None}
    node = helper.make_node(op, [k if v is not None else '' for k, v in inputs.items()], outputs, domain=domain, **(attributes or {}))
    graph = helper.make_graph([node], name,
        [helper.make_tensor_value_info(k, helper.np_dtype_to_tensor_dtype(v.dtype), v.shape) for k, v in feeds.items() if k not in initializers],
        [helper.make_tensor_value_info(k, helper.np_dtype_to_tensor_dtype(next(iter(feeds.values())).dtype), None) for k in outputs],
        [numpy_helper.from_array(v, k) for k, v in initializers.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', opset)] + ([helper.make_opsetid(domain, 1)] if domain else []), ir_version=10)
    session = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    feeds = {k: v for k, v in feeds.items() if k not in initializers}
    expected = dict(zip(outputs, session.run(None, feeds)))
    del model.graph.output[:]
    model.graph.output.extend([helper.make_tensor_value_info(k, helper.np_dtype_to_tensor_dtype(v.dtype), v.shape) for k, v in expected.items()])
    onnx.checker.check_model(model)
    def tensor(v):
        return {'dtype': 'int64' if v.dtype == np.bool_ else str(v.dtype), 'shape': list(v.shape), 'data': v.astype(np.int64).ravel().tolist() if v.dtype == np.bool_ else v.ravel().tolist()}
    folder = ROOT / name
    folder.mkdir(exist_ok=True)
    (folder / 'model.onnx').write_bytes(model.SerializeToString())
    (folder / 'case.json').write_text(json.dumps({'inputs': {k: tensor(v) for k, v in feeds.items()}, 'expected': {k: tensor(v) for k, v in expected.items()}}, allow_nan=True))
    portable_cases[name] = {'model': base64.b64encode(model.SerializeToString()).decode('ascii'), 'case': json.loads((folder / 'case.json').read_text())}
    print(name)

for axis in [-1, 1, 0]:
    emit(f'standard_rms_axis_{axis}', 'RMSNormalization', {'X': random((2,3,4)), 'scale': random((1,4))}, {'axis': axis})
for interleaved in [0,1]:
    x, cos, sin = random((2,3,2,8)), random((7,2)), random((7,2))
    emit(f'standard_rotary_partial_{interleaved}', 'RotaryEmbedding', {'X': x, 'cos': cos, 'sin': sin, 'positions': np.array([[1,6],[3,0]], np.int64)}, {'interleaved': interleaved, 'rotary_embedding_dim': 4})
emit('standard_rotary_preindexed', 'RotaryEmbedding', {'X': random((2,3,16)), 'cos': random((2,3,4)), 'sin': random((2,3,4))}, {'num_heads': 2})
for mode in range(4):
    emit(f'standard_attention_debug_{mode}', 'Attention', {'Q': random((2,4,3,4)), 'K': random((2,2,5,4)), 'V': random((2,2,5,6)), 'mask': random((3,5)) * .1}, {'scale': .3, 'softcap': 2.0, 'qk_matmul_output_mode': mode}, ['Y','PK','PV','debug'])
emit('standard_attention_causal_cache', 'Attention', {'Q': random((1,4,2,4)), 'K': random((1,2,2,4)), 'V': random((1,2,2,3)), 'mask': None, 'pastK': random((1,2,5,4)), 'pastV': random((1,2,5,3))}, {'is_causal': 1}, ['Y','PK','PV'])
emit('standard_attention_3d', 'Attention', {'Q': random((2,3,16)), 'K': random((2,5,8)), 'V': random((2,5,12)), 'mask': np.array([[True,False,True,False,True]]*3)}, {'q_num_heads': 4, 'kv_num_heads': 2})
for rows in [1,3]:
    n,k,block=5,35,16
    b=rng.integers(0,256,(n,3,8),dtype=np.uint8)
    scales=random((n,3)) * .1
    for packed in [False,True]:
        zp=rng.integers(0,256,(n,2),dtype=np.uint8) if packed else random((n,3)) + 8
        bias=random((n,))
        weights={'B': b, 'scales': scales, 'zp': zp, 'bias': bias}
        emit(f'matmulnbits_bias_{rows}_{packed}', 'MatMulNBits', {'A': random((rows,k)), 'B': b, 'scales': scales, 'zp': zp, 'g_idx': None, 'bias': bias}, {'K':k,'N':n,'bits':4,'block_size':block}, domain='com.microsoft', initializers=weights)

emit('standard_rms_fp16', 'RMSNormalization', {'X':random((2,3,4)).astype(np.float16),'scale':random((4,)).astype(np.float16)}, {'epsilon':1e-6}, opset=24)
emit('standard_rotary_fp16', 'RotaryEmbedding', {'X':random((2,3,16)).astype(np.float16),'cos':random((2,3,4)).astype(np.float16),'sin':random((2,3,4)).astype(np.float16)}, {'num_heads':2}, opset=24)
emit('standard_attention24_fp16', 'Attention', {'Q':random((1,4,2,4)).astype(np.float16),'K':random((1,1,2,4)).astype(np.float16),'V':random((1,1,2,4)).astype(np.float16),'mask':None,'pastK':random((1,1,3,4)).astype(np.float16),'pastV':random((1,1,3,4)).astype(np.float16)}, {'is_causal':1,'scale':1.0}, ['Y','PK','PV'], opset=24)

# Embedded data lets browser tests run the same native-ORT oracles without IO.
(ROOT.parent / 'transformer_oracle_data.dart').write_text(
    '// Generated by tool/gen_transformer_fixtures.py (ORT 1.29.0 CPU).\n'
    "const transformerOraclesJson = r'''" + json.dumps(portable_cases, separators=(',', ':')) + "''';\n")
