"""Bounded ORT oracle for the Gemma vision encoder with synthetic inputs.

python tool/gemma_vision_parity.py MODEL_DIRECTORY OUTPUT_DIRECTORY [--grid 4]
Runs a small synthetic patch grid through native ORT and records the output
tensors in float32 .bin files plus a reference.json the Dart harness compares
against. Only the vision_encoder graph is exercised.
"""
import argparse
import json
import os
import atexit
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument('model', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--grid', type=int, default=4)
parser.add_argument('--probe', type=str, default='',
                    help='comma-separated intermediate tensor names to capture')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

part = 'vision_encoder'
model = onnx.load(args.model / part / 'model.onnx', load_external_data=False)
folder = args.output / part
folder.mkdir(exist_ok=True)
for t in model.graph.initializer:
    for e in t.external_data:
        if e.key == 'location':
            link = folder / e.value
            if not link.exists():
                os.link(args.model / part / e.value, link)
                atexit.register(lambda p=link: p.unlink(missing_ok=True))
path = folder / 'reference-model.onnx'
probes = [s for s in args.probe.split(',') if s]
probe_out_names = {}
for i, name in enumerate(probes):
    node = model.graph.node.add()
    node.name = f'probe_{i}'
    node.op_type = 'Identity'
    oname = f'probe_out_{i}'
    node.input.append(name)
    node.output.append(oname)
    out = model.graph.output.add()
    out.name = oname
    probe_out_names[oname] = name
onnx.save(model, path)
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
opts.intra_op_num_threads = 4
opts.log_severity_level = 3
sess = ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])

H = W = args.grid
rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
pos = np.stack([rows.ravel(), cols.ravel()], axis=1)[None, :, :].astype(np.int64)
pixels = np.random.default_rng(7).normal(size=(1, H * W, 768)).astype(np.float16)

values = dict(zip([o.name for o in sess.get_outputs()],
                  sess.run(None, {'pixel_values': pixels, 'pixel_position_ids': pos})))
records = {}
for name, value in values.items():
    dtype = 'int64' if value.dtype == np.int64 else 'float32'
    binf = f'vision-{name}.bin'
    value.astype('<i8' if dtype == 'int64' else '<f4').tofile(folder / binf)
    key = probe_out_names.get(name, name)
    records[key] = {'file': binf, 'dtype': dtype, 'shape': list(value.shape)}
info = {'ort': ort.__version__, 'grid': args.grid,
        'outputs': records}
pixels.astype('<f2').tofile(folder / 'pixel-values.bin')
pos.astype('<i8').tofile(folder / 'pixel-position-ids.bin')
(args.output / 'reference.json').write_text(
    json.dumps(info, indent=2), encoding='utf8')
print('grid', args.grid, 'outputs', records.keys())