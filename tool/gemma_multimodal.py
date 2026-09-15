"""Bounded ORT oracle for the Gemma multimodal flow through all three graphs.

python tool/gemma_multimodal.py MODEL_DIRECTORY OUTPUT_DIRECTORY
  [--image path] [--grid 4] [--steps 3]

A real image is resized to (grid*16)^2, split into 16x16 RGB patches,
normalized to [0,1] and pushed as pixel_values through the vision encoder;
its image_features are spliced into the embedding input_ids after N copies of
the image token 258880; the decoder then auto-regresses `--steps` tokens with
persistent past. All outputs are recorded as float32 .bin files plus a
reference.json the Dart harness compares against. Embedding tables are never
expanded (reuses the static index-chain substitution from gemma_parity.py).
"""
import argparse
import gc
import json
import os
import atexit
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
from tokenizers import Tokenizer

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

parser = argparse.ArgumentParser()
parser.add_argument('model', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--image', type=Path,
                    default=Path(r'C:\Users\PC\Desktop\code\plugin\webpt_workspace\webpt_converter\bin\test.webp'))
parser.add_argument('--grid', type=int, default=4)
parser.add_argument('--steps', type=int, default=3)
parser.add_argument('--prompt', type=str, default='Describe this image in detail.')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

_ONNX_TO_NP = {1: np.float32, 2: np.uint8, 3: np.int8, 6: np.int32, 7: np.int64,
               9: np.bool_, 10: np.float16, 11: np.float64}
IMAGE_TOKEN = 258880
IMAGE_END = 258882
NUM_PATCHES = args.grid * args.grid


def _const_value(model, by_out, name):
    node = by_out.get(name)
    if node is not None and node.op_type == 'Constant':
        for a in node.attribute:
            if a.name == 'value_int': return np.array(a.i)
            if a.name == 'value_float': return np.array(a.f)
            if a.name == 'value_ints': return np.array(list(a.ints))
            if a.name == 'value_floats': return np.array(list(a.floats))
            if a.name == 'value': return numpy_helper.to_array(a.t)
    for t in model.graph.initializer:
        if t.name == name:
            return numpy_helper.to_array(t)
    return None


def _eval_index(model, by_out, target, feeds):
    allowed = {'Where', 'Equal', 'Add', 'Sub', 'Mul', 'Div', 'Neg', 'Not', 'And',
               'Or', 'Xor', 'Greater', 'Less', 'GreaterOrEqual', 'LessOrEqual',
               'CumSum', 'Clip', 'Cast', 'Unsqueeze', 'Squeeze', 'Concat',
               'Reshape', 'Gather', 'Constant', 'Identity', 'Floor', 'Ceil',
               'Round', 'Abs', 'Sign', 'Mod', 'Pow', 'Min', 'Max', 'Transpose'}
    collect = set(); order = []
    def visit(name):
        if name in by_out and name not in collect and by_out[name].op_type in allowed:
            node = by_out[name]
            collect.add(name)
            for i in node.input:
                if i and i not in feeds:
                    visit(i)
            order.append(node)
    visit(target)
    env = dict(feeds)
    for node in order:
        args_ = [env[i] for i in node.input] if node.input else []
        op = node.op_type
        attr = lambda n, fb: next((a for a in node.attribute if a.name == n), None)
        if op == 'Constant':
            v = _const_value(model, by_out, node.output[0])
            if v is None: raise RuntimeError(f'Unresolvable Constant feeding {target}')
            env[node.output[0]] = v
        elif op == 'Identity': env[node.output[0]] = args_[0]
        elif op == 'Where': env[node.output[0]] = np.where(args_[0], args_[1], args_[2])
        elif op == 'Equal': env[node.output[0]] = np.equal(args_[0], args_[1])
        elif op in ('Add', 'Sub', 'Mul', 'Div', 'Mod', 'Pow', 'Min', 'Max'):
            env[node.output[0]] = getattr(np, op.lower())(*args_)
        elif op == 'Neg': env[node.output[0]] = -args_[0]
        elif op == 'Not': env[node.output[0]] = np.logical_not(args_[0])
        elif op in ('And', 'Or', 'Xor'):
            env[node.output[0]] = {'And': np.logical_and, 'Or': np.logical_or,
                                   'Xor': np.logical_xor}[op](*args_)
        elif op == 'Greater': env[node.output[0]] = np.greater(*args_)
        elif op == 'Less': env[node.output[0]] = np.less(*args_)
        elif op == 'GreaterOrEqual': env[node.output[0]] = np.greater_equal(*args_)
        elif op == 'LessOrEqual': env[node.output[0]] = np.less_equal(*args_)
        elif op == 'CumSum':
            env[node.output[0]] = np.cumsum(args_[0], axis=int(args_[1].item()))
        elif op == 'Clip':
            lo = args_[1].item() if len(args_) > 1 else None
            hi = args_[2].item() if len(args_) > 2 else None
            env[node.output[0]] = np.clip(args_[0], lo, hi)
        elif op == 'Cast':
            to = attr('to', None).i
            env[node.output[0]] = args_[0].astype(_ONNX_TO_NP[to])
        elif op == 'Unsqueeze':
            env[node.output[0]] = np.expand_dims(
                args_[0], tuple(int(x) for x in args_[1].reshape(-1)))
        elif op == 'Squeeze':
            axes = [int(x) for x in args_[1].reshape(-1)] if len(args_) > 1 else None
            env[node.output[0]] = np.squeeze(args_[0], axis=axes)
        elif op == 'Concat':
            ax = getattr(attr('axis', None), 'i', 0)
            env[node.output[0]] = np.concatenate(args_, axis=ax)
        elif op == 'Reshape':
            env[node.output[0]] = np.reshape(
                args_[0], tuple(int(x) for x in args_[1].reshape(-1)))
        elif op == 'Gather':
            ax = getattr(attr('axis', None), 'i', 0)
            env[node.output[0]] = np.take(args_[0], args_[1], axis=ax)
        elif op == 'Transpose':
            perm = None
            a = attr('perm', None)
            if a is not None: perm = [int(x) for x in a.ints]
            env[node.output[0]] = np.transpose(args_[0], perm)
        elif op == 'Floor': env[node.output[0]] = np.floor(args_[0])
        elif op == 'Ceil': env[node.output[0]] = np.ceil(args_[0])
        elif op == 'Round': env[node.output[0]] = np.round(args_[0])
        elif op == 'Abs': env[node.output[0]] = np.abs(args_[0])
        elif op == 'Sign': env[node.output[0]] = np.sign(args_[0])
        else: raise RuntimeError(f'Unsupported index-chain operator {op}')
    if target not in env:
        raise RuntimeError(f'Index {target} is not statically evaluated')
    return np.ascontiguousarray(env[target])


def graph(part):
    return onnx.load(args.model / part / 'model.onnx', load_external_data=False)


def session(model, part):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 4
    options.log_severity_level = 3
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
    onnx.save(model, path)
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def embeddings(ids, image_features):
    m = graph('embedding')
    by_out = {o: n for n in m.graph.node for o in n.output}
    graph_inputs = {i.name for i in m.graph.input}
    feeds = {'input_ids': np.ascontiguousarray(ids),
             'image_features': np.ascontiguousarray(image_features)}
    input_types = {i.name: i.type.tensor_type.elem_type for i in m.graph.input}
    def feed(name):
        if name in feeds: return feeds[name]
        if name in input_types and _ONNX_TO_NP.get(input_types[name]):
            feeds[name] = (np.empty((0, 1536), np.float16)
                           if name.endswith('_features')
                           else np.empty(0, _ONNX_TO_NP[input_types[name]]))
            return feeds[name]
        return None
    remove = []
    for t in m.graph.initializer:
        if np.prod(t.dims, dtype=np.int64) < 20_000_000: continue
        uses = [n for n in m.graph.node if t.name in n.input]
        for n in uses:
            axis = getattr(next((a for a in n.attribute if a.name == 'axis'), None), 'i', 0) or 0
            if not (n.op_type == 'Gather' and n.input[0] == t.name and axis == 0):
                raise RuntimeError('Unexpected large embedding consumer: ' + t.name)
        fields = {e.key: e.value for e in t.external_data}
        rows = None
        for n in uses:
            index = n.input[1]
            if index in graph_inputs:
                rows = feed(index)
            else:
                rows = _eval_index(m, by_out, index, feeds)
            if rows is None or rows.shape != ids.shape:
                raise RuntimeError(f'Large table {t.name} gathered with non-input index')
        table = np.memmap(args.model / 'embedding' / fields['location'], mode='r',
                          dtype=np.float16, offset=int(fields.get('offset', 0)),
                          shape=tuple(t.dims))
        selected = np.ascontiguousarray(np.array(table[rows], copy=True))
        for n in uses:
            n.CopyFrom(helper.make_node('Constant', [], list(n.output), name=n.name,
                                        value=numpy_helper.from_array(selected)))
        del table
        remove.append(t.name)
    kept = [t for t in m.graph.initializer if t.name not in remove]
    del m.graph.initializer[:]
    m.graph.initializer.extend(kept)
    sess = session(m, 'embedding')
    result = sess.run(None, {'input_ids': ids, 'image_features': image_features,
                             'audio_features': np.empty((0, 1536), np.float16)})
    values = dict(zip([o.name for o in sess.get_outputs()], result))
    del sess
    return values


def save(value, name):
    dtype = 'int64' if value.dtype == np.int64 else 'float32'
    path = args.output / (name + '.bin')
    value.astype('<i8' if dtype == 'int64' else '<f4').tofile(path)
    return {'file': path.name, 'dtype': dtype, 'shape': list(value.shape)}

# --- real image -> pixel_values -------------------------------------------------
if Image is None:
    raise SystemExit('Pillow is required for the multimodal oracle')
with Image.open(args.image) as img:
    img = img.convert('RGB').resize((args.grid * 16, args.grid * 16), Image.BILINEAR)
pixels = (np.array(img, dtype=np.float16).reshape(-1, 16 * 16 * 3) / 255.0)
pixels = pixels.astype(np.float16)[None, :, :]
rows, cols = np.meshgrid(np.arange(args.grid), np.arange(args.grid), indexing='ij')
pos = np.stack([rows.ravel(), cols.ravel()], axis=1)[None, :, :].astype(np.int64)
print('image', args.image, 'grid', args.grid, 'patches', NUM_PATCHES, flush=True)

vision = session(graph('vision_encoder'), 'vision_encoder')
vision_out = dict(zip([o.name for o in vision.get_outputs()],
                      vision.run(None, {'pixel_values': pixels,
                                        'pixel_position_ids': pos})))
iso_f16 = vision_out['image_features']
iso = iso_f16.astype(np.float32)
print('image_features', iso.shape, flush=True)

NUM_IMG = iso.shape[0]
print('image_tokens', NUM_IMG, flush=True)
# --- multimodal prompt -----------------------------------------------------------
tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
text = [t for t in tokenizer.encode(args.prompt, add_special_tokens=False).ids
        if t != 1]
ids = np.array([[2, *([IMAGE_TOKEN] * NUM_IMG), IMAGE_END, *text, 1]], np.int64)
print('prompt', ids.tolist(), flush=True)

decoder = session(graph('decoder'), 'decoder')
cache = {i.name: np.empty((1, i.shape[1], 0, i.shape[3]), np.float16)
         for i in decoder.get_inputs() if i.name.startswith('past_key_values.')}
past = 0
records = []
generated = []
pixels.astype('<f2').tofile(args.output / 'pixel-values.bin')
pos.astype('<i8').tofile(args.output / 'pixel-position-ids.bin')
iso.astype('<f4').tofile(args.output / 'image-features.bin')
for step in range(args.steps):
    e = embeddings(ids, iso_f16)
    sequence = ids.shape[1]
    feeds = {**e, **cache,
             'position_ids': np.arange(past, past + sequence, dtype=np.int64)[None, :],
             'attention_mask': np.ones((1, past + sequence), np.int64)}
    values = dict(zip([o.name for o in decoder.get_outputs()], decoder.run(None, feeds)))
    logits = values['logits']
    token = int(np.argmax(logits[0, -1]))
    generated.append(token)
    records.append({'ids': ids.tolist(),
                    'embedding': {k: save(v, f'{step}-embed-{k}') for k, v in e.items()},
                    'logits': save(logits, f'{step}-logits'), 'token': token})
    print('step', step, 'token', token, 'text', repr(tokenizer.decode(generated)), flush=True)
    cache = {name: values[name.replace('past_key_values.', 'present.')] for name in cache}
    past += sequence
    ids = np.array([[token]], np.int64)
    gc.collect()
(args.output / 'reference.json').write_text(json.dumps(
    {'ort': ort.__version__, 'grid': args.grid, 'prompt': args.prompt,
     'image': str(args.image.name),
     'image-features': save(iso, 'image-features'),
     'pixel-values': {'file': 'pixel-values.bin', 'dtype': 'float16',
                      'shape': list(pixels.shape)},
     'pixel-position-ids': {'file': 'pixel-position-ids.bin', 'dtype': 'int64',
                            'shape': list(pos.shape)},
     'steps': records, 'generated': generated, 'text': tokenizer.decode(generated)},
    indent=2), encoding='utf8')