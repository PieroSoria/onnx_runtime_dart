import onnx, collections
m = onnx.load(r'C:\Users\PC\Desktop\code\plugin\webpt_workspace\webpt_converter\assets\models\lama\lama_manga_large_1024.onnx')
try:
    m = onnx.shape_inference.infer_shapes(m)
except Exception as e:
    print('shape inference failed:', e)
inits = {t.name: t for t in m.graph.initializer}
vi = {v.name: v for v in m.graph.value_info}
vi.update({v.name: v for v in m.graph.output}); vi.update({v.name: v for v in m.graph.input})
elig = []
for n in m.graph.node:
    if n.op_type != 'Conv': continue
    w = inits[n.input[1]]
    dims = [int(d) for d in w.dims]
    a = {x.name: x for x in n.attribute}
    pads = list(a['pads'].ints) if 'pads' in a else [0,0,0,0]
    st = list(a['strides'].ints) if 'strides' in a else [1,1]
    dl = list(a['dilations'].ints) if 'dilations' in a else [1,1]
    g = a['group'].i if 'group' in a else 1
    ok = len(dims)==4 and dims[2]==3 and dims[3]==3 and g==1 and st==[1,1] and dl==[1,1] and pads in ([0,0,0,0],[1,1,1,1])
    if ok:
        oshape = [d.dim_value for d in vi[n.output[0]].type.tensor_type.shape.dim] if n.output[0] in vi else []
        elig.append((n.name, dims, pads, oshape, dims[0]*dims[1]))
print('conv nodes:', sum(1 for n in m.graph.node if n.op_type=='Conv'), 'eligible 3x3:', len(elig))
tot_u = sum(mc*16*4 for *_ , mc in elig)
print('distinct weights:', len({n for n,_,_,_,_ in elig}), 'total U bytes: %.1f MB' % (tot_u/1e6))
print('U weighted by spatial (units of MB*oh*ow):')
import math
for name, dims, pads, oshape, mc in sorted(elig, key=lambda e: -e[4]*(e[3][2]*e[3][3] if len(e[3])==4 else 0))[:20]:
    print(' ', name, dims, pads, oshape, 'U=%.2fMB' % (mc*16*4/1e6), 'spatial', oshape[2:] if len(oshape)==4 else '?')



