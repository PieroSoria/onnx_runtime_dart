"""Bounded ORT oracle for the Gemma split export; never expands embedding tables.

python tool/gemma_parity.py MODEL_DIRECTORY OUTPUT_DIRECTORY [--steps 3]
Large Gather-only tables are evaluated by reading the selected rows via memmap,
then replaced by constants for the fixed token inputs of each oracle step. Dense
decoder weights are loaded by native ORT. Original model files are never edited.
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

_ONNX_TO_NP = {1:np.float32,2:np.uint8,3:np.int8,6:np.int32,7:np.int64,9:np.bool_,10:np.float16,11:np.float64}

def _const_value(model, by_out, name):
    node=by_out.get(name)
    if node is not None and node.op_type=='Constant':
        for a in node.attribute:
            if a.name=='value_int': return np.array(a.i)
            if a.name=='value_float': return np.array(a.f)
            if a.name=='value_ints': return np.array(list(a.ints))
            if a.name=='value_floats': return np.array(list(a.floats))
            if a.name=='value': return numpy_helper.to_array(a.t)
    for t in model.graph.initializer:
        if t.name==name:
            return numpy_helper.to_array(t)
    return None

def _eval_index(model, by_out, target, feeds, dtype_of):
    """Evaluate the static subgraph producing ``target`` over tensor inputs."""
    allowed={'Where','Equal','Add','Sub','Mul','Div','Neg','Not','And','Or','Xor',
             'Greater','Less','GreaterOrEqual','LessOrEqual','CumSum','Clip','Cast',
             'Unsqueeze','Squeeze','Concat','Reshape','Gather','Constant','Identity',
             'Floor','Ceil','Round','Abs','Sign','Mod','Pow','Min','Max','Transpose'}
    collect=set(); order=[]
    def visit(name):
        if name in by_out and name not in collect and by_out[name].op_type in allowed:
            node=by_out[name]
            collect.add(name)
            for i in node.input:
                if i and i not in feeds: visit(i)
            order.append(node)
    visit(target)
    env=dict(feeds)
    for node in order:
        args=[env[i] for i in node.input] if node.input else []
        op=node.op_type
        attr=lambda n,fb: next((a for a in node.attribute if a.name==n),None)
        if op=='Constant':
            v=_const_value(model,by_out,node.output[0])
            if v is None: raise RuntimeError(f'Unresolvable Constant feeding {target}')
            env[node.output[0]]=v
        elif op=='Identity': env[node.output[0]]=args[0]
        elif op=='Where': env[node.output[0]]=np.where(args[0],args[1],args[2])
        elif op=='Equal': env[node.output[0]]=np.equal(args[0],args[1])
        elif op in ('Add','Sub','Mul','Div','Mod','Pow','Min','Max'):
            env[node.output[0]]=getattr(np,op.lower())(*args)
        elif op=='Neg': env[node.output[0]]=-args[0]
        elif op=='Not': env[node.output[0]]=np.logical_not(args[0])
        elif op in ('And','Or','Xor'):
            env[node.output[0]]={'And':np.logical_and,'Or':np.logical_or,'Xor':np.logical_xor}[op](*args)
        elif op=='Greater': env[node.output[0]]=np.greater(*args)
        elif op=='Less': env[node.output[0]]=np.less(*args)
        elif op=='GreaterOrEqual': env[node.output[0]]=np.greater_equal(*args)
        elif op=='LessOrEqual': env[node.output[0]]=np.less_equal(*args)
        elif op=='CumSum':
            env[node.output[0]]=np.cumsum(args[0],axis=int(args[1].item()))
        elif op=='Clip':
            lo=args[1].item() if len(args)>1 else None
            hi=args[2].item() if len(args)>2 else None
            env[node.output[0]]=np.clip(args[0],lo,hi)
        elif op=='Cast':
            to=attr('to',None).i
            env[node.output[0]]=args[0].astype(_ONNX_TO_NP[to])
        elif op=='Unsqueeze':
            env[node.output[0]]=np.expand_dims(args[0],tuple(int(x) for x in args[1].reshape(-1)))
        elif op=='Squeeze':
            axes=[int(x) for x in args[1].reshape(-1)] if len(args)>1 else None
            env[node.output[0]]=np.squeeze(args[0],axis=axes)
        elif op=='Concat':
            ax=getattr(attr('axis',None),'i',0)
            env[node.output[0]]=np.concatenate(args,axis=ax)
        elif op=='Reshape':
            env[node.output[0]]=np.reshape(args[0],tuple(int(x) for x in args[1].reshape(-1)))
        elif op=='Gather':
            ax=getattr(attr('axis',None),'i',0)
            env[node.output[0]]=np.take(args[0],args[1],axis=ax)
        elif op=='Transpose':
            perm=None
            a=attr('perm',None)
            if a is not None: perm=[int(x) for x in a.ints]
            env[node.output[0]]=np.transpose(args[0],perm)
        elif op=='Floor': env[node.output[0]]=np.floor(args[0])
        elif op=='Ceil': env[node.output[0]]=np.ceil(args[0])
        elif op=='Round': env[node.output[0]]=np.round(args[0])
        elif op=='Abs': env[node.output[0]]=np.abs(args[0])
        elif op=='Sign': env[node.output[0]]=np.sign(args[0])
        else: raise RuntimeError(f'Unsupported index-chain operator {op}')
    if target not in env: raise RuntimeError(f'Index {target} is not statically evaluated')
    return np.ascontiguousarray(env[target])

parser = argparse.ArgumentParser()
parser.add_argument('model',type=Path)
parser.add_argument('output',type=Path)
parser.add_argument('--steps',type=int,default=3)
args = parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)

def graph(part):
    m=onnx.load(args.model/part/'model.onnx',load_external_data=False)
    return m

def session(model,part='decoder'):
    options=ort.SessionOptions()
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads=4
    options.log_severity_level=3
    folder=args.output/part
    folder.mkdir(exist_ok=True)
    for t in model.graph.initializer:
        for e in t.external_data:
            if e.key=='location':
                link=folder/e.value
                if not link.exists():
                    os.link(args.model/part/e.value,link)
                    atexit.register(lambda p=link: p.unlink(missing_ok=True))
    path=folder/'reference-model.onnx'
    onnx.save(model,path)
    return ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])

def embeddings(ids):
    m=graph('embedding')
    by_out={o:n for n in m.graph.node for o in n.output}
    graph_inputs={i.name for i in m.graph.input}
    feeds={'input_ids':np.ascontiguousarray(ids)}
    input_types={i.name:i.type.tensor_type.elem_type for i in m.graph.input}
    def feed(name):
        if name in feeds: return feeds[name]
        if name in input_types and _ONNX_TO_NP.get(input_types[name]):
            feeds[name]=np.empty((0,1536),np.float16) if name.endswith('_features') else np.empty(0,_ONNX_TO_NP[input_types[name]])
            return feeds[name]
        return None
    remove=[]
    for t in m.graph.initializer:
        if np.prod(t.dims,dtype=np.int64) < 20_000_000: continue
        uses=[n for n in m.graph.node if t.name in n.input]
        for n in uses:
            axis=getattr(next((a for a in n.attribute if a.name=='axis'),None),'i',0) or 0
            if not (n.op_type=='Gather' and n.input[0]==t.name and axis==0):
                raise RuntimeError('Unexpected large embedding consumer: '+t.name)
        fields={e.key:e.value for e in t.external_data}
        rows=None
        for n in uses:
            index=n.input[1]
            if index in graph_inputs:
                rows=feed(index)
            else:
                rows=_eval_index(m,by_out,index,feeds,input_types)
            if rows is None or rows.shape!=ids.shape:
                raise RuntimeError(f'Large table {t.name} gathered with non-input index')
        table=np.memmap(args.model/'embedding'/fields['location'],mode='r',dtype=np.float16,offset=int(fields.get('offset',0)),shape=tuple(t.dims))
        selected=np.ascontiguousarray(np.array(table[rows],copy=True))
        for n in uses:
            n.CopyFrom(helper.make_node('Constant',[],list(n.output),name=n.name,value=numpy_helper.from_array(selected)))
        del table
        remove.append(t.name)
    kept=[t for t in m.graph.initializer if t.name not in remove]
    del m.graph.initializer[:]
    m.graph.initializer.extend(kept)
    sess=session(m,'embedding')
    result=sess.run(None,{'input_ids':ids,'image_features':np.empty((0,1536),np.float16),'audio_features':np.empty((0,1536),np.float16)})
    values=dict(zip([o.name for o in sess.get_outputs()],result))
    del sess
    return values

def save(value, name):
    # Store float32 so the Dart reader doesn't depend on NumPy's npy format.
    dtype='int64' if value.dtype==np.int64 else 'float32'
    path=args.output/(name+'.bin')
    value.astype('<i8' if dtype=='int64' else '<f4').tofile(path)
    return {'file':path.name,'dtype':dtype,'shape':list(value.shape)}

tokenizer=Tokenizer.from_file(str(args.model/'tokenizer.json'))
ids=np.array([tokenizer.encode('Hello',add_special_tokens=True).ids],np.int64)
print('prompt IDs:',ids.tolist(),flush=True)
decoder=session(graph('decoder'))
cache={i.name:np.empty((1,i.shape[1],0,i.shape[3]),np.float16) for i in decoder.get_inputs() if i.name.startswith('past_key_values.')}
past=0
records=[]
generated=[]
for step in range(args.steps):
    e=embeddings(ids)
    sequence=ids.shape[1]
    feeds={**e,**cache,'position_ids':np.arange(past,past+sequence,dtype=np.int64)[None,:], 'attention_mask':np.ones((1,past+sequence),np.int64)}
    values=dict(zip([o.name for o in decoder.get_outputs()],decoder.run(None,feeds)))
    logits=values['logits']
    token=int(np.argmax(logits[0,-1]))
    generated.append(token)
    records.append({'ids':ids.tolist(),'embedding':{k:save(v,f'{step}-embed-{k}') for k,v in e.items()},'logits':save(logits,f'{step}-logits'),'token':token})
    print('step',step,'token',token,'text',repr(tokenizer.decode(generated)),flush=True)
    cache={name:values[name.replace('past_key_values.','present.')] for name in cache}
    past+=sequence
    ids=np.array([[token]],np.int64)
    gc.collect()
(args.output/'reference.json').write_text(json.dumps({'ort':ort.__version__,'steps':records,'generated':generated,'text':tokenizer.decode(generated)},indent=2),encoding='utf8')
