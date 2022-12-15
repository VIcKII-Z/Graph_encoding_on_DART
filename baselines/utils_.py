from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter

# from evaluate import load
def zero_shot(model, pred_file, out_file, prompt=''):
    tokenizer = T5Tokenizer.from_pretrained(model)
    model = T5ForConditionalGeneration.from_pretrained(model)

    # training
    with open(pred_file, 'r') as f:
        samples = f.readlines()
    out = []
    for sample in tqdm(samples):
        # inference
        input_ids = tokenizer(
            prompt + sample, return_tensors="pt"
        ).input_ids  # Batch size 1
        outputs = model.generate(input_ids)
        out.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    out_file = out_file+'with_prompt' if prompt!='' else out_file
    with open(out_file, 'w') as f:
        for o in out:
            f.write(o+'\n')


def bleu(ref, pred):
    with open(ref, 'r') as fr:
        ref = fr.readlines()
    with open(pred, 'r') as fp:
        pred = fp.readlines()
    sbleu = load('sacrebleu')
    print(sbleu.compute(pred, ref))


def write(file, idx, data):
    with open(file, 'w') as f:
        for i in idx:
            f.write(data[i])


def data_statitics(sent_file, graph_file, reentry, pred_file):
    with open(sent_file, 'r') as f:
        lines_s = f.readlines()
    total_words = 0
    n = len(lines_s)
    max_word = 0
    for line in lines_s:
        line = line.split(' ')
        total_words += len(line)
        max_word =  max(max_word, len(line))
    print('avg words/samle: ', total_words/n, 'max words in a sentence: ', max_word)
    with open(graph_file, 'r') as f:
        lines = f.readlines()
    all_triples = []
    nn = len(lines)
    assert n == nn, 'n!=nn'
    small_gs = []
    mid_gs = []
    mid_gss = []
    large_gs = []
    for idx, line in enumerate(lines):
        line = line.split(' ')
        n_edge = len(line)
        all_triples.append(n_edge)
        # if n_edge < 10:
        #     small_gs.append(idx)
        # elif n_edge < 20:
        #     mid_gs.append(idx)
        # # elif n_edge <40:
        # #     mid_gss.append(idx)
        # else:
        #     large_gs.append(idx)
    # with open(pred_file, 'r') as f:
    #     lines_p = f.readlines()
    # write('test_small_tgt', small_gs, lines_s)
    # write('test_mid_tgt', mid_gs, lines_s)
    # # write('test_mid1_tgt', mid_gss, lines_s)

    # write('test_large_tgt', large_gs, lines_s)
    
    # write('test_small_pred', small_gs, lines_p)
    # write('test_mid_pred', mid_gs, lines_p)
    # # write('test_mid1_pred', mid_gss, lines_p)
    # write('test_large_pred', large_gs, lines_p)
    
    # print('# all nodes', len(all_triples))
    # print('# small, mid, large grapsh: ', len(small_gs), 
    # #len(mid_gs), 
    # #len(mid_gss),
    #  len(large_gs))
    # print('percentile', np.percentile(all_triples, (50)))
        
    with open(reentry, 'r') as f:
        lines = f.readlines()
    entries = []
    for idx, line in enumerate(lines):
        line = int(line.strip()) 
        entries.append(line)
        if line < 3:
            small_gs.append(idx)
        elif line < 4:
            mid_gs.append(idx)
        # elif n_edge <40:
        #     mid_gss.append(idx)
        else:
            large_gs.append(idx)
    with open(pred_file, 'r') as f:
        lines_p = f.readlines()
    write('test_small_tgt', small_gs, lines_s)
    write('test_mid_tgt', mid_gs, lines_s)
    # write('test_mid1_tgt', mid_gss, lines_s)

    write('test_large_tgt', large_gs, lines_s)
    
    write('test_small_pred', small_gs, lines_p)
    write('test_mid_pred', mid_gs, lines_p)
    # write('test_mid1_pred', mid_gss, lines_p)
    write('test_large_pred', large_gs, lines_p)
    
    print('# all nodes', len(entries))
    print('# small, mid, large grapsh: ', len(small_gs), 
    # len(mid_gs), 
    #len(mid_gss),
     len(large_gs))
    print('percentile', np.percentile(entries, (50)))
    return entries, all_triples




def plot(data, label):
    x, y = zip(*Counter(data).items())
    print(x,y)
    plt.figure(1)   
                                                                                                                                                                                                                                                        
    # prep axes                                                                                                                      
    plt.xlabel(label)                                                                                                             
    # plt.xscale('log')                                                                                                                
    plt.xlim(1, max(x))  
                                                                                                            
    plt.ylabel('frequency')                                                                                                          
    # plt.yscale('log')                                                                                                                
    plt.scatter(x, y, marker='.')                                                                                                    
    plt.savefig(f'{label}.png')

def plot_(data):
    x, y = data
    # Sequences
  

    fig, ax = plt.subplots()

    # Plot linear sequence, and set tick labels to the same color
    ax.plot(x, color='red', label = 'RGCN')
    ax.tick_params(axis='y', labelcolor='red')

    # Generate a new Axes instance, on the twin-X axes (same position)
    ax2 = ax.twinx()

    # Plot exponential sequence, set scale to logarithmic and change tick color
    ax2.plot(y, color='green', label='fine-tune')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper left')
    ax.legend(loc='upper right')
    plt.xlabel('graph size')
    plt.ylabel('BLEU')
    plt.xticks([0, 1, 2, 3], ['tiny', 'small', 'medium', 'large'])
    plt.savefig('size_bleu.png')

        

        
    
pred_file = '/home/lily/wz336/StructAdapt/data/dart_exp0/test_seen.source'
out_file = '/home/lily/wz336/StructAdapt/zero_shot'

ref = '/home/lily/wz336/StructAdapt/data/dart_exp0/test_seen.target'
pred = out_file
# bleu(ref, pred)
# zero_shot('t5-base', pred_file, out_file, prompt='summarize the triples: ')
graph = '/home/lily/wz336/StructAdapt/data/processed_dart/test.graph.tok'
entry = '/home/lily/wz336/StructAdapt/data/processed_dart/test.reentrances'
pred_file = '/home/lily/wz336/StructAdapt/outputs/exp-6270/val_outputs/pred_-1 copy.txt'
# pred_file = '/home/lily/wz336/StructAdapt/evaluation/evaluation/dart-outputs/t5-base.txt'
# ref = '/home/lily/wz336/StructAdapt/evaluation/evaluation/ref.toknized'
# lines, tris = data_statitics(ref, graph, entry, pred_file)
# plot(tris, 'sum_degree')
# plot(lines, 'max_degree')
# data = [[20.47, 24.40, 23.84, 26.87],[38.17, 32.63, 32.73, 37.58]]
# plot_(data)