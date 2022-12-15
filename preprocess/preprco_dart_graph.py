"""
Preprocess Dart to linearlized format

Options:

- linearise: for seq2seq learning
- triples: for graph2seq learning


"""
import re
import argparse
from tqdm import tqdm


def edge_pos_emb(src):
 
    tokens = re.split('<H> | <R> | <T> ', src)[1:]
    graph_triple = set()
    entries = dict()
    reentries = 1
    idx = -1
    
    for i, token in enumerate(tokens):
        
        token = token.strip()
        len_tok = len(token.split(' '))

        if token in entries:
            entries[token] += 1
            reentries = max(reentries, entries[token])
        else:
            entries[token] = 1
        
            
        if i % 3 == 0:
            head = range(idx+1, len_tok+idx+1)
        elif i % 3 == 1:
            tail_stt = head[-1] + len_tok + 1
        else:
            tail = range(tail_stt, tail_stt+len_tok)
            for j in head:
                for k in tail:
                    graph_triple.add((j, k, 'd'))
                    graph_triple.add((k, j, 'r'))
        # print(i, token, graph_triple)
        idx += len_tok
    return sorted(graph_triple), reentries, tokens


def main(args):
    # read source file 
    with open(args.src_path) as f:
        srcs = f.readlines()

    if args.triples_output is not None:
        triple_out = open(args.triples_output, 'w')
        ## multiple entries
        reentrances_file = open(args.triples_output.replace("graph.tok","reentrances"), 'w')

    count_instance = 0
    with open(args.src_processed, 'w') as src_out:
        for idx, src in tqdm(enumerate(srcs)):
            if args.mode == 'LINE_GRAPH':
                graph_triples, reentrancies, tokens = edge_pos_emb(src)
                
                
                triple_out.write(' '.join(['(%d,%d,%s)' % adj for adj in graph_triples]) + '\n')
            
                # alignment consistancy
                try:
                    src_out.write(' '.join(tokens))
                    reentrances_file.write(str(reentrancies) + '\n')
                    count_instance += 1
                except:
                    print(idx, tokens)
                    exit()
        print('count_instance', count_instance)


###########################

            
if __name__ == "__main__":
    
    # Parse input
    parser = argparse.ArgumentParser(description="Preprocess DART into linearised forms")
    parser.add_argument('--src_path',
                        # default = '/home/lily/wz336/StructAdapt/data/tmp_amr/test/graphs.txt', 
                        default='/home/lily/wz336/StructAdapt/data/dart_exp0/train.source',
                        type=str, help='input AMR file')
    parser.add_argument('--src_processed', 
                        #default= '/home/lily/wz336/StructAdapt/preprocess/test.source', 
                        default='/home/lily/wz336/StructAdapt/data/processed_dart/train.source',
                        type=str, help='output file, either AMR or concept list')
    parser.add_argument('--mode', type=str, default='LINE_GRAPH', help='preprocessing mode',
                        choices=['GRAPH','LIN','LINE_GRAPH'])
    parser.add_argument('--triples-output',  type=str, 
                        #default='/home/lily/wz336/StructAdapt/preprocess/test.graph', 
                        default='/home/lily/wz336/StructAdapt/data/processed_dart/train.graph.tok', 
                        help='triples output for graph2seq')
 
    args = parser.parse_args()

    SENSE_PATTERN = re.compile('-[0-9][0-9]$')
    
    main(args)
