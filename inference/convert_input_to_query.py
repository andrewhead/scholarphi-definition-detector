import sys, os, json, re
import argparse

def get_queries(sentences):
    new_sentences = []
    for s in sentences:
        symbols = [[m.start(),m.end()] for m in re.finditer('SYMBOL', s)]
        if len(symbols)>0:
            for start,end in symbols:
                temp = s[:start] + '--- ' + s[start:end] + ' ---' + s[end:]
                new_sentences.append(temp)
        else:
            new_sentences.append(s)
    new_sentences = [x + '\n' for x in new_sentences]
    return new_sentences

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Input file')
    parser.add_argument('--output', help='Output file', default="query_input.txt")
    args = parser.parse_args()

    if args.input==None:
        print('Enter input file path')
        exit(0)
    
    else:
        with open(args.input,'r') as fob:
            sentences = fob.readlines()
        sentences = [x.strip() for x in sentences]
        queries = get_queries(sentences)
        with open(args.output,'w') as fob:
            fob.writelines(queries)
        