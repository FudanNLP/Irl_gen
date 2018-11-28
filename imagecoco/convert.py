import cPickle

def convert(input_file):
    data_Name = "cotra"
    vocab_file = "vocab_" + data_Name + ".pkl"

    word, vocab = cPickle.load(open('save/'+vocab_file))
    wordlen = len(word)
    input = 'save/' + input_file
    output_file = 'speech/' + data_Name + '_' + input_file.split('le')[-1]
    with open(output_file, 'w')as fout:
        with open(input)as fin:
            for line in fin:
                line = line.split()
                line = [int(x) for x in line]
                # 4839 is vocab size
                if all(i < wordlen for i in line) is False:
                    continue
                line = [word[x] for x in line]
                line = ' '.join(line) + '\n'
                fout.write(line)  # .encode('utf-8'))

if __name__ == '__main__':
    convert('evaler_file0.7550')
