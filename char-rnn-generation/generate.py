# Practical PyTorch: Generating Names with a Conditional Character-Level RNN
from data import *
import torch

rnn = torch.load('char-rnn-generation.pt')

# Generating from the Network

max_length = 20


# Sample from a category and starting letter
def sample(category, start_letter='A', temperature=0.5):
    category_input = make_category_input(category)
    chars_input = make_chars_input(start_letter)
    hidden = rnn.init_hidden()

    output_name = start_letter

    for i in range(max_length):
        output, hidden = rnn(category_input, chars_input[0], hidden)
        topv, topi = output.topk(1)
        topi = topi[0][0]
        if topi == n_letters - 1:
            break
        else:
            letter = all_letters[topi]
            output_name += letter
        chars_input = make_chars_input(letter)

    return output_name


def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))


samples('Chinese', 'CHI')
