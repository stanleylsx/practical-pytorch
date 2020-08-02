import torch
import glob
import unicodedata
import string
import random

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)


def find_files(path): return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def read_lines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicode2ascii(line) for line in lines]


# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in find_files('../data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)


# Find letter index from all_letters, e.g. "a" = 0
def letter2index(letter):
    return all_letters.find(letter)


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line2tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter2index(letter)] = 1
    return tensor


def category_from_output(output):
    top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def random_training_pair():
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.LongTensor([all_categories.index(category)])
    line_tensor = line2tensor(line)
    return category, line, category_tensor, line_tensor
