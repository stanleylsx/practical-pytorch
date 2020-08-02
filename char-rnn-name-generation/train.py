# Practical PyTorch: Generating Names with a Conditional Character-Level RNN
import time
import math
from data import *
from model import *

n_epochs = 100000
print_every = 5000
plot_every = 500
all_losses = []
loss_avg = 0  # Zero every plot_every epochs to keep a running average
hidden_size = 128
learning_rate = 0.0005

rnn = RNN(n_categories, n_letters, hidden_size, n_letters)
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

start = time.time()


# Training the Network

def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.init_hidden()
    optimizer.zero_grad()
    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()
    optimizer.step()

    return output, loss.item() / input_line_tensor.size(0)


def time_since(t):
    now = time.time()
    s = now - t
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


print("Training for %d epochs..." % n_epochs)
for epoch in range(1, n_epochs + 1):
    category, input_line, target_line = random_training_set()
    output, loss = train(category, input_line, target_line)
    loss_avg += loss

    if epoch % print_every == 0:
        print('%s (%d %d%%) %.4f' % (time_since(start), epoch, epoch / n_epochs * 100, loss))

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0


print("Saving before quit...")
torch.save(rnn, 'char-rnn-name-generation.pt')
