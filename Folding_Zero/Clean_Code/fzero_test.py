import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import collections

import math
import random

#TODO: Once all bugs are fixed, create different files for different classes

#import matplotlib
#import matplotlib.pyplot as plt
#TODO: For some reason matpotlib.pyplot doesn't work on WSL (Windows Subsystem for Linux), need to resolve issue

import pdb

"""import sys
sys.stdout = open("fzerosingle.txt", "a+")"""

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#############################################################
#Framework for folding proteins with fixed board size (32x32)
#############################################################
class FoldFramework():
    
    def __init__(self, sequence):
        self.reward = 0
        self.sequence = sequence
        self.dim = 32
        self.done = False
        self.index = 1
        self.pos = [16, 16]
        self.board = np.zeros((self.dim, self.dim))
        self.direction = 1 #0 for left, 1 for up, 2 for right, 3 for down
        self.reset()

    def reset(self):
        self.reward = 0
        self.done = False
        self.index = 1
        self.pos = [16, 16]
        self.board = np.zeros((self.dim, self.dim))
        add = 1 #1 for H, 2 for P, 3 for primary connect, 4 for H-H connect
        if (self.sequence[0] == 'P'):
            add = 2
        self.board[self.pos[0]][self.pos[1]] = add
        self.direction = 1
        return (self.board, self.reward, self.done)

    def collision(self):
        self.done = True
        self.reward = -1000
        if Debug: print("Finished, collision")

    def left(self, add):
        if (self.pos[1] == 0):
            self.collision()
            return
        self.pos[1] -= 1
        self.board[self.pos[0]][self.pos[1]] = 3
        self.pos[1] -= 1
        if (self.board[self.pos[0]][self.pos[1]] != 0):
            self.collision()
        self.board[self.pos[0]][self.pos[1]] = add

    def up(self, add):
        if (self.pos[0] == 0):
            self.collision()
            return
        self.pos[0] -= 1
        self.board[self.pos[0]][self.pos[1]] = 3
        self.pos[0] -= 1
        if (self.board[self.pos[0]][self.pos[1]] != 0):
            self.collision()
        self.board[self.pos[0]][self.pos[1]] = add
    
    def right(self, add):
        if (self.pos[1] >= 30):
            self.collision()
            return
        self.pos[1] += 1
        self.board[self.pos[0]][self.pos[1]] = 3
        self.pos[1] += 1
        if (self.board[self.pos[0]][self.pos[1]] != 0):
            self.collision()
        self.board[self.pos[0]][self.pos[1]] = add

    def down(self, add):
        if (self.pos[0] >= 30):
            self.collision()
            return
        self.pos[0] += 1
        self.board[self.pos[0]][self.pos[1]] = 3
        self.pos[0] += 1
        if (self.board[self.pos[0]][self.pos[1]] != 0):
            self.collision()
        self.board[self.pos[0]][self.pos[1]] = add

    def testleft(self):
        if (self.pos[1] == 0):
            return False
        if (self.board[self.pos[0]][self.pos[1] - 2] != 0):
            return False
        return True

    def testup(self):
        if (self.pos[0] == 0):
            return False
        if (self.board[self.pos[0] - 2][self.pos[1]] != 0):
            return False
        return True
    
    def testright(self):
        if (self.pos[1] == 30):
            return False
        if (self.board[self.pos[0]][self.pos[1] + 2] != 0):
            return False
        return True

    def testdown(self):
        if (self.pos[0] == 30):
            return False
        if (self.board[self.pos[0] + 2][self.pos[1]] != 0):
            return False
        return True


    def updateH(self):
        if (self.pos[0] > 0 and self.board[self.pos[0] - 2][self.pos[1]] == 1 and self.board[self.pos[0] - 1][self.pos[1]] == 0):
            self.board[self.pos[0] - 1][self.pos[1]] = 4
            self.reward += 1
        if (self.pos[1] > 0 and self.board[self.pos[0]][self.pos[1] - 2] == 1 and self.board[self.pos[0]][self.pos[1] - 1] == 0):
            self.board[self.pos[0]][self.pos[1] - 1] = 4
            self.reward += 1
        if (self.pos[0] < 30 and self.board[self.pos[0] + 2][self.pos[1]] == 1 and self.board[self.pos[0] + 1][self.pos[1]] == 0):
            self.board[self.pos[0] + 1][self.pos[1]] = 4
            self.reward += 1
        if (self.pos[1] < 30 and self.board[self.pos[0]][self.pos[1] + 2] == 1 and self.board[self.pos[0]][self.pos[1] + 1] == 0):
            self.board[self.pos[0]][self.pos[1] + 1] = 4
            self.reward += 1

    def steptest(self, move): #Move: -1 for left, 0 for straight, 1 for right
        if (self.done):
            print("Folding finished, please reset")
            return False
        if (self.direction == 0):
            if (move == -1):
                return self.testdown()
            elif (move == 0):
                return self.testleft()
            elif (move == 1):
                return self.testup()

        elif (self.direction == 1):
            if (move == -1):
                return self.testleft()
            elif (move == 0):
                return self.testup()
            elif (move == 1):
                return self.testright()

        elif (self.direction == 2):
            if (move == -1):
                return self.testup()
            elif (move == 0):
                return self.testright()
            elif (move == 1):
                return self.testdown()

        elif (self.direction == 3):
            if (move == -1):
                return self.testright()
            elif (move == 0):
                return self.testdown()
            elif (move == 1):
                return self.testleft()

    def step(self, move): #Move: -1 for left, 0 for straight, 1 for right
        if (self.done):
            print("Folding finished, please reset")
            return (self.board, self.reward, self.done)
        add = 1
        if (self.sequence[self.index] == 'P'):
            add = 2
        if (self.direction == 0):
            if (move == -1):
                self.down(add)
            elif (move == 0):
                self.left(add)
            elif (move == 1):
                self.up(add)

        elif (self.direction == 1):
            if (move == -1):
                self.left(add)
            elif (move == 0):
                self.up(add)
            elif (move == 1):
                self.right(add)

        elif (self.direction == 2):
            if (move == -1):
                self.up(add)
            elif (move == 0):
                self.right(add)
            elif (move == 1):
                self.down(add)

        elif (self.direction == 3):
            if (move == -1):
                self.right(add)
            elif (move == 0):
                self.down(add)
            elif (move == 1):
                self.left(add)
        self.direction += move
        self.direction %= 4
        if (add == 1):
             self.updateH()
        if (self.index == len(self.sequence) - 1):
            self.done = True
            if Debug: print("Folding Complete")
        self.index += 1
        return (self.board, self.reward, self.done)

#Folding environment for variable board size (not 32x32)
"""
class FoldFramework():

    def __init__(self, sequence):
        self.reward = 0
        self.sequence = sequence
        self.dim = (len(sequence) - 1)*4 + 1
        self.done = False
        self.index = 1
        self.pos = [math.ceil(self.dim/2) - 1, math.ceil(self.dim/2) - 1]
        self.board = np.zeros((self.dim, self.dim))
        self.direction = 1 #0 for left, 1 for up, 2 for right, 3 for down
        self.reset()
    
    def collision(self):
        self.done = True
        self.reward = -1000
        print("Finished, collision")

    def left(self, add):
        self.pos[1] -= 1
        self.board[self.pos[0]][self.pos[1]] = 3
        self.pos[1] -= 1
        if (self.board[self.pos[0]][self.pos[1]] != 0):
            self.collision()
        self.board[self.pos[0]][self.pos[1]] = add

    def up(self, add):
        self.pos[0] -= 1
        self.board[self.pos[0]][self.pos[1]] = 3
        self.pos[0] -= 1
        if (self.board[self.pos[0]][self.pos[1]] != 0):
            self.collision()
        self.board[self.pos[0]][self.pos[1]] = add
    
    def right(self, add):
        self.pos[1] += 1
        self.board[self.pos[0]][self.pos[1]] = 3
        self.pos[1] += 1
        if (self.board[self.pos[0]][self.pos[1]] != 0):
            self.collision()
        self.board[self.pos[0]][self.pos[1]] = add

    def down(self, add):
        self.pos[0] += 1
        self.board[self.pos[0]][self.pos[1]] = 3
        self.pos[0] += 1
        if (self.board[self.pos[0]][self.pos[1]] != 0):
            self.collision()
        self.board[self.pos[0]][self.pos[1]] = add

    def reset(self):
        self.reward = 0
        self.done = False
        self.index = 1
        self.pos = [math.ceil(self.dim/2) - 1, math.ceil(self.dim/2) - 1]
        self.board = np.zeros((self.dim, self.dim))
        add = 1 #1 for H, 2 for P, 3 for primary connect, 4 for H-H connect
        if (self.sequence[0] == 'P'):
            add = 2
        self.board[self.pos[0]][self.pos[1]] = add
        self.direction = 1 #0 for left, 1 for up, 2 for right, 3 for down
        return (self.board, self.reward, self.done)

    def updateH(self):
        if (self.pos[0] > 0 and self.board[self.pos[0] - 2][self.pos[1]] == 1 and self.board[self.pos[0] - 1][self.pos[1]] == 0):
            self.board[self.pos[0] - 1][self.pos[1]] = 4
            self.reward += 1
        if (self.pos[1] > 0 and self.board[self.pos[0]][self.pos[1] - 2] == 1 and self.board[self.pos[0]][self.pos[1] - 1] == 0):
            self.board[self.pos[0]][self.pos[1] - 1] = 4
            self.reward += 1
        if (self.pos[0] < self.dim - 1 and self.board[self.pos[0] + 2][self.pos[1]] == 1 and self.board[self.pos[0] + 1][self.pos[1]] == 0):
            self.board[self.pos[0] + 1][self.pos[1]] = 4
            self.reward += 1
        if (self.pos[1] < self.dim - 1 and self.board[self.pos[0]][self.pos[1] + 2] == 1 and self.board[self.pos[0]][self.pos[1] + 1] == 0):
            self.board[self.pos[0]][self.pos[1] + 1] = 4
            self.reward += 1

    def step(self, move): #Move: -1 for left, 0 for straight, 1 for right
        if (self.done):
            print("Folding finished, please reset")
            return (self.board, self.reward, self.done)
        add = 1
        if (self.sequence[self.index] == 'P'):
            add = 2
        if (self.direction == 0):
            if (move == -1):
                self.down(add)
            elif (move == 0):
                self.left(add)
            elif (move == 1):
                self.up(add)

        elif (self.direction == 1):
            if (move == -1):
                self.left(add)
            elif (move == 0):
                self.up(add)
            elif (move == 1):
                self.right(add)

        elif (self.direction == 2):
            if (move == -1):
                self.up(add)
            elif (move == 0):
                self.right(add)
            elif (move == 1):
                self.down(add)

        elif (self.direction == 3):
            if (move == -1):
                self.right(add)
            elif (move == 0):
                self.down(add)
            elif (move == 1):
                self.left(add)
        self.direction += move
        self.direction %= 4
        if (add == 1):
             self.updateH()
        if (self.index == len(self.sequence) - 1):
            self.done = True
            print("Folding Complete")
        self.index += 1
        return (self.board, self.reward, self.done)
"""

##########################################################
#Architecture for Neural Net - 
#20 Stacked Residual blocks followed by
#two output heads. Pi output is a 3x1 vector and v output
# is a scalar
##########################################################

####################################################
#2.0 Implementation of neural network
####################################################

class Expand(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(17, 128, 3, padding = 1)
        self.bn = nn.BatchNorm2d(128)
    
    def forward(self, x):
        x = x.view(-1, 17, 32, 32)
        x = self.bn(self.conv(x))
        return x

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)


    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(x))
        y.add_(x)
        y = F.relu(y)
        return y

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.rblayers = self.layer_blocks()
        self.conv1 = nn.Conv2d(128, 1, 1)
        self.conv2 = nn.Conv2d(128, 1, 1)
        self.fc1 = nn.Linear(1024, 3)
        self.fc2 = nn.Linear(1024, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward (self, x):
        x = self.rblayers(x)

        pi = self.conv1(x)
        v = self.conv2(x)

        pi = pi.view(-1, 32*32)
        v = pi.view(-1, 32*32)
        pi = self.fc1(pi)
        v = self.fc2(v)

        return self.softmax(pi), torch.tanh(v)
    
    def layer_blocks(self):
        layers = []
        layers.append(Expand())
        for _ in range(19):
            layers.append(ResidualBlock())
        return nn.Sequential(*layers)


###################################################
#Implementation of Monte Carl Search Tree
###################################################


#Node and State are utility classes for the search tree and for training the model, respectively
class Node():
    def __init__(self, p, parent = None):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = p
        self.children = [] #Should always contain either 0 or 3 elements, index 0 is left, index 1 is straight, index 2 is right
        self.parent = parent

class State():
    def __init__(self, tensor, pi, v):
        self.tensor = tensor
        self.pi = pi
        self.v = v

#Monte Carlo Search Tree
class Tree():
    def __init__(self, env, nnet, seq):
        self.seq = seq
        self.rupper = self.calc_rupper()
        self.moves = []
        self.env = FoldFramework(seq)
        self.nnet = nnet
        self.snode = Node(p = 1)
        self.cnode = self.snode
        self.board, _, _ = env.reset()
        self.pis = []
        self.finalstates = [self.board]
        self.boards = [self.board]
        with torch.no_grad():
            pitemp, _ = self.nnet(self.get_input(1, self.boards))
            m = torch.distributions.Dirichlet(pitemp)
            pitemp = m.sample()
            pitemp = pitemp.view(3)
            for pi in pitemp:
                addnode = Node(pi, parent = self.cnode)
                self.cnode.children.append(addnode)
    
    #returns a numpy array of states to add to training set from the search tree
    def run(self):
        while len(self.snode.children) > 0:
            while (self.snode.children[0].n + self.snode.children[1].n + self.snode.children[2].n < 300):
                if Debug: print("Current tree search number: ", self.snode.children[0].n + self.snode.children[1].n + self.snode.children[2].n)
                self.selection()
            temp = [self.snode.children[0].n, self.snode.children[1].n, self.snode.children[2].n]
            ttemp = [x / 300 for x in temp]
            self.pis.append(ttemp)
            self.moves.append(temp.index(max(temp)) - 1)
            self.snode = self.snode.children[temp.index(max(temp))]
            if Debug: 
                print("Current node id:", len(self.moves) + 1)
                print("Current Moveset:")
                for move in self.moves:
                    print(move, end=" ")
                print()
        self.env.reset()
        tempboard, _, _ = self.env.step(self.moves[0])
        for i in range(1, len(self.moves)):
            self.finalstates.append(tempboard)
            tempboard, _, _ = self.env.step(self.moves[i])
        ans = []
        for i in range(len(self.finalstates)):
            databoard = self.get_input(i+1, self.finalstates)
            addans = State(databoard, self.pis[i], self.env.reward)
            ans.append(addans)
        return ans, self.env.reward, self.env.board


    def selection(self):
        self.board, _, _ = self.env.reset()
        self.boards = [self.board]
        for m in self.moves:
            self.board, _, _ = self.env.step(m)
            self.boards.append(self.board)
        self.cnode = self.snode
        while (len(self.cnode.children) > 0) and not self.env.done:
            temp = [self.selector(self.cnode.children[0]), self.selector(self.cnode.children[1]), self.selector(self.cnode.children[2])]
            self.cnode = self.cnode.children[temp.index(max(temp))]
            self.board, _, _ = self.env.step(temp.index(max(temp)) - 1)
            self.boards.append(self.board)
        if (not self.env.done):
            self.expansion()
        else:
            self.backprop(self.env.reward, done=True)



    def expansion(self):
        with torch.no_grad():
            probs, val = self.nnet(self.get_input(self.env.index, self.boards))
            m = torch.distributions.Dirichlet(probs)
            probs = m.sample()
            probs = probs.view(3)
            val = val.view(1)
            for pi in probs:
                addnode = Node(pi, parent = self.cnode)
                self.cnode.children.append(addnode)
            if not self.env.steptest(-1):
                self.cnode.children[0].p = -20000
            if not self.env.steptest(0):
                self.cnode.children[1].p = -20000
            if not self.env.steptest(1):
                self.cnode.children[2].p = -20000
        self.backprop(val, done = False)

    def backprop(self, v, done):
        while self.cnode.parent != None:
            self.cnode.n += 1
            self.cnode.w += v
            self.cnode.q = self.cnode.w / self.cnode.n
            self.cnode = self.cnode.parent

    #given an id to the boards list, returns the input tensor for the neural net (use 1 indexing)
    def get_input(self, id, arr):
        first = torch.empty(32, 32)
        if (id < 4):
            first = board2tensor(arr[id - 1])
            for i in range(id - 2, -1, -1):
                first = torch.cat((first, board2tensor(arr[i])), 0)
            for i in range(0, 4 - id):
                first = torch.cat((first, torch.zeros(4, 32, 32)), 0)
        else:
            first = board2tensor(arr[id - 1])
            for i in range(id - 2, id - 5, -1):
                first = torch.cat((first, board2tensor(arr[i])), 0)
        if (self.seq[id - 1] == 'P'):
            first = torch.cat((first, torch.zeros(1, 32, 32)), 0)
        else:
            first = torch.cat((first, torch.ones(1, 32, 32)), 0)
        return first

    def selector(self, node):
        sumn = node.parent.children[0].n + node.parent.children[1].n + node.parent.children[2].n
        qstar = node.q / self.rupper
        u = node.p * math.sqrt(sumn) / (1 + node.n)
        return qstar + u
                

    def calc_rupper(self):
        odds = 0
        evens = 0
        for i in range(len(self.seq)):
            if (self.seq[i-1] == 'H' and i % 2 == 0):
                evens += 1
            elif (self.seq[i-1] == 'H' and i%2 == 1):
                odds += 1
        return 2*min(odds, evens)


#################################################
#Utility and Training Functions:
#################################################

#function that converts a board to a tensor
def board2tensor(board):
    h = torch.from_numpy((board == 1).astype(int)).float()
    p = torch.from_numpy((board == 2).astype(int)).float()
    c = torch.from_numpy((board == 3).astype(int)).float()
    b = torch.from_numpy((board == 4).astype(int)).float()
    h = h.view(1, 32, 32)
    p = p.view(1, 32, 32)
    c = c.view(1, 32, 32)
    b = b.view(1, 32, 32)
    out = torch.cat((h, p, c, b),0)
    return out


def pi_loss_func(outputs, targets):
    out = -1*torch.sum(targets[0] * torch.log(torch.clamp(outputs[0], 1e-15, 1)))
    for i in range(1, len(outputs)):
        out.add_(-1*torch.sum(targets[i] * torch.log(torch.clamp(outputs[i], 1e-15, 1))))
    return out
v_loss_func = nn.MSELoss(reduction = "sum")

def optimize_net(nnet, traindata):
    if len(traindata) >= 256:
        print("training...")
        trainset = random.sample(traindata, 256)
        optimizer = optim.SGD(nnet.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00004)

        """for state in trainset:
            optimizer.zero_grad()

            out_pi, out_v = nnet(state.tensor)
            state.v = torch.tensor([state.v], dtype=torch.float)
            state.pi = torch.tensor(state.pi, dtype=torch.float)
            loss_v = v_loss_func(out_v, state.v)
            loss_pi = pi_loss_func(out_pi, state.pi)

            total_loss = loss_v + loss_pi
            total_loss.backward()

            optimizer.step()"""
        
        inputTensors = trainset[0].tensor.view(-1, 32, 32, 17)
        vLabels = torch.tensor([trainset[0].v], dtype=torch.float).view(-1, 1)
        piLabels = torch.tensor(trainset[0].pi, dtype=torch.float).view(-1, 3)

        for it in range(1, len(trainset)):
            inputTensors = torch.cat((inputTensors, trainset[it].tensor.view(-1, 32, 32, 17)), 0)
            vLabels = torch.cat((vLabels, torch.tensor([trainset[it].v], dtype = torch.float).view(-1, 1)), 0)
            piLabels = torch.cat((piLabels, torch.tensor(trainset[it].pi, dtype=torch.float).view(-1, 3)), 0)
        inputTensors = inputTensors.to(device)
        vLabels = vLabels.to(device)
        piLabels = piLabels.to(device)

        out_pi, out_v = nnet(inputTensors)
        loss_v = v_loss_func(out_v, vLabels)
        loss_pi = pi_loss_func(out_pi, piLabels)
        total_loss = loss_v + loss_pi
        total_loss.backward()

        optimizer.step()



        

######################################################
#Main instructions
######################################################

Debug = False #If true, prints out extra status indicators to console while running
Mode = "Multiple" #"Single" for running a single protein sequence to run, "Multiple" for running all protein sequences in list


Sequences = ["HHPPPPPHHPPPHPPPHP",
"HPHPHHHPPPHHHHPPHH",
"PHPPHPHHHPHHPHHHHH",
"HPHPPHHPHPPHPHHPPHPH",
"HHHPPHPHPHPPHPHPHPPH",
"HHPPHPPHPPHPPHPPHPPHPPHH",
"PPHPPHHPPPPHHPPPPHHPPPPHH",
"PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP",
"PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH",
"HHPHPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH",
"PPHHHPHHHHHHHHPPPHHHHHHHHHHPHPPPHHHHHHHHHHHHPPPPHHHHHHPHHPHP",
"HHHHHHHHHHHHPHPHPPHHPPHHPPHPPHHPPHHPPHPPHHPPHHPPHPHPHHHHHHHHHHH",
"HHHHPPPPHHHHHHHHHHHHPPPPPPHHHHHHHHHHHHPPPHHHHHHHHHHHHPPPHHHHHHHHHHHHPPPHPPHHPPHHPPHPH",
"PPPPPPHPHHPPPPPHHHPHHHHHPHHPPPPHHPPHHPHHHHHPHHHHHHHHHHPHHPHHHHHHHPPPPPPPPPPPHHHHHHHPPHPHHHPPPPPPHPHH",
"PPPHHPPHHHHPPHHHPHHPHHPHHHHPPPPPPPPHHHHHHPPHHHHHHPPPPPPPPPHPHHPHHHHHHHHHHHPPHHHPHHPHPPHPHHHPPPPPPHHH"]

traindata = collections.deque(maxlen=60000) # Circular buffer containing states to train the model

print("Starting...")

if Mode == "Single":
    net = Net()
    net = net.to(device)
    seq = Sequences[0] #change this number depending on which sequence to run
    board = None
    maxe = 0
    for _ in range(750):
        #set up and run MCST
        env = FoldFramework(seq)
        mcst = Tree(env, net, seq)
        newdata, score, newboard = mcst.run()
        #determine if new high score is reached
        if score >= maxe:
            maxe = score
            board = newboard
        #optimize the model
        for data in newdata:
            traindata.append(data)
        optimize_net(net, traindata)
        #print out relevant information
        print("Score this run: ", score)
        for row in enumerate(newboard):
            print(row)
        print("Current Maximum Score: ", maxe)
        for row in enumerate(board):
            print(row)

elif Mode == "Multiple":
    count = 0
    prevnet = Net()
    currnet = Net() #Neural Net
    currnet = currnet.to(device)
    prevnet = prevnet.to(device)
    board = [0] * 15 #board to be printed to console
    maxe = [0] * 15 #Maximum score reached
    for _ in range(750):
        for i in range(15):

            #competetive mechanism
            count += 1
            if count % 1000 == 0:
                seq = random.choice(Sequences)
                env = FoldFramework(seq)
                mcst1 = Tree(env, currnet, seq)
                mcst2 = Tree(env, prevnet, seq)
                _, score1, _ = mcst1.run()
                _, score2, _ = mcst2.run()
                if (score1 >= score):
                    prevnet = currnet
                else:
                    currnet = prevnet

            seq = Sequences[i]
            env = FoldFramework(seq)
            mcst = Tree(env, currnet, seq)
            newdata, score, newboard = mcst.run()
            if score >= maxe[i]:
                maxe[i] = score
                board[i] = newboard
            for data in newdata:
                traindata.append(data)
            optimize_net(currnet, traindata)
            print("Current Sequence:", seq)
            print("Score this run:", score)
            print("Maximum Scores:")
            print(maxe)
    for brd in board:
            for row in enumerate(brd):
                print(row)






    
