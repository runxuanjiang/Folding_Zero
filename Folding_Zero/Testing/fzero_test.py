import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import math
import random

#TODO: Once all bugs are fixed, create different files for different classes

#import matplotlib
#import matplotlib.pyplot as plt
#TODO: For some reason matpotlib.pyplot doesn't work on WSL (Windows Subsystem for Linux), need to resolve issue

import pdb

"""import sys
sys.stdout = open("fzerosingle.txt", "a+")"""



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

#First residual block for the neural net
class FirstResBlock(nn.Module):
    def __init__(self):
        super(FirstResBlock, self).__init__()
        self.conv1 = nn.Conv2d(17, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)

    def forward (self, x):
        x = x.view(-1, 17, 32, 32)
        x = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(x)))
        y += x
        return y

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(x)))
        y += x
        return y

#Neural Network to be used in the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rblayers = self.layer_blocks()
        self.conv1 = nn.Conv2d(128, 1, 1)
        self.conv2 = nn.Conv2d(128, 1, 1)
        self.fc1 = nn.Linear(1024, 3)
        self.fc2 = nn.Linear(1024, 1)

    def forward (self, x):
        x = self.rblayers(x)
        pi = self.conv1(x)
        v = self.conv2(x)

        pi = pi.view(32*32)
        v = pi.view(32*32)
        pi = self.fc1(pi)
        v = self.fc2(v)
        return F.softmax(pi, dim=0), torch.tanh(v)
    
    def layer_blocks(self):
        layers = []
        layers.append(FirstResBlock())
        for _ in range(19):
            layers.append(ResBlock())
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
            pitemp, _ = nnet(self.get_input(1, self.boards))
            #m = Dirichlet(pitemp)
            #pitemp = m.sample()
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
            #m = Dirichlet(probs)
            #probs = m.sample()
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
                first = torch.cat((first, board2tensor(arr[i])), 2)
            for i in range(0, 4 - id):
                first = torch.cat((first, torch.zeros(32, 32, 4)), 2)
        else:
            first = board2tensor(arr[id - 1])
            for i in range(id - 2, id - 5, -1):
                first = torch.cat((first, board2tensor(arr[i])), 2)
        if (self.seq[id - 1] == 'P'):
            first = torch.cat((first, torch.zeros(32, 32, 1)), 2)
        else:
            first = torch.cat((first, torch.ones(32, 32, 1)), 2)
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
    h = torch.zeros(board.shape)
    p = torch.zeros(board.shape)
    c = torch.zeros(board.shape)
    b = torch.zeros(board.shape)
    for i in range(board.shape[0]):
        for j in range(board.shape[0]):
            if (board[i][j] == 1):
                h[i][j] = 1
            elif board[i][j] == 2:
                p[i][j] = 1
            elif board[i][j] == 3:
                c[i][j] = 1
            elif board[i][j] == 4:
                b[i][j] = 1
    h = h.view(32, 32, 1)
    p = p.view(32, 32, 1)
    c = c.view(32, 32, 1)
    b = b.view(32, 32, 1)
    out = torch.cat((h, p, c, b),2)
    return out


def pi_loss_func(targets, outputs):
    return -torch.sum(targets*torch.log(outputs))
v_loss_func = nn.MSELoss(reduction = "none")
def optimize_debug(nnet, inputstates):
    print("training...")
    optimizer = optim.SGD(nnet.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00004)
    for state in inputstates:
        print(state.tensor)
        print(state.pi)
        print(state.v)
        optimizer.zero_grad()

        out_pi, out_v = nnet(state.tensor)
        state.v = torch.tensor([state.v], dtype=torch.float)
        state.pi = torch.tensor(state.pi, dtype=torch.float)
        loss_v = v_loss_func(out_v, state.v)
        print(out_pi)
        print(state.pi)
        loss_pi = pi_loss_func(out_pi, state.pi)
        total_loss = loss_v + loss_pi
        total_loss.backward()
        optimizer.step()




#TODO: Bugs in this function:
#Including the loss function syntax is incorrect and
#cannot use append when updating the traindata list (use + instead)
def optimize_net(nnet, traindata):
    print("training...")
    if len(traindata) >= 256:
        trainset = random.sample(traindata, 256)
        optimizer = optim.SGD(nnet.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00004)

        #for batch input
        """
        batch = trainset[0].tensor
        batch = batch.view(-1, 17, 32, 32)
        for i in range(1, 256):
            tensor = trainset[i].tensor
            tensor.view(-1, 17, 32, 32)
            batch = torch.cat((batch, tensor), 3)
        optimizer.zero_grad()
        out_pi, out_v = nnet(state.tensor)
        loss_pi = nn.MSELoss(out_pi, )
        """

        #for single input
        
        for state in trainset:
            optimizer.zero_grad()

            out_pi, out_v = nnet(state.tensor)
            loss_v = nn.MSELoss(out_v, state.v)
            loss_pi = nn.CrossEntropyLoss(out_pi, state.pi)
            total_loss = loss_pi + loss_v

            total_loss.backward()
            optimizer.step()
        

######################################################
#Main instructions
######################################################

Debug = True #If true, prints out extra status indicators to console while running

board = None #board to be printed to console
net = Net() #Neural Net
traindata = [] # Array containing states to train the model
maxe = 0 #Maximum score reached
sequences = ["HHPPPPPHHPPPHPPPHP",
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

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

seq = "HHPH"#sequences[0] #change this number depending on which sequence to run
for _ in range(2000):
    if Debug:
        env = FoldFramework(seq)
        mcst = Tree(env, net, seq)
        newdata, score, newboard = mcst.run()
        if score >= maxe:
            maxe = score
            board = newboard
        optimize_debug(net, newdata)
        print("Score this run: ", score)
        for row in enumerate(newboard):
            print(row)
    else:
        env = FoldFramework(seq)
        mcst = Tree(env, net, seq)
        newdata, score, newboard = mcst.run()
        if score >= maxe:
            maxe = score
            board = newboard
        traindata.append(newdata)
        optimize_net(net, traindata)

    print("Current Maximum Score: ", maxe)
    for row in enumerate(board):
        print(row)



    
