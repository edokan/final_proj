#NN Model
class NNmodel(nn.Module):
    def __init__(self):
        super(NNmodel,self).__init__()
        self.linear1 = nn.Linear(32*32,200)
        self.linear2 = nn.Linear(200,100)
        self.final_linear = nn.Linear(100,62)
        
        self.relu = nn.ReLU()
        
    def forward(self,images):
        x = images.view(-1,32*32)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.final_linear(x)
        return x