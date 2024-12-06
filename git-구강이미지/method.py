
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset,ConcatDataset, DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report
import numpy as np



def train(model, partition, optimizer, criterion):
    trainloader = DataLoader(partition['train'],
                                        batch_size=32,
                                              shuffle=True, num_workers=2)
    model.train()

    correct = 0
    total = 0
    train_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        images, labels = data
        images = images.to('cuda')
        labels = labels.to('cuda')
        
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(trainloader)
    train_acc = 100 * correct / total
    return model, train_loss, train_acc

def validate(model, partition, criterion):
    valloader = torch.utils.data.DataLoader(partition['val'],
                                            batch_size=10, 
                                            shuffle=False, num_workers=2)
    model.eval()

    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images = images.to('cuda')
            labels = labels.to('cuda')

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(valloader)
        val_acc = 100 * correct / total
    return val_loss, val_acc

def test_for_inference_time(model, partition):
    testloader = torch.utils.data.DataLoader(partition['test'],
                                        batch_size=10,
                                             shuffle=False, num_workers=2)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to('cuda')
            labels = labels.to('cuda')
            outputs = model(images)
    return 
    
def test(model, partition):
    testloader = torch.utils.data.DataLoader(partition['test'],
                                        batch_size=10,
                                             shuffle=False, num_workers=2)
    model.eval()

    correct = 0
    total = 0
    total_predicted=[]
    real=[]
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to('cuda')
            labels = labels.to('cuda')

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_predicted.append(predicted)
            real.append(labels)
        test_acc = 100 * correct / total
    return test_acc, total_predicted,real




def model_train_eval(model,model_name, epochs,partition,results):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model, train_loss, train_acc = train(model, partition, optimizer, criterion)
        val_loss, val_acc = validate(model, partition, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(epoch,"/",epochs,":",val_loss)

    test_acc,test_pred,test_real = test(model, partition)
    results.append({
        'model': model_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'test_pred': test_pred,
        'test_real': test_real,
    })
    return 

def CAE_train(model, partition, optimizer, criterion):
    trainloader = DataLoader(partition['train'],
                                        batch_size=10,
                                              shuffle=True, num_workers=2)
    model.train()

    train_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        images, labels = data
        images = images.to('cuda')
        
        outputs = model(images)

        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return model, train_loss

def CAE_validate(model, partition, criterion):
    valloader = torch.utils.data.DataLoader(partition['val'],
                                        batch_size=10,
                                            shuffle=False, num_workers=2)
    model.eval()

    val_loss = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images = images.to('cuda')

            outputs = model(images)

            loss = criterion(outputs, images)#.reshape(-1))

            val_loss += loss.item()
    return val_loss

def CAE_train_eval(model,epochs_1,partition,results):
    torch.cuda.empty_cache()
    model=model.to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    CAE_train_losses = []
    CAE_val_losses = []

    for epoch in range(epochs_1):
        model, train_loss= CAE_train(model, partition, optimizer, criterion)
        val_loss = CAE_validate(model, partition, criterion)
        
        CAE_train_losses.append(train_loss)
        CAE_val_losses.append(val_loss)
        print(epoch)
    return CAE_train_losses,CAE_val_losses


def plot_loss_and_accuracy(results):
    f=plt.figure(figsize=(20, 8))
    cmap = plt.cm.get_cmap('viridis')
    color_list=['b','r','g','m','y','k']
    marker=['o','*','x','D','s','^']
    
    
    for type,num in zip(['train_losses','train_accs','val_losses','val_accs'],[1,4,2,5]):
        ax=f.add_subplot(2,3,num)
        ax.grid()
    
        for i in range(len(results)):
            y=results[i][type]
            x=np.arange(len(results[i][type]))
            
            ax.plot(x,y,color=color_list[i%6])
            ax.plot(x[::5], y[::5], marker[i%6],  color=color_list[i%6],markersize=8)
            ax.plot([],[], label=results[i]['model'],color=color_list[i%6],marker=marker[i%6],markersize=8)

            ax.set_xlabel('Epoch')
            ax.set_ylabel(type.split('_')[1])
            ax.set_title(type.split('_')[0])
            ax.grid()
        
    lines, labels = ax.get_legend_handles_labels() 
    f.legend(lines, labels,ncols=3,frameon=False, loc=(0.2,0.1))#"lower left")
        
    test_acc_list=[]
    model_list=[]
    for i in range(len(results)):
        test_acc_list.append(results[i]['test_acc'])
        model_list.append(results[i]['model'])
    ax=f.add_subplot(1,3,3)
    sns.barplot(test_acc_list, palette="viridis")
    ax.set_xticks(range(len(results)),model_list,rotation=90)
    ax.set_ylabel('acc')
    ax.set_title('test')
    ax.grid()
    plt.tight_layout()
    
    f=plt.figure(figsize=(5*len(results), 5))
    num=1
    for i in range(len(results)):
        ax=f.add_subplot(1,len(results),num)
        num+=1
        
        pred=results[i]['test_pred']
        real=results[i]['test_real']
        try :
            pred=torch.cat(pred).cpu().reshape(-1)
            real=torch.cat(real).cpu().reshape(-1)
        except:
            pred=pred
            real=real
        accuracy = accuracy_score(real,pred )
        conf_matrix = confusion_matrix(real,pred )
        ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=np.arange(2)).plot(ax=ax)
        ax.set_title(results[i]['model'])
        print('#########'+results[i]['model']+'#########\n'+classification_report(real,pred))