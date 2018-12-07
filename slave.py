import torch
from datasource import Mnist
import model
import time
import copy
from torch.multiprocessing import Process



def get_new_model(model):
    for param in model.parameters():
        torch.distributed.recv(param.data, src=0)
    return  model

def run(size, rank):

    modell = model.CNN()
    optimizer = torch.optim.Adam(modell.parameters(), lr=0.001)
    loss_func = torch.nn.CrossEntropyLoss()

    #size = torch.distributed.get_world_size()
    #rank = torch.distributed.get_rank()

    train_loader = Mnist().get_train_data()

    test_data = Mnist().get_test_data()
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.test_labels[:2000]

    group_list = []
    for i in range(size):
        group_list.append(i)
    group = torch.distributed.new_group(group_list)

    print( rank, group_list)

    for epoch in range(50):

        for step, (b_x, b_y) in enumerate(train_loader):

            modell = get_new_model(modell)

            current_model = copy.deepcopy(modell)


            output = modell(b_x)[0]
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()   
            optimizer.step()

            new_model = copy.deepcopy(modell)
            

            if step % 50 == 0:
                test_output, last_layer = modell(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
                
              
            for param1, param2 in zip( current_model.parameters(), new_model.parameters() ):
                torch.distributed.reduce(param2.data-param1.data, dst=0, op=torch.distributed.reduce_op.SUM, group=group)
                
def init_processes(size, rank, run):
    torch.distributed.init_process_group(backend='tcp', init_method='tcp://127.0.0.1:5000', world_size=size, rank=rank)
    run(size, rank)

if __name__ == "__main__":
    size = 9
    processes = []
    for rank in range(1, size):
        p = Process(target=init_processes, args=(size, rank, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
 
 