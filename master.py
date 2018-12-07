import torch
import model
import time


def run():

    modell = model.CNN()

    size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    group_list = []
    for i in range(size):
        group_list.append(i)
    group = torch.distributed.new_group(group_list)

    while(1):
        for param in modell.parameters():
            torch.distributed.broadcast(param.data, src=0, group=group)

        #param_list = []
        for param in modell.parameters():
            #param_list.append(param.data)
            tensor_update = torch.zeros_like(param.data)
            torch.distributed.reduce(tensor_update, dst=0, op=torch.distributed.reduce_op.SUM, group=group)
            tensor_update /= (size-1)
            param.data += tensor_update
            #print('Rank ', rank, ' has data ', tensor_update)
    
    

if __name__ == "__main__":
    size = 9
    rank = 0
    torch.distributed.init_process_group(backend='tcp', init_method='tcp://127.0.0.1:5000', world_size=size, rank=rank)

    run()

    