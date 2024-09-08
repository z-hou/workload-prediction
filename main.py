import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from model import LSTM, LSTMAttention
import time, os
import torch
import torch.nn as nn
import math
from sklearn.metrics import mean_squared_error


def data_process(root_path, seq_length):

    Bd = []
    count = 0
    for csv_data in os.listdir(root_path):
        count += 1
        data_path = os.path.join(root_path, csv_data)
        with open(data_path, 'r') as file:
            vm_trace = csv.reader(file, delimiter=';')
            headers = next(vm_trace)
            print(headers)

            #data_list
            cpu_usage = []
            memory_usage = []
            disk_write = []
            idx_cpu_usage = headers.index('\tCPU usage [%]')
            idx_memory_usage = headers.index('\tMemory usage [KB]')
            idx_disk_write = headers.index('\tDisk write throughput [KB/s]')

            for row in vm_trace:
                #print(float(row[idx_cpu_usage]))
                cpu_usage.append(float(row[idx_cpu_usage]))
                memory_usage.append(float(row[idx_memory_usage]))
                disk_write.append(float(row[idx_disk_write]))

            all_data_list = [cpu_usage, memory_usage, disk_write]
            nd_data = np.array(all_data_list)

            remainder = nd_data.shape[1]%seq_length
            #print(remainder)
            new_nd_data = nd_data[:, :-remainder].reshape(len(all_data_list), seq_length, -1)
            new_nd_data = new_nd_data.transpose((2, 1, 0))
            #print(new_nd_data.shape)

            Bd.append(new_nd_data)
        count += 1
        print("!!!!!!!!!", count)
        if count == 100:
            break

        
    
    All_Series = np.vstack(Bd)
    All_Series = All_Series.astype(np.float32)
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    #data_raw = scaler.fit_transform(nd_data)
    print(All_Series.dtype)

    print(All_Series.shape)

    return All_Series



def data_process_SV(root_path, seq_length):
    count = 0
    Bd = []
    for csv_data in os.listdir(root_path):

        data_path = os.path.join(root_path, csv_data)
        with open(data_path, 'r') as file:
             vm_trace = csv.reader(file, delimiter=';')
             headers = next(vm_trace)
             #print(headers)

             #data_list
             cpu_usage = []
             memory_usage = []
             disk_write = []
             idx_cpu_usage = headers.index('\tCPU usage [%]')
             #idx_cpu_usage = headers.index('\tMemory usage [KB]')
             #idx_cpu_usage = headers.index('\tDisk write throughput [KB/s]')


             for row in vm_trace:
                 #print(float(row[idx_cpu_usage]))
                 cpu_usage.append(float(row[idx_cpu_usage]))

             all_data_list = cpu_usage
             nd_data = np.array(all_data_list)
             #print(nd_data.shape)
             #print(nd_data)

             remainder = nd_data.shape[0]%seq_length
             #print(remainder)
             new_nd_data = nd_data[:-remainder].reshape(-1, seq_length)
             #new_nd_data = new_nd_data.transpose((2, 1, 0))
             #print(new_nd_data.shape)

             Bd.append(new_nd_data)
        count += 1
    print("Totally {} *.csv files".format(count))
        #if count == 100:
        #    break
    
    All_Series = np.vstack(Bd)
    All_Series = All_Series.astype(np.float32)
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    #data_raw = scaler.fit_transform(nd_data)
    #print(All_Series.dtype)

    print(All_Series.shape)

    return All_Series






def data_split(Time_series_data, amount, subset):

    #amount: choose a subset of data
    if subset == True:
        Time_series_data = Time_series_data[:amount, :, :]
    
    test_set_size = int(np.round(0.2 * Time_series_data.shape[0]))
    train_set_size = Time_series_data.shape[0] - (test_set_size)

    train_data = Time_series_data[:train_set_size, :-1, :]
    train_label = Time_series_data[:train_set_size, -1, :]

    test_data = Time_series_data[train_set_size:, :-1]
    test_label = Time_series_data[train_set_size:, -1, :]

    print("x_train's shape: ", train_data.shape)
    print("y_train's shape: ", train_label.shape)
    print("x_test's shape: ", test_data.shape)
    print("y_test's shape: ", test_label.shape)

    return train_data, train_label, test_data, test_label


def data_scaler(data):
    scaler_cpu = MinMaxScaler(feature_range=(0, 1))
    cpu_series = data[:, :, 0]
    cpu_series = scaler_cpu.fit_transform(cpu_series.reshape(-1,1))
    cpu_series = cpu_series.reshape(-1, 100)

    scaler_mem = MinMaxScaler(feature_range=(0, 1))
    mem_series = data[:, :, 1]
    mem_series = scaler_mem.fit_transform(mem_series.reshape(-1,1))
    mem_series = mem_series.reshape(-1, 100)

    scaler_disk = MinMaxScaler(feature_range=(0, 1))
    disk_series = data[:, :, 2]
    disk_series = scaler_disk.fit_transform(disk_series.reshape(-1,1))
    disk_series = disk_series.reshape(-1, 100)
    print("disk_series's shape: ", disk_series.shape)



    All_Series = np.stack((cpu_series, mem_series, disk_series), axis=2)

    print("All_Series's shape: ", All_Series.shape)

    return All_Series, scaler_cpu, scaler_mem, scaler_disk





def build_model(input_dim, hidden_dim, num_layers, output_dim, device):
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, device=device)
    #model = LSTMAttention(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, output_size=output_dim)

    return model

def train(model, num_epochs, x_train, y_train_lstm, criterion, optimiser, device):

    model = model.to(device)
    model.train()
    x_train = x_train.to(device)
    y_train_lstm = y_train_lstm.to(device)

    loss_list=[]
    for i in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train_lstm)
        print("Epoch ", i, "MSE: ", loss.item())
        loss_list.append(loss.item())
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    #torch.save(model, "price_predictor_epo{}.pth".format(num_epochs))
    return y_train_pred, loss_list

def evaluation(model, x_test, test_label, device):
    model = model.eval()
    #torch.onnx.export(model, x_test, "model.onnx", verbose=True)
    x_test = x_test.to(device)
    test_pred = model(x_test)
    test_pred = test_pred.detach().cpu().numpy()
    #print("check y_test_pred: ", y_test_pred)
    print("predcition is: ", test_pred.shape, type(test_pred), type(test_label), test_label.shape)
    rme_value = math.sqrt(mean_squared_error(test_label[:,0], test_pred[:,0]))
    print('RME value: %.2f RMSE' % (rme_value))

    return test_pred


if __name__ == '__main__':


    root_data = r"/proj/zhou-cognit/users/x_zhozh/project/faststorage/VMmonitoring"
    All_Series = data_process_SV(root_data, 100)
    
    
    #length = All_Series.shape[0]
    All_Series = All_Series.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    All_Series = scaler.fit_transform(All_Series)
    #print(np.max(All_Series), " ", np.min(All_Series))
    All_Series = All_Series.reshape(-1, 100)
    All_Series = np.expand_dims(All_Series, axis=-1)

    train_data, train_label, test_data, test_label = data_split(All_Series, 2000, subset=False)

    train_data = torch.from_numpy(train_data).type(torch.Tensor)
    test_data = torch.from_numpy(test_data).type(torch.Tensor)
    
    train_label_lstm = torch.from_numpy(train_label).type(torch.Tensor)
    test_label_lstm = torch.from_numpy(test_label).type(torch.Tensor)

    input_dim = 1
    hidden_dim = 256
    num_layers = 2
    output_dim = 1
    num_epochs = 100

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    model = build_model(input_dim, hidden_dim, num_layers, output_dim, device).to(device)
    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.005)


    y_train_pred, loss_list = train(model, num_epochs, train_data, train_label_lstm, criterion, optimiser, device)
    plt.figure()
    plt.plot(loss_list, color='b', label='loss curve')
    plt.savefig('CPU_loss_curve.pdf')

    test_pred = evaluation(model, test_data, test_label, device)
    #print(test_pred.shape)
    #print(test_label.shape)
    #test_pred = test_pred.reshape(-1,300)
    #test_label = test_label.reshape(-1,300)
    #print(test_pred.shape)
    #print(test_label.shape)
    test_predict = scaler.inverse_transform(test_pred)
    test_label = scaler.inverse_transform(test_label)
    
    test_pred = test_pred.reshape(-1,1)
    test_label = test_label.reshape(-1,1)

    plt.figure()
    plt.plot(test_predict[:100], color='b')
    plt.plot(test_label[:100], color='g')
    plt.xlabel('timestamp')
    plt.ylabel('CPU Usage')
    plt.savefig('CPU_prediction.pdf')
    #plt.legend()
    #plt.show()




'''

if __name__ == '__main__':

    root_data = r"/Users/zhouz/Project/VM_Workload_Predictor/fastStorage/VMmonitoring"

    if os.path.exists("/Users/zhouz/Project/VM_Workload_Predictor/All_Series.bin") == True:
        All_Series = np.fromfile("/Users/zhouz/Project/VM_Workload_Predictor/All_Series.bin", dtype=np.float32)
        
    else:
        All_Series = data_process(root_data, 100)



    print(All_Series.shape)
    length = All_Series.shape[0]

    #All_Series = All_Series.reshape(-1, 3)
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    #All_Series = scaler.fit_transform(All_Series)
    #All_Series = All_Series.reshape(length, 100, 3)
    #print(np.max(All_Series), ' ', np.min(All_Series))

    All_Series, scaler_cpu, scaler_mem, scaler_disk = data_scaler(All_Series)


    train_data, train_label, test_data, test_label = data_split(All_Series, 2000, subset=False)

    train_data = torch.from_numpy(train_data).type(torch.Tensor)
    test_data = torch.from_numpy(test_data).type(torch.Tensor)
    
    train_label_lstm = torch.from_numpy(train_label).type(torch.Tensor)
    test_label_lstm = torch.from_numpy(test_label).type(torch.Tensor)

    input_dim = 3
    hidden_dim = 32
    num_layers = 3
    output_dim = 1
    num_epochs = 1000

    model = build_model(input_dim, hidden_dim, num_layers, output_dim)
    #print("##########")
    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    #hist = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []

    y_train_pred, loss_list = train(model, num_epochs, train_data, train_label_lstm, criterion, optimiser)
    plt.figure()
    plt.plot(loss_list, color='b', label='loss curve')
    plt.savefig('loss_curve.png')

    test_predict = evaluation(model, test_data, test_label)
    print("pred: ", test_predict.shape)
    print("label: ", test_label.shape)
    #test_pred = test_pred.reshape(-1,300)
    #test_label = test_label.reshape(-1,300)
    print(test_predict.shape)
    print(test_label.shape)
    #test_predict = scaler.inverse_transform(test_pred)
    #test_label = scaler.inverse_transform(test_label)
    
    #test_pred = test_pred.reshape(-1,1)
    #test_label = test_label.reshape(-1,1)

    predict_cpu = test_predict[:, 0].reshape(-1,1)
    predict_cpu = scaler_cpu.inverse_transform(predict_cpu)

    predict_mem = test_predict[:, 1].reshape(-1,1)
    predict_mem = scaler_mem.inverse_transform(predict_mem)

    predict_disk = test_predict[:, 2].reshape(-1,1)
    predict_disk = scaler_disk.inverse_transform(predict_disk)

    label_cpu = test_label[:, 0].reshape(-1,1)
    label_cpu = scaler_cpu.inverse_transform(label_cpu)

    label_mem = test_label[:, 1].reshape(-1,1)
    label_mem = scaler_mem.inverse_transform(label_mem)

    label_disk = test_label[:, 2].reshape(-1,1)
    label_disk = scaler_disk.inverse_transform(label_disk)


    plt.figure()
    plt.plot(predict_cpu, color='r')
    plt.plot(label_cpu, color='g')
    plt.xlabel('timestamp')
    plt.ylabel('cpu')
    plt.savefig('cpu.png')
    #plt.legend()
    #plt.show()

    plt.figure()
    plt.plot(predict_mem, color='r')
    plt.plot(label_mem, color='g')
    plt.xlabel('timestamp')
    plt.ylabel('memory')
    plt.savefig('mem.png')
    #plt.legend()
    #plt.show()

    plt.figure()
    plt.plot(predict_disk, color='r')
    plt.plot(label_disk, color='g')
    plt.xlabel('timestamp')
    plt.ylabel('disk')
    plt.savefig('disk.png')
    #plt.legend()
    #plt.show()

'''



















