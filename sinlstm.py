from lstmRegression import get_batch_data
from lstmRegression import lstm_model


if __name__ == "__main__":
    time_steps = 20
    batch_size = 50
    input_size = 1
    output_size = 1
    cell_size = 10
    lr = 0.006
    iter_num = 2000
    model = lstm_model(time_steps, input_size, output_size, cell_size, batch_size)
    model.train(Logdir = './log', iter_num = iter_num)