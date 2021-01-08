import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.utils
from torchvision import models
from customized_utils import if_violate_constraints, customized_standardize, customized_inverse_standardize

class VanillaDataset(Data.Dataset):
    def __init__(self, X, y, one_hot=False):
        self.X = X
        self.y = y
        self.one_hot = one_hot

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, device=None):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

        if not device:
            self.device = torch.device("cuda")
        else:
            self.device = device
    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
    def predict(self, x):
        x = torch.from_numpy(x).to(self.device).float()
        out = self.forward(x)
        out = torch.round(out)
        return out.cpu().detach().numpy()
    def predict_proba(self, x):
        x = torch.from_numpy(x).to(self.device).float()
        out = self.forward(x)

        out = torch.stack([1 - out, out], dim=1).squeeze()
        # print(out.cpu().detach().numpy().shape)
        return out.cpu().detach().numpy()


class SimpleNetMulti(SimpleNet):
    def predict(self, x):
        x = torch.from_numpy(x).to(self.device).float()
        out = self.forward(x)
        out = torch.argmax(out)
        return out.cpu().detach().numpy()
    def predict_proba(self, x):
        x = torch.from_numpy(x).to(self.device).float()
        out = self.forward(x)
        return out.cpu().detach().numpy()


def validation(model, test_loader, device, one_hot=True):
    mean_loss = []
    mean_acc = []
    model.eval()
    criterion = nn.BCELoss()
    for i, (x_batch, y_batch) in enumerate(test_loader):
        x_batch = x_batch.to(device).float()
        y_batch = y_batch.to(device).float()

        if one_hot:
            y_batch = y_batch.long()
        y_pred_batch = model(x_batch).squeeze()
        loss = criterion(y_pred_batch, y_batch)
        loss_np = loss.cpu().detach().numpy()

        y_batch_np = y_batch.cpu().detach().numpy()
        y_pred_batch_np = y_pred_batch.cpu().detach().numpy()

        acc = np.mean(np.round(y_pred_batch_np) == y_batch_np)


        mean_loss.append(loss_np)
        mean_acc.append(acc)
        # print('test', y_pred_batch, y_batch, loss_np, acc)

    mean_loss = np.mean(mean_loss)
    mean_acc = np.mean(mean_acc)

    return mean_loss, mean_acc



def pgd_attack(model, images, labels, xl, xu, encoded_fields, labels_used, customized_constraints, standardize, device=None, eps=1.01, alpha=8/255, iters=40):
    if not device:
        device = torch.device("cuda")
    n = len(images)
    m = np.sum(encoded_fields)

    images_all = torch.from_numpy(images).to(device).float()
    labels_all = torch.from_numpy(labels).to(device).float()
    ori_images_all = torch.clone(images_all)

    xl = torch.from_numpy(xl).to(device).float()
    xu = torch.from_numpy(xu).to(device).float()

    loss = nn.BCELoss()

    new_images_all = []
    new_outputs_all = []
    initial_outputs_all = []

    # we deal with images sequentially
    for j in range(n):
        images = torch.unsqueeze(images_all[j], 0)
        labels = labels_all[j]
        ori_images = torch.unsqueeze(ori_images_all[j], 0)
        prev_outputs = torch.zeros(1).to(device).float()
        prev_images = None

        for i in range(iters):

            images.requires_grad = True
            outputs = model(images).squeeze()
            model.zero_grad()





            cost = loss(outputs, labels).to(device)
            cost.backward()

            print('\n'*2)
            print('-'*20, j, i, outputs.squeeze().cpu().detach().numpy(), '-'*20)
            if i == 0:
                initial_outputs_all.append(outputs.squeeze().cpu().detach().numpy())

            # if forward prob not improving break
            if outputs < prev_outputs:
                break
            else:
                prev_images = torch.clone(images)
                prev_outputs = torch.clone(outputs)

            adv_images = images + alpha*images.grad.sign()
            eta = torch.clip(adv_images - ori_images, min=-eps, max=eps)
            images = torch.max(torch.min(ori_images + eta, xu), xl).detach_()


            one_hotezed_images_embed = torch.zeros([images.shape[0], m])
            s = 0

            for field_len in encoded_fields:
                max_inds = torch.argmax(images[:, s:s+field_len], axis=1)
                one_hotezed_images_embed[torch.arange(images.shape[0]), s+max_inds] = 1
                # print(images.cpu().detach().numpy())
                # print(field_len, max_inds.cpu().detach().numpy())
                # print(one_hotezed_images_embed.cpu().detach().numpy())
                s += field_len
            images[:, :m] = one_hotezed_images_embed


            images_non_encode = images[:, m:]
            ori_images_non_encode = ori_images[:, m:]
            images_delta_non_encode = images_non_encode - ori_images_non_encode

            # keep checking violation, exit only when satisfying
            ever_violate = False

            images_non_encode_np = images_non_encode.squeeze().cpu().numpy()
            ori_images_non_encode_np = ori_images_non_encode.squeeze().cpu().numpy()
            images_delta_non_encode_np = images_delta_non_encode.squeeze().cpu().numpy()

            while True:
                # print('images_non_encode_np', images_non_encode_np.shape)
                images_non_encode_np_inv_std = customized_inverse_standardize(np.array([images_non_encode_np]), standardize, m, False)[0]
                print('\n'*10)
                # print(labels_used)
                print(images_non_encode_np_inv_std)
                if_violate, [violated_constraints, involved_labels] = if_violate_constraints(images_non_encode_np_inv_std, customized_constraints, labels_used, verbose=True)
                # if violate, pick violated constraints, project perturbation back to linear constraints via LR
                if if_violate:
                    ever_violate = True
                    # print(len(images_delta_non_encode_np), m)
                    # print(images_delta_non_encode_np)
                    images_delta_non_encode_np_inv_std = customized_inverse_standardize(np.array([images_delta_non_encode_np]), standardize, m, False, True)

                    new_images_delta_non_encode_np_inv_std = project_into_constraints(images_delta_non_encode_np_inv_std[0], violated_constraints, labels_used, involved_labels)

                    # print(ori_images.squeeze().cpu().numpy())
                    print(images_delta_non_encode_np_inv_std[0])
                    print(new_images_delta_non_encode_np_inv_std)
                else:
                    break

                new_images_delta_non_encode_np = customized_standardize(np.array([new_images_delta_non_encode_np_inv_std]), standardize, m, False, True)[0]


                # print(new_images_delta_non_encode_np.shape, new_images_delta_non_encode_np.shape)

                images_non_encode_np = ori_images_non_encode_np + new_images_delta_non_encode_np

                # print(ori_images_non_encode_np)
                # print(new_images_delta_non_encode_np)
                # print(images_non_encode_np)

                ori_images_non_encode_np_inv_std = customized_inverse_standardize(np.array([ori_images_non_encode_np]), standardize, m, False)[0]
                images_non_encode_np_inv_std = customized_inverse_standardize(np.array([images_non_encode_np]), standardize, m, False)[0]
                # print(standardize.mean_, standardize.scale_)
                # print(ori_images_non_encode_np_inv_std)
                # print(new_images_delta_non_encode_np_inv_std)
                print(images_non_encode_np_inv_std)

            if ever_violate:
                images_non_encode = torch.from_numpy(images_non_encode_np).to(device)
                images[:, m:] = images_non_encode



            # elif after perturbation, similar to previous bugs, break

            # else update and continue


            if i == iters - 1:
                print('iter', i, ':', 'cost :', cost.cpu().detach().numpy(), 'outputs :', outputs.cpu().detach().numpy())

        new_images_all.append(prev_images.squeeze().cpu().detach().numpy())
        new_outputs_all.append(prev_outputs.squeeze().cpu().detach().numpy())

    return np.array(new_images_all), np.array(new_outputs_all), np.array(initial_outputs_all)

def train_net(X_train, y_train, X_test, y_test, batch_train=200, batch_test=20, model_type='one_output', device=None):
    if not device:
        device = torch.device("cuda")
    input_size = X_train.shape[1]
    hidden_size = 150

    num_epochs = 20

    if model_type == 'one_output':
        num_classes = 1
        model = SimpleNet(input_size, hidden_size, num_classes)
        criterion = nn.BCELoss()
        one_hot = False
    elif model_type == 'two_output':
        num_classes = 2
        model = SimpleNetMulti(input_size, hidden_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        one_hot = True
    else:
        raise 'unknown model_type '+model_type

    model.cuda()


    # optimizer = torch.optim.LBFGS(model.parameters())
    optimizer = torch.optim.Adam(model.parameters())

    d_train = VanillaDataset(X_train, y_train)
    train_loader = Data.DataLoader(d_train, batch_size=batch_train, shuffle=True)

    if len(y_test) > 0:
        d_test = VanillaDataset(X_test, y_test)
        test_loader = Data.DataLoader(d_test, batch_size=batch_test, shuffle=True)


    # Train the Model
    counter = 0
    for epoch in range(num_epochs):
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            if one_hot:
                y_batch = y_batch.long()

            # LBFGS
            # def closure():
            #     optimizer.zero_grad()
            #     y_pred_batch = model(x_batch).squeeze()
            #     loss = criterion(y_pred_batch, y_batch)
            #     loss.backward()
            #     return loss
            # optimizer.step(closure)


            # Adam
            optimizer.zero_grad()
            y_pred_batch = model(x_batch).squeeze()

            loss = criterion(y_pred_batch, y_batch)
            loss.backward()
            optimizer.step()

            counter += 1
            # if epoch % 1 == 0:
            #     print ('Epoch [%d/%d], Step %d, Loss: %.4f'
            #            %(epoch+1, num_epochs, counter, loss))
            #     print('train', y_pred_batch, y_batch)
            if epoch % 1 == 0 and len(y_test) > 0:
                mean_loss, mean_acc = validation(model, test_loader, device, one_hot)
                print ('Epoch [%d/%d], Step %d, Test Mean Loss: %.4f, Test Mean Accuracy: %.4f'
                       %(epoch+1, num_epochs, counter, mean_loss, mean_acc))
                model.train()

    return model



class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, weights):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(weights)

    def forward(self, x):
        out = self.linear(x)
        return out


def project_into_constraints(x, violated_constraints
, labels, involved_labels):
    assert len(labels) == len(x), str(len(labels))+' VS '+str(len(x))
    labels_to_id = {label:i for i, label in enumerate(labels)}
    print(labels_to_id)
    print(involved_labels)
    involved_ids = np.array([labels_to_id[label] for label in involved_labels])
    map_ids = {involved_id:i for i, involved_id in enumerate(involved_ids)}

    m = len(violated_constraints)
    r = len(involved_ids)
    A_train = np.zeros((m, r))
    x_new = x.copy()
    print('involved_ids', involved_ids)
    x_start = x[involved_ids]
    y_train = np.zeros(m)

    for i, constraint in enumerate(violated_constraints):
        ids = np.array([map_ids[labels_to_id[label]] for label in constraint['labels']])
        A_train[i, ids] = np.array(constraint['coefficients'])
        y_train[i] = constraint['value']

    x_projected = LR(A_train, x_start, y_train)
    x_new[involved_ids] = x_projected
    return x_new

def LR(A_train, x_start, y_train):
    # A_train ~ m * r, constraints
    # x_start ~ r * 1, initial x
    # y_train ~ m * 1, target values
    # m = constraints number
    # r = number of variables involved
    x_start = torch.from_numpy(x_start).cuda().float()

    inputDim = A_train.shape[1]
    outputDim = 1
    learningRate = 0.01
    epochs = 200
    eps = 1e-5
    model = linearRegression(inputDim, outputDim, x_start)
    model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    for epoch in range(epochs):
        inputs = torch.from_numpy(A_train).cuda().float()
        labels = torch.from_numpy(y_train).cuda().float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print('epoch {}, loss {}'.format(epoch, loss.item()))
        if loss.item() < eps:
            break

    return model.linear.weight.data.cpu().numpy()
