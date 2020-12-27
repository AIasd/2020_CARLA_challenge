import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.utils
from torchvision import models

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
        out = torch.stack([1 - out, out], dim=1)
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



def pgd_attack(model, images, labels, xl, xu, encode_fields, device=None, eps=1.01, alpha=4/255, iters=100):
    if not device:
        device = torch.device("cuda")
    images = torch.from_numpy(images).to(device).float()
    labels = torch.from_numpy(labels).to(device).float()



    xl = torch.from_numpy(xl).to(device).float()
    xu = torch.from_numpy(xu).to(device).float()

    loss = nn.BCELoss()

    ori_images = images.data
    m = np.sum(encode_fields)
    # print('\n'*2)
    for i in range(iters):

        images.requires_grad = True
        outputs = model(images).squeeze()
        model.zero_grad()

        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clip(adv_images - ori_images, min=-eps, max=eps)
        images = torch.max(torch.min(ori_images + eta, xu), xl).detach_()


        one_hotezed_images_embed = torch.zeros([images.shape[0], m])
        s = 0
        # print(encode_fields)
        for field_len in encode_fields:
            max_inds = torch.argmax(images[:, s:s+field_len], axis=1)
            one_hotezed_images_embed[torch.arange(images.shape[0]), s+max_inds] = 1
            # print(images.cpu().detach().numpy())
            # print(field_len, max_inds.cpu().detach().numpy())
            # print(one_hotezed_images_embed.cpu().detach().numpy())
            s += field_len
        images[:, :m] = one_hotezed_images_embed
        print('iter', i, ':', 'cost :', cost.cpu().detach().numpy(), 'outputs :', outputs.cpu().detach().numpy())


    return images.cpu().detach().numpy(), outputs.cpu().detach().numpy()

def train_net(X_train, y_train, X_test, y_test, model_type='one_output', device=None):
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
    train_loader = Data.DataLoader(d_train, batch_size=200, shuffle=True)

    if len(y_test) > 0:
        d_test = VanillaDataset(X_test, y_test)
        test_loader = Data.DataLoader(d_test, batch_size=10, shuffle=True)


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
