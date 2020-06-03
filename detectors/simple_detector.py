from models.VGG16 import VGGnet
from dataloader import get_train_test_num, get_src_tgt_num, get_dataset, Source_Target_Train_Set, Source_Target_Train_Pair_Set, Set

def train(ce_loss, optim, device, net, loader, method):
    # method will be ignored
    net.train()
    for i, (img, label) in enumerate(loader):
        img = img.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        pred, feature = net(img)

        ce = ce_loss(pred, label)
        loss = ce
        optim.zero_grad()
        loss.backward()
        optim.step()

    return loss.item()


def test(device, net, loader):
    correct = 0
    net.eval()
    with torch.no_grad():
        for img, label in loader:
            img = img.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            pred, _ = net(img)
            _, idx = pred.max(dim=1)
            cor = (idx == label).sum().cpu().item()
            correct += cor
            # if (cor):
                # print(label,cor)
    acc = correct / len(loader.dataset)
    # print(correct, len(loader.dataset))
    return acc


if __name__ == '__main__':


    train_weather_indexes = [0]
    train_routes = [i for i in range(10)]

    test_weather_indexes = [0]
    test_routes = [i for i in range(10, 20)]


    # get datasets
    source_train_data, _ = get_dataset(dataset_names[0], source_train_num, -1, random_seed)
    target_train_data, target_test_data = get_dataset(dataset_names[1], target_train_num, -1, random_seed, balanced_class=True)

    source_train = Set(source_train_data, model_arch)
    target_train = Set(target_train_data, model_arch)



    source_target_train = Source_Target_Train_Set(source_train_data, target_train_data, model_arch, [])

    train_set_loader = DataLoader(source_target_train, batch_size=batch, shuffle=True)

    test_loader = DataLoader(target_test, batch_size=batch, shuffle=True)

    #
    device = torch.device("cuda")

     elif model_arch == 'vgg':
            net = VGGnet().to(device)

            if sampling == 'pair':
                optim = torch.optim.Adadelta(net.parameters(), lr=0.1)
            else:
                optim = torch.optim.SGD(net.parameters(), momentum=0.95, lr=0.0003, weight_decay=0.0003)



    best_test_acc = 0
        ce_loss = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            if sampling == 'pair':
                train_loss = train(ce_loss, optim, device, net, train_set_loader, method)
            else:
                train_loss = train_plain(ce_loss, optim, device, net, train_set_loader, method)

            test_acc = test(device, net, test_loader)
            # writer.add_scalar('plain/train_loss', train_loss, epoch)
            # writer.add_scalar('plain/test_acc', test_acc, epoch)
            print('epoch:', epoch, 'test_acc:', test_acc)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                save_model(net, train_baseline_path)
