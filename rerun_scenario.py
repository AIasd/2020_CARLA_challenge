import os
from ga_fuzzing import run_simulation
from object_types import pedestrian_types, vehicle_types, static_types, vehicle_colors
import random
import pickle
import numpy as np
from datetime import datetime
from customized_utils import make_hierarchical_dir, convert_x_to_customized_data, exit_handler, customized_routes, parse_route_and_scenario, check_bug
import atexit

import traceback
from distutils.dir_util import copy_tree

from regression_analysis import get_sorted_subfolders, load_data


import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.utils
from torchvision import models

from sklearn.preprocessing import StandardScaler



os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_deterministic(True)
torch.backends.cudnn.benchmark = False


os.environ['HAS_DISPLAY'] = '0'
# '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
port = 2033
ego_car_model = 'lbc'
is_save = True


# def get_gan_configs():
#
#     n_epoch = 200
#     batch_size = 64
#     lr = 0.0002
#     b1 = 0.5
#     b2 = 0.999
#     n_cpu = 8
#     latent_dim = 100
#     n_classes = 10
#     img_size = 32
#     channels = 1
#     sample_interval = 400
#
#
#     img_shape = (channels, img_size, img_size)
#
#     cuda = True if torch.cuda.is_available() else False
#
#
#     class Generator(nn.Module):
#         def __init__(self):
#             super(Generator, self).__init__()
#
#             self.label_emb = nn.Embedding(n_classes, n_classes)
#
#             def block(in_feat, out_feat, normalize=True):
#                 layers = [nn.Linear(in_feat, out_feat)]
#                 if normalize:
#                     layers.append(nn.BatchNorm1d(out_feat, 0.8))
#                 layers.append(nn.LeakyReLU(0.2, inplace=True))
#                 return layers
#
#             self.model = nn.Sequential(
#                 *block(latent_dim + n_classes, 128, normalize=False),
#                 *block(128, 256),
#                 *block(256, 512),
#                 *block(512, 1024),
#                 nn.Linear(1024, int(np.prod(img_shape))),
#                 nn.Tanh()
#             )
#
#         def forward(self, noise, labels):
#             # Concatenate label embedding and image to produce input
#             gen_input = torch.cat((self.label_emb(labels), noise), -1)
#             img = self.model(gen_input)
#             img = img.view(img.size(0), *img_shape)
#             return img
#
#
#     class Discriminator(nn.Module):
#         def __init__(self):
#             super(Discriminator, self).__init__()
#
#             self.label_embedding = nn.Embedding(n_classes, n_classes)
#
#             self.model = nn.Sequential(
#                 nn.Linear(n_classes + int(np.prod(img_shape)), 512),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.Linear(512, 512),
#                 nn.Dropout(0.4),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.Linear(512, 512),
#                 nn.Dropout(0.4),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.Linear(512, 1),
#             )
#
#         def forward(self, img, labels):
#             # Concatenate label embedding and image to produce input
#             d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
#             validity = self.model(d_in)
#             return validity
#
#
#     # Loss functions
#     adversarial_loss = torch.nn.MSELoss()
#
#     # Initialize generator and discriminator
#     generator = Generator()
#     discriminator = Discriminator()
#
#     if cuda:
#         generator.cuda()
#         discriminator.cuda()
#         adversarial_loss.cuda()
#
#     # Configure data loader
#     os.makedirs("../../data/mnist", exist_ok=True)
#     dataloader = torch.utils.data.DataLoader(
#         datasets.MNIST(
#             "../../data/mnist",
#             train=True,
#             download=True,
#             transform=transforms.Compose(
#                 [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#             ),
#         ),
#         batch_size=batch_size,
#         shuffle=True,
#     )
#
#     # Optimizers
#     optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
#     optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
#
#     FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#     LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
#
#
#     def sample_image(n_row, batches_done):
#         """Saves a grid of generated digits ranging from 0 to n_classes"""
#         # Sample noise
#         z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))
#         # Get labels ranging from 0 to n_classes for n rows
#         labels = np.array([num for _ in range(n_row) for num in range(n_row)])
#         labels = Variable(LongTensor(labels))
#         gen_imgs = generator(z, labels)
#         save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)
#
#
#     # ----------
#     #  Training
#     # ----------
#     for epoch in range(n_epochs):
#         for i, (imgs, labels) in enumerate(dataloader):
#
#             batch_size = imgs.shape[0]
#
#             # Adversarial ground truths
#             valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
#             fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
#
#             # Configure input
#             real_imgs = Variable(imgs.type(FloatTensor))
#             labels = Variable(labels.type(LongTensor))
#
#             # -----------------
#             #  Train Generator
#             # -----------------
#
#             optimizer_G.zero_grad()
#
#             # Sample noise and labels as generator input
#             z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
#             gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))
#
#             # Generate a batch of images
#             gen_imgs = generator(z, gen_labels)
#
#             # Loss measures generator's ability to fool the discriminator
#             validity = discriminator(gen_imgs, gen_labels)
#             g_loss = adversarial_loss(validity, valid)
#
#             g_loss.backward()
#             optimizer_G.step()
#
#             # ---------------------
#             #  Train Discriminator
#             # ---------------------
#
#             optimizer_D.zero_grad()
#
#             # Loss for real images
#             validity_real = discriminator(real_imgs, labels)
#             d_real_loss = adversarial_loss(validity_real, valid)
#
#             # Loss for fake images
#             validity_fake = discriminator(gen_imgs.detach(), gen_labels)
#             d_fake_loss = adversarial_loss(validity_fake, fake)
#
#             # Total discriminator loss
#             d_loss = (d_real_loss + d_fake_loss) / 2
#
#             d_loss.backward()
#             optimizer_D.step()
#
#             print(
#                 "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#                 % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
#             )
#
#             batches_done = epoch * len(dataloader) + i
#             if batches_done % sample_interval == 0:
#                 sample_image(n_row=10, batches_done=batches_done)




def get_adv_configs(X_train, y_train, X_test, y_test, xl, xu, standardize, mask):
    # use_cuda = True
    # device = torch.device("cuda" if use_cuda else "cpu")
    # input_size = X_train.shape[1]
    # hidden_size = 100
    # num_classes = 1
    # num_epochs = 500
    #
    #
    # class VanillaDataset(Data.Dataset):
    #     def __init__(self, X, y):
    #         self.X = X
    #         self.y = y
    #
    #     def __len__(self):
    #         return len(self.y)
    #
    #     def __getitem__(self, idx):
    #         return (self.X[idx], self.y[idx])
    #
    # class Net(nn.Module):
    #     def __init__(self, input_size, hidden_size, num_classes):
    #         super(Net, self).__init__()
    #         self.fc1 = nn.Linear(input_size, hidden_size)
    #         self.tanh = nn.Tanh()
    #         self.fc2 = nn.Linear(hidden_size, num_classes)
    #         self.sigmoid = nn.Sigmoid()
    #     def forward(self, x):
    #         out = self.fc1(x)
    #         out = self.tanh(out)
    #         out = self.fc2(out)
    #         out = self.sigmoid(out)
    #         return out
    #
    # def pgd_attack(model, images, labels, xl, xu, eps=0.3, alpha=2/255, iters=40) :
    #     images = torch.from_numpy(images).to(device).float()
    #     labels = torch.from_numpy(labels).to(device).float()
    #
    #     xl = torch.from_numpy(xl).to(device).float()
    #     xu = torch.from_numpy(xu).to(device).float()
    #
    #     loss = nn.BCELoss()
    #
    #     ori_images = images.data
    #
    #     for i in range(iters) :
    #         images.requires_grad = True
    #         outputs = model(images)
    #         model.zero_grad()
    #
    #         cost = loss(outputs, labels).to(device)
    #         cost.backward()
    #
    #         adv_images = images + alpha*images.grad.sign()
    #         eta = torch.clip(adv_images - ori_images, min=-eps, max=eps)
    #         images = torch.max(torch.min(ori_images + eta, xu), xl).detach_()
    #
    #     images = standardize.inverse_transform(images.cpu())[0]
    #     images[mask=='int'] = np.round(images[mask=='int'])
    #
    #     return images
    #
    # def validation(model, test_loader):
    #     mean_loss = []
    #     model.eval()
    #     for i, (x_batch, y_batch) in enumerate(test_loader):
    #         x_batch = x_batch.to(device).float()
    #         y_batch = y_batch.to(device).float()
    #
    #         y_pred_batch = model(x_batch).squeeze()
    #         loss = criterion(y_pred_batch, y_batch)
    #         mean_loss.append(loss.cpu().detach().numpy())
    #         print('test', y_pred_batch, y_batch)
    #
    #     mean_loss = np.mean(mean_loss)
    #
    #     return mean_loss
    #
    # model = Net(input_size, hidden_size, num_classes)
    # model.cuda()
    #
    # criterion = nn.BCELoss()
    #
    #
    # # optimizer = torch.optim.LBFGS(model.parameters())
    # optimizer = torch.optim.Adam(model.parameters())
    #
    #
    # d_train = VanillaDataset(X_train, y_train)
    # train_loader = Data.DataLoader(d_train, batch_size=20, shuffle=True)
    #
    # d_test = VanillaDataset(X_test, y_test)
    # test_loader = Data.DataLoader(d_test, batch_size=5, shuffle=True)
    #
    #
    #
    #
    #
    #
    # # Train the Model
    # counter = 0
    # for epoch in range(num_epochs):
    #     for i, (x_batch, y_batch) in enumerate(train_loader):
    #         x_batch = x_batch.to(device).float()
    #         y_batch = y_batch.to(device).float()
    #
    #         # LBFGS
    #         # def closure():
    #         #     optimizer.zero_grad()
    #         #     y_pred_batch = model(x_batch).squeeze()
    #         #     loss = criterion(y_pred_batch, y_batch)
    #         #     loss.backward()
    #         #     return loss
    #         # optimizer.step(closure)
    #
    #
    #         # Adam
    #         optimizer.zero_grad()
    #         y_pred_batch = model(x_batch).squeeze()
    #         loss = criterion(y_pred_batch, y_batch)
    #         loss.backward()
    #         optimizer.step()
    #
    #         counter += 1
    #         if counter % 10 == 0:
    #             print ('Epoch [%d/%d], Step %d, Loss: %.4f'
    #                    %(epoch+1, num_epochs, counter, loss))
    #             print('train', y_pred_batch, y_batch)
    #         if counter % 100 == 0:
    #             mean_loss = validation(model, test_loader)
    #             print ('Epoch [%d/%d], Step %d, Test Mean Loss: %.4f'
    #                    %(epoch+1, num_epochs, counter, mean_loss))
    #             model.train()
    #
    #
    # # adv attack the trained mlp to get adversarial configs
    # model.eval()
    # label = 1
    test_x_adv_list = []
    #
    # for test_x in X_test:
    #     test_x_adv = pgd_attack(model, np.array([test_x]), np.array([[label]]), xl, xu)
    #     test_x_adv_list.append(test_x_adv)


    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', activation='tanh', max_iter=10000)
    clf.fit(X_train, y_train)
    # score = clf.score(X_test, y_test)
    # y_pred = clf.predict(X_test)
    y_pred = clf.predict_proba(X_test)[:, 1]
    print(y_test, y_pred)
    print(np.mean(np.abs(y_test-y_pred)))

    # from customized_utils import draw_auc_roc_for_scores
    # draw_auc_roc_for_scores(-1*y_pred, y_test)


    return test_x_adv_list



def rerun_simulation(pickle_filename, is_save, rerun_save_folder, ind, sub_folder_name, scenario_file, ego_car_model='lbc', x=[]):
    is_bug = False

    # parameters preparation
    if ind == 0:
        launch_server = True
    else:
        launch_server = False

    with open(pickle_filename, 'rb') as f_in:
        d = pickle.load(f_in)['info']

        if len(x) == 0:
            x = d['x']
            # TBD: save port separately so we won't need to repetitvely save cur_info in ga_fuzzing
            x[-1] = port

        waypoints_num_limit = d['waypoints_num_limit']
        max_num_of_static = d['num_of_static_max']
        max_num_of_pedestrians = d['num_of_pedestrians_max']
        max_num_of_vehicles = d['num_of_vehicles_max']
        customized_center_transforms = d['customized_center_transforms']

        if 'parameters_min_bounds' in d:
            parameters_min_bounds = d['parameters_min_bounds']
            parameters_max_bounds = d['parameters_max_bounds']
        else:
            parameters_min_bounds = None
            parameters_max_bounds = None


        episode_max_time = 60
        call_from_dt = d['call_from_dt']
        town_name = d['town_name']
        scenario = d['scenario']
        direction = d['direction']
        route_str = d['route_str']
        route_type = d['route_type']


    folder = '_'.join([route_type, scenario, ego_car_model, route_str])
    route_info = customized_routes[route_type]
    location_list = route_info['location_list']
    parse_route_and_scenario(location_list, town_name, scenario, direction, route_str, scenario_file)




    customized_data = convert_x_to_customized_data(x, waypoints_num_limit, max_num_of_static, max_num_of_pedestrians, max_num_of_vehicles, static_types, pedestrian_types, vehicle_types, vehicle_colors, customized_center_transforms, parameters_min_bounds, parameters_max_bounds)
    print('x', x)


    objectives, loc, object_type, route_completion, info, save_path = run_simulation(customized_data, launch_server, episode_max_time, call_from_dt, town_name, scenario, direction, route_str, scenario_file, ego_car_model, rerun=True)



    is_bug = check_bug(objectives)

    # save data
    if is_save:
        rerun_bugs_folder = make_hierarchical_dir([rerun_save_folder, folder, 'rerun_bugs'])
        rerun_non_bugs_folder = make_hierarchical_dir([rerun_save_folder, folder, 'rerun_non_bugs'])

        if is_bug:

            print('\n'*3, 'rerun also causes a bug!!!', '\n'*3)
            try:
                # use this version to merge into the existing folder
                copy_tree(save_path, os.path.join(rerun_bugs_folder, sub_folder_name))
            except:
                print('fail to copy from', save_path)
                traceback.print_exc()
        else:
            try:
                # use this version to merge into the existing folder
                copy_tree(save_path, os.path.join(rerun_non_bugs_folder, sub_folder_name))
            except:
                print('fail to copy from', save_path)
                traceback.print_exc()

    return is_bug, objectives



def rerun_list_of_scenarios(rerun_save_folder, scenario_file, data, mode):
    if data == 'bugs':
        folder = 'run_results/nsga2-un/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/800_partial_collision/bugs'
    elif data == 'non_bugs':
        folder = 'run_results/nsga2-un/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/800_partial_collision/non_bugs'

    subfolder_names = [sub_folder_name for sub_folder_name in sorted(os.listdir(folder))]


    random.seed(0)
    random.shuffle(subfolder_names)


    # assert len(subfolder_names) >= 2
    # mid = int(len(subfolder_names)//2)
    mid = 118

    train_subfolder_names = subfolder_names[:mid]
    test_subfolder_names = subfolder_names[mid:]

    print(train_subfolder_names)

    if mode == 'train':
        chosen_subfolder_names = train_subfolder_names
    elif mode == 'test':
        chosen_subfolder_names = test_subfolder_names
    elif mode == 'all':
        chosen_subfolder_names = subfolder_names

    bug_num = 0
    objectives_avg = 0
    chosen_subfolder_names = chosen_subfolder_names
    for ind, sub_folder_name in enumerate(chosen_subfolder_names):
        print('episode:', ind+1, '/', len(chosen_subfolder_names), 'bug num:', bug_num)
        sub_folder = os.path.join(folder, sub_folder_name)
        if os.path.isdir(sub_folder):
            pickle_filename = os.path.join(sub_folder, 'cur_info.pickle')
            if os.path.exists(pickle_filename):
                print(pickle_filename)
                is_bug, objectives = rerun_simulation(pickle_filename, is_save, rerun_save_folder, ind, sub_folder_name, scenario_file, ego_car_model=ego_car_model)


                objectives_avg += np.array(objectives)

                if is_bug:
                    bug_num += 1

    print('bug_ratio :', bug_num / len(chosen_subfolder_names))
    print('objectives_avg :', objectives_avg / len(chosen_subfolder_names))


if __name__ == '__main__':
    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


    scenario_folder = 'scenario_files'
    if not os.path.exists('scenario_files'):
        os.mkdir(scenario_folder)
    scenario_file = scenario_folder+'/'+'current_scenario_'+time_str+'.json'

    atexit.register(exit_handler, [port])

    # ['rerun', 'adv']
    task = 'adv'

    if task == 'rerun':
        # ['bugs', 'non_bugs']
        data = 'bugs'
        # ['train', 'test', 'all']
        mode = 'test'
        rerun_save_folder = make_hierarchical_dir(['rerun', data, mode, time_str])
        rerun_list_of_scenarios(rerun_save_folder, scenario_file, data, mode)

    elif task == 'adv':
        rerun_save_folder = make_hierarchical_dir(['adv', time_str])
        cutoff = 300
        cutoff_end = 400
        parent_folder = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/run_results/nsga2/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/50_8_all'

        pickle_filename = parent_folder + '/bugs/1/cur_info.pickle'
        with open(pickle_filename, 'rb') as f_in:
            d = pickle.load(f_in)
        # hack: since we are only using the elements after the first five

        xl_ori = d['xl'][5:]
        xu_ori =d['xu'][5:]
        mask = d['mask'][5:]
        x_5 = d['x'][:5]

        subfolders = get_sorted_subfolders(parent_folder)
        X, y, _ = load_data(subfolders)


        X_train, X_test = X[:cutoff], X[cutoff:cutoff_end]
        y_train, y_test = y[:cutoff], y[cutoff:cutoff_end]
        standardize = StandardScaler()
        X_train = standardize.fit_transform(X_train)
        X_test = standardize.transform(X_test)
        xl = standardize.transform([xl_ori])[0]
        xu = standardize.transform([xu_ori])[0]


        test_x_adv_list = get_adv_configs(X_train, y_train, X_test, y_test, xl, xu, standardize, mask)

        is_bug_list = []

        for i, test_x_adv in enumerate(test_x_adv_list):
            np.clip(test_x_adv, xl_ori, xu_ori)
            test_x_adv = np.append(test_x_adv, port)
            test_x_adv = np.append(x_5, test_x_adv)

            # is_bug, objectives = rerun_simulation(pickle_filename, True, rerun_save_folder, i, str(i), scenario_file, ego_car_model=ego_car_model, x=test_x_adv)
            #
            # is_bug_list.append(check_bug(objectives))
            # print(np.sum(is_bug_list), '/', len(is_bug_list))
