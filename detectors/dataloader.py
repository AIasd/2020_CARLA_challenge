class Set(Dataset):
    def __init__(self, dataset, model_arch):
        self.x, self.y = dataset
        self.model_arch = model_arch

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.model_arch == 'cnn':
            x = torch.from_numpy(x).unsqueeze(0)
        elif self.model_arch == 'vgg':
            x = torch.from_numpy(x)
        return x, y

    def __len__(self):
        return self.x.shape[0]

def load_data(data_dir, weather_indexes, route_indexes):
    for weather in weather_indexes:
        for route in route_indexes:
            data_df = pd.read_csv(os.path.join(data_dir, 'route_'+str(route)+'_'+str(weather), 'driving_log.csv'))
            if x_center is None:
                x_center = data_df['center'].values
                y = data_df['steering'].values
                x_left = data_df['left'].values
                x_right = data_df['right'].values
            else:
                x_center = numpy.concatenate((x_center, data_df['center'].values), axis=0)
                y = numpy.concatenate((y, data_df['steering'].values), axis=0)
                x_left = numpy.concatenate((x_left, data_df['left'].values), axis=0)
                x_right = numpy.concatenate((x_right, data_df['right'].values), axis=0)

    if restrict_size and len(x_center) > self.args.train_abs_size != -1 and not eval_data_mode:
    shuffle_seed = numpy.random.randint(low=1)
    numpy.random.seed(shuffle_seed)
    numpy.random.shuffle(x_center)
    numpy.random.seed(shuffle_seed)
    numpy.random.shuffle(y)
    per_image_size = int(self.args.train_abs_size / 3)
    x_center = x_center[:per_image_size]
    if self.args.simulator != 'carla_096':
    numpy.random.seed(shuffle_seed)
    numpy.random.shuffle(x_left)
    numpy.random.seed(shuffle_seed)
    numpy.random.shuffle(x_right)
    x_left = x_left[:per_image_size]
    x_right = x_right[:per_image_size]
    y = y[:self.args.train_abs_size]

    if eval_data_mode:
    return x_center, frame_ids, are_crashes
    else:
    print("Train dataset: " + str(len(x_center)) + " elements")
    if self.args.simulator == 'carla_096':
    images = x_center
    labels = x_center
    else:
    images = numpy.concatenate((x_left, x_center, x_right))
    labels = numpy.concatenate((x_left, x_center, x_right))

    return images, labels
