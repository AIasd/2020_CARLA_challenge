def visualize_route(route):
    n = len(route)

    x_list = []
    y_list = []

    # The following code prints out the planned route
    for i, (transform, command) in enumerate(route):
        x = transform.location.x
        y = transform.location.y
        z = transform.location.z
        pitch = transform.rotation.pitch
        yaw = transform.rotation.yaw
        if i == 0:
            s = 'start'
            x_s = [x]
            y_s = [y]
        elif i == n-1:
            s = 'end'
            x_e = [x]
            y_e = [y]
        else:
            s = 'point'
            x_list.append(x)
            y_list.append(y)

        # print(s, x, y, z, pitch, yaw, command

    import matplotlib.pyplot as plt
    plt.gca().invert_yaxis()
    plt.scatter(x_list, y_list)
    plt.scatter(x_s, y_s, c='red', linewidths=5)
    plt.scatter(x_e, y_e, c='black', linewidths=5)

    plt.show()


def perturb_route(route, perturbation):
    num_to_perturb = min([len(route), len(perturbation)+2])
    for i in range(num_to_perturb):
        if i != 0 and i != num_to_perturb-1:
            route[i][0].location.x += perturbation[i-1][0]
            route[i][0].location.y += perturbation[i-1][1]


def create_transform(x, y, z, pitch, yaw, roll):
    location = carla.Location(x, y, z)
    rotation = carla.Rotation(pitch, yaw, roll)
    transform = carla.Transform(location, rotation)
    return transform
