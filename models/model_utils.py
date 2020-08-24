

def get_mean_std(data):
    return data.mean(0), data.std(0)


def normalize(data):
    mean, std = get_mean_std(data)
    return mean, std, normalize_by(mean, std, data)


def normalize_by(mean, std, data):
    return (data - mean) / std


def normalize_by_train(train, data):
    mean, std, normalized_train = normalize(train)
    normalized_data = normalize_by(mean, std, data)
    return normalized_train, normalized_data
