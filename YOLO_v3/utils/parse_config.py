def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    module_defs = []
    with open(path, 'r') as f:
        for eachline in f:
            eachline = eachline.strip('\n')
            if eachline and not eachline.startswith('#'):
                eachline = eachline.rstrip().lstrip()
                if eachline.startswith('['):
                    module_defs.append({})
                    module_defs[-1]['type'] = eachline[1:-1].rstrip()
                    if module_defs[-1]['type'] == 'convolutional':
                        module_defs[-1]['batch_normalize'] = 0
                else:
                    key, value = eachline.split("=")
                    value = value.strip()
                    module_defs[-1][key.rstrip()] = value.strip()

    # file = open(path, 'r')
    # lines = file.read().split('\n')
    # lines = [x for x in lines if x and not x.startswith('#')]
    # lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    # module_defs1 = []
    # for line in lines:
    #     if line.startswith('['):  # This marks the start of a new block
    #         module_defs1.append({})
    #         module_defs1[-1]['type'] = line[1:-1].rstrip()
    #         if module_defs1[-1]['type'] == 'convolutional':
    #             module_defs1[-1]['batch_normalize'] = 0
    #     else:
    #         key, value = line.split("=")
    #         value = value.strip()
    #         module_defs1[-1][key.rstrip()] = value.strip()
    # return module_defs, module_defs1
    return module_defs


def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


if __name__ == "__main__":
    # config = parse_model_config("/home/chenxi/tmp/YOLO_v3/config/yolov3.cfg")
    config = parse_model_config("../config/yolov3.cfg")
    print(config.__len__())
    print(config)

