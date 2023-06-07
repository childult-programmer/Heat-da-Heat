from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(DATA_IN_PATH='/data/datasets/kaist-rgbt',
                      split='visible', # lwir
                      DATA_OUT_PATH='../kaistPD_json')
