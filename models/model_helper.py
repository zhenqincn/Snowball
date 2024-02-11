import torch

def get_model(model_name: str = 'resnet18', num_classes=10, client_id=None) -> torch.nn.Module:
    inner_model_name = model_name.replace('-', '').lower()
    if 'mnistcnn' == inner_model_name:
        from models.cnn_mnist import MNISTCNN
        return MNISTCNN(num_classes=num_classes)
    elif 'cifarnet' == inner_model_name:
        from models.cifarnet import CifarNet
        return CifarNet(num_classes=num_classes)
    elif 'cnn1' == inner_model_name:
        from models.cifarnet import CNN1
        return CNN1(num_classes=num_classes)
    elif 'cnn1_bn' == inner_model_name:
        from models.cifarnet import CNN1_BN
        return CNN1_BN(num_classes=num_classes, client_id=client_id)
    elif 'cnn1_bn_drop' == inner_model_name:
        from models.cifarnet import CNN1_BN_DROP
        return CNN1_BN_DROP(num_classes=num_classes, client_id=client_id)
    elif 'cnn2' == inner_model_name:
        from models.cifarnet import CNN2
        return CNN2(num_classes=num_classes)
    elif 'cnn3' == inner_model_name:
        from models.cifarnet import CNN3
        return CNN3(num_classes=num_classes)
    elif 'cnn2_bn' == inner_model_name:
        from models.cifarnet import CNN2_BN
        return CNN2_BN(num_classes=num_classes, client_id=client_id)
    elif 'cnn2_bn_drop' == inner_model_name:
        from models.cifarnet import CNN2_BN_DROP
        return CNN2_BN_DROP(num_classes=num_classes, client_id=client_id)
    elif 'gru' == inner_model_name:
        from models.m_rnn import GRUModel
        return GRUModel()