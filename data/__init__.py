import data.img_transforms as T
from data.dataloader import DataLoaderX
from data.dataset_loader import ImageDataset, VideoDataset
from data.samplers import DistributedRandomIdentitySampler, DistributedInferenceSampler
from data.datasets.ltcc import LTCC
from data.datasets.prcc import PRCC
from data.datasets.Celeb_light import Celeb_light
from data.datasets.last import LaST
from data.datasets.vc_clothes import VC_Clothes
from torch.utils.data import ConcatDataset, DataLoader


__factory = {
    'ltcc': LTCC,
    'prcc': PRCC,
    'celeb_light': Celeb_light,
    'last': LaST,
    'vc_clothes': VC_Clothes,
}



def get_names():
    return list(__factory.keys())


def build_dataset(config):
    return __factory[config.DATA.DATASET](
        root=config.DATA.ROOT,
        aux_info=config.DATA.AUX_INFO,
        meta_dir=config.DATA.META_DIR,
        meta_dims=(config.MODEL.META_DIMS[0] if (config.DATA.AUX_INFO and len(config.MODEL.META_DIMS)>0) else 0),
    )



def build_img_transforms(config):
    transform_train = T.Compose([
        T.Resize((config.DATA.IMG_HEIGHT , config.DATA.IMG_WIDTH)),
        T.RandomCroping(p=config.AUG.RC_PROB),
        T.RandomHorizontalFlip(p=config.AUG.RF_PROB),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=config.AUG.RE_PROB)
    ])
    transform_test = T.Compose([
        T.Resize((config.DATA.IMG_HEIGHT , config.DATA.IMG_WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform_train, transform_test



def build_dataloader(config):
    dataset = build_dataset(config)
    # image dataset
    transform_train, transform_test = build_img_transforms(config)
    # transform_train = build_transform(config,is_train=True)
    # transform_test = build_transform(config,is_train=False)
    train_sampler = DistributedRandomIdentitySampler(dataset.train,
                                                    #  batch_size=config.DATA.BATCH_SIZE,
                                                     num_instances=config.DATA.NUM_INSTANCES,
                                                     seed=config.SOLVER.SEED)
    trainloader = DataLoaderX(dataset=ImageDataset(dataset.train, transform=transform_train,aux_info=config.DATA.AUX_INFO),
                             sampler=train_sampler,
                             batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS,
                             pin_memory=config.DATA.PIN_MEMORY, drop_last=True)

    galleryloader = DataLoaderX(dataset=ImageDataset(dataset.gallery, transform=transform_test,aux_info=config.DATA.AUX_INFO),
                               sampler=DistributedInferenceSampler(dataset.gallery),
                               batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                               pin_memory=config.DATA.PIN_MEMORY, drop_last=False, shuffle=False)
    if config.DATA.DATASET == 'prcc':
        queryloader_same = DataLoaderX(dataset=ImageDataset(dataset.query_same, transform=transform_test,aux_info=config.DATA.AUX_INFO),
                                 sampler=DistributedInferenceSampler(dataset.query_same),
                                batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=config.DATA.PIN_MEMORY,drop_last=False, shuffle=False)
        queryloader_diff = DataLoaderX(dataset=ImageDataset(dataset.query_diff, transform=transform_test,aux_info=config.DATA.AUX_INFO),
                                 sampler=DistributedInferenceSampler(dataset.query_diff),
                                 batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                 pin_memory=True, drop_last=False, shuffle=False)


        combined_dataset = ConcatDataset([queryloader_diff.dataset, galleryloader.dataset])

        val_loader = DataLoader(
            dataset=combined_dataset,
            batch_size=config.DATA.TEST_BATCH,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=False,
            drop_last=False,
            shuffle=False
        )

        combined_dataset_same = ConcatDataset([queryloader_same.dataset, galleryloader.dataset])

        val_loader_same = DataLoader(
            dataset=combined_dataset_same,
            batch_size=config.DATA.TEST_BATCH,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=False,
            drop_last=False,
            shuffle=False
        )

        return trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler,val_loader,val_loader_same
    if config.DATA.DATASET == 'vc_clothes':
        # 1. 构建 queryloader (根据 config.DATA.MODE，通常是 CC)
        queryloader_diff = DataLoaderX(
            dataset=ImageDataset(dataset.query, transform=transform_test, aux_info=config.DATA.AUX_INFO),
            sampler=DistributedInferenceSampler(dataset.query),
            batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=True, drop_last=False, shuffle=False
        )

        # 2. 对于 VC-Clothes，如果需要专门的 same-clothes 验证集
        # 注意：这里需要确保 VC_Clothes 类能提供对应的列表，或者重新调用处理函数
        # 推荐做法：在 build_dataloader 内部为 SC 模式重新构建一个临时 dataset 对象获取列表
        sc_query, sc_gallery, _, _, _, _ = dataset._process_dir_test(dataset.query_dir, dataset.gallery_dir, mode='sc')
        
        queryloader_same = DataLoaderX(
            dataset=ImageDataset(sc_query, transform=transform_test, aux_info=config.DATA.AUX_INFO),
            sampler=DistributedInferenceSampler(sc_query),
            batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=True, drop_last=False, shuffle=False
        )

        # 3. 构建 galleryloader
        galleryloader = DataLoaderX(
            dataset=ImageDataset(dataset.gallery, transform=transform_test, aux_info=config.DATA.AUX_INFO),
            sampler=DistributedInferenceSampler(dataset.gallery),
            batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=True, drop_last=False, shuffle=False
        )

        # 4. 组合 val_loaders 用于 processor.py 中的双重测试
        val_loader = DataLoader(
            dataset=ConcatDataset([queryloader_diff.dataset, galleryloader.dataset]),
            batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=False, drop_last=False, shuffle=False
        )

        val_loader_same = DataLoader(
            dataset=ConcatDataset([queryloader_same.dataset, galleryloader.dataset]),
            batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=False, drop_last=False, shuffle=False
        )

        return trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler, val_loader, val_loader_same
    else:
        queryloader = DataLoaderX(dataset=ImageDataset(dataset.query, transform=transform_test,aux_info=config.DATA.AUX_INFO),
                                 sampler=DistributedInferenceSampler(dataset.query),
                                 batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                 pin_memory=True, drop_last=False, shuffle=False)

        combined_dataset = ConcatDataset([queryloader.dataset, galleryloader.dataset])

        val_loader = DataLoader(
            dataset=combined_dataset,
            batch_size=config.DATA.TEST_BATCH,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=False,
            drop_last=False,
            shuffle=False
        )



        # return trainloader, queryloader, galleryloader, dataset, train_sampler,val_loader

        return trainloader, queryloader, None, galleryloader, dataset, train_sampler, val_loader, None
    

    
