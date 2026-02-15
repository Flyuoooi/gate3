# # vc_clothes.py
# import os
# import re
# import glob
# import logging
# import numpy as np
# import os.path as osp


# class VC_Clothes(object):
#     """
#     VC-Clothes

#     Reference:
#         Wang et al. When Person Re-identification Meets Changing Clothes. CVPRW, 2020.
#     Common folder layout:
#         VC-Clothes/
#           train/*.jpg
#           query/*.jpg
#           gallery/*.jpg
#           (optional) human_parsing/{train,query,gallery}/*.npy
#           (optional) PAR_PETA_105.txt (or your meta file)
#     """
#     dataset_dir = 'VC-Clothes'

#     def __init__(self, root='data', aux_info=False, meta_dir='PAR_PETA_105.txt', meta_dims=105, **kwargs):
#         """
#         Keep same signature style as LTCC/PRCC/Celeb_light:
#             (root, aux_info, meta_dir, meta_dims, **kwargs)

#         kwargs supported (optional):
#             mode: 'all' | 'sc' | 'cc'
#                 - 'all': use all cameras
#                 - 'sc' : same-clothes protocol (cam2 & cam3)
#                 - 'cc' : clothes-changing protocol (cam3 & cam4)
#             with_mask: bool (default False)
#                 - if True and aux_info=False, append mask_path as 5th item
#                 - if aux_info=True, mask will be ignored to keep tuple format stable
#         """
#         self.dataset_dir = osp.join(root, self.dataset_dir)
#         self.aux_info = aux_info
#         self.meta_dir = meta_dir
#         self.meta_dims = meta_dims

#         self.train_dir = osp.join(self.dataset_dir, 'train')
#         self.query_dir = osp.join(self.dataset_dir, 'query')
#         self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

#         self.mode = kwargs.get('mode', 'all')
#         self.with_mask = bool(kwargs.get('with_mask', False))

#         # optional human parsing masks (same as your old vcclothes.py naming)
#         self.train_mask_dir = osp.join(self.dataset_dir, 'human_parsing/train')
#         self.query_mask_dir = osp.join(self.dataset_dir, 'human_parsing/query')
#         self.gallery_mask_dir = osp.join(self.dataset_dir, 'human_parsing/gallery')

#         self._check_before_run()

#         train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
#             self._process_dir_train(self.train_dir)
#         query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
#             self._process_dir_test(self.query_dir, self.gallery_dir)

#         num_total_pids = num_train_pids + num_test_pids
#         num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
#         num_test_imgs = num_query_imgs + num_gallery_imgs
#         num_total_clothes = num_train_clothes + num_test_clothes

#         logger = logging.getLogger('reid.dataset')
#         logger.info("=> VC-Clothes loaded")
#         logger.info("Dataset statistics:")
#         logger.info("  ----------------------------------------")
#         logger.info("  subset   | # ids | # images | # clothes")
#         logger.info("  ----------------------------------------")
#         logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
#         logger.info("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
#         logger.info("  query    | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs))
#         logger.info("  gallery  | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
#         logger.info("  ----------------------------------------")
#         logger.info("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
#         logger.info("  ----------------------------------------")

#         self.train = train
#         self.query = query
#         self.gallery = gallery

#         self.num_train_pids = num_train_pids
#         self.num_train_clothes = num_train_clothes
#         self.num_query_imgs = num_query_imgs

#         # align naming with your other datasets
#         self.pid2clothes = pid2clothes
#         # backward compat with your old vcclothes.py (it used pid2cloth)
#         self.pid2cloth = pid2clothes

#     def _check_before_run(self):
#         if not osp.exists(self.dataset_dir):
#             raise RuntimeError("'{}' is not available".format(self.dataset_dir))
#         if not osp.exists(self.train_dir):
#             raise RuntimeError("'{}' is not available".format(self.train_dir))
#         if not osp.exists(self.query_dir):
#             raise RuntimeError("'{}' is not available".format(self.query_dir))
#         if not osp.exists(self.gallery_dir):
#             raise RuntimeError("'{}' is not available".format(self.gallery_dir))

#         # meta file only required when aux_info=True (same as Celeb_light fix idea)
#         if self.aux_info:
#             meta_path = osp.join(self.dataset_dir, self.meta_dir)
#             if not osp.exists(meta_path):
#                 raise RuntimeError(f"aux_info=True but meta file not found: {meta_path}")

#         # masks are optional; only check existence when user explicitly wants them and aux_info=False
#         if (not self.aux_info) and self.with_mask:
#             for d in [self.train_mask_dir, self.query_mask_dir, self.gallery_mask_dir]:
#                 if not osp.exists(d):
#                     raise RuntimeError(f"with_mask=True but mask dir not found: {d}")

#     @staticmethod
#     def _normpath(p: str) -> str:
#         return osp.normpath(p)

#     def _load_meta_map(self):
#         """
#         meta file format expected (same style as your LTCC/PRCC):
#             <img_path> <attribute_id> <is_present>

#         We store keys for both:
#             - raw string as in file
#             - normalized absolute path (joined with dataset_dir if relative)
#         """
#         if not self.aux_info:
#             return None

#         meta_path = osp.join(self.dataset_dir, self.meta_dir)
#         img2attr = {}

#         with open(meta_path, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 imgdir, attribute_id, is_present = line.split()
#                 aid = int(attribute_id)
#                 if aid >= self.meta_dims:
#                     # protect when meta_dims config mismatched
#                     continue

#                 # build both keys
#                 keys = [imgdir]
#                 if not osp.isabs(imgdir):
#                     keys.append(self._normpath(osp.join(self.dataset_dir, imgdir)))
#                 else:
#                     keys.append(self._normpath(imgdir))

#                 for k in keys:
#                     if k not in img2attr:
#                         img2attr[k] = [0 for _ in range(self.meta_dims)]
#                     img2attr[k][aid] = int(is_present)

#         return img2attr

#     def _cam_allowed(self, camid_raw: int) -> bool:
#         """
#         camid_raw is 1-based in filename (before '-1' normalization).
#         mode follows your old vcclothes.py:
#             - sc: cam2 & cam3
#             - cc: cam3 & cam4
#             - all: all cams
#         """
#         if self.mode == 'sc':
#             return camid_raw in [2, 3]
#         if self.mode == 'cc':
#             return camid_raw in [3, 4]
#         return True

#     def _process_dir_train(self, dir_path):
#         img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
#         img_paths.sort()
#         pattern = re.compile(r'(\d+)-(\d+)-(\d+)-(\d+)')

#         imgdir2attribute = self._load_meta_map()

#         pid_container = set()
#         clothes_container = set()
#         for img_path in img_paths:
#             m = pattern.search(img_path)
#             if m is None:
#                 continue
#             pid, camid, clothes, _ = m.groups()
#             # keep same clothes-id definition as your vcclothes.py: clothes_id = pid + clothes (string concat)
#             clothes_id = pid + clothes
#             pid = int(pid)
#             pid_container.add(pid)
#             clothes_container.add(clothes_id)

#         pid_container = sorted(pid_container)
#         clothes_container = sorted(clothes_container)
#         pid2label = {pid: label for label, pid in enumerate(pid_container)}
#         clothes2label = {cid: label for label, cid in enumerate(clothes_container)}

#         num_pids = len(pid_container)
#         num_clothes = len(clothes_container)

#         dataset = []
#         pid2clothes = np.zeros((num_pids, num_clothes))

#         for img_path in img_paths:
#             m = pattern.search(img_path)
#             if m is None:
#                 continue
#             pid, camid, clothes, _ = m.groups()
#             clothes_id = pid + clothes
#             pid_int, camid_int = int(pid), int(camid)

#             # VC-Clothes train typically uses all cams; if you want strict protocol filtering, you can enable here,
#             # but we keep consistent with your old vcclothes.py: only filter in test.
#             camid0 = camid_int - 1  # to 0-based
#             pid_lbl = pid2label[pid_int]
#             clothes_lbl = clothes2label[clothes_id]

#             if self.aux_info:
#                 k_abs = self._normpath(img_path)
#                 aux = imgdir2attribute.get(img_path, imgdir2attribute.get(k_abs))
#                 if aux is None:
#                     # if meta missing, still keep shape stable
#                     aux = [0 for _ in range(self.meta_dims)]
#                 dataset.append((img_path, pid_lbl, camid0, clothes_lbl, aux))
#             else:
#                 if self.with_mask:
#                     mask_path = osp.join(self.train_mask_dir, osp.basename(img_path)[:-4] + '.npy')
#                     dataset.append((img_path, pid_lbl, camid0, clothes_lbl, mask_path))
#                 else:
#                     dataset.append((img_path, pid_lbl, camid0, clothes_lbl))

#             pid2clothes[pid_lbl, clothes_lbl] = 1

#         num_imgs = len(dataset)
#         return dataset, num_pids, num_imgs, num_clothes, pid2clothes

#     def _process_dir_test(self, query_path, gallery_path):
#         query_img_paths = glob.glob(osp.join(query_path, '*.jpg'))
#         gallery_img_paths = glob.glob(osp.join(gallery_path, '*.jpg'))
#         query_img_paths.sort()
#         gallery_img_paths.sort()
#         pattern = re.compile(r'(\d+)-(\d+)-(\d+)-(\d+)')

#         imgdir2attribute = self._load_meta_map()

#         pid_container = set()
#         clothes_container = set()

#         # collect ids/clothes within selected protocol cams
#         for img_path in query_img_paths:
#             m = pattern.search(img_path)
#             if m is None:
#                 continue
#             pid, camid, clothes, _ = m.groups()
#             pid_int, camid_int = int(pid), int(camid)
#             if not self._cam_allowed(camid_int):
#                 continue
#             clothes_id = pid + clothes
#             pid_container.add(pid_int)
#             clothes_container.add(clothes_id)

#         for img_path in gallery_img_paths:
#             m = pattern.search(img_path)
#             if m is None:
#                 continue
#             pid, camid, clothes, _ = m.groups()
#             pid_int, camid_int = int(pid), int(camid)
#             if not self._cam_allowed(camid_int):
#                 continue
#             clothes_id = pid + clothes
#             pid_container.add(pid_int)
#             clothes_container.add(clothes_id)

#         pid_container = sorted(pid_container)
#         clothes_container = sorted(clothes_container)
#         clothes2label = {cid: label for label, cid in enumerate(clothes_container)}

#         num_pids = len(pid_container)
#         num_clothes = len(clothes_container)

#         query_dataset = []
#         gallery_dataset = []

#         # IMPORTANT: for test, keep pid as original int (same as your LTCC/PRCC style)
#         for img_path in query_img_paths:
#             m = pattern.search(img_path)
#             if m is None:
#                 continue
#             pid, camid, clothes, _ = m.groups()
#             pid_int, camid_int = int(pid), int(camid)
#             if not self._cam_allowed(camid_int):
#                 continue

#             clothes_id = clothes2label[pid + clothes]
#             camid0 = camid_int - 1

#             if self.aux_info:
#                 k_abs = self._normpath(img_path)
#                 aux = imgdir2attribute.get(img_path, imgdir2attribute.get(k_abs))
#                 if aux is None:
#                     aux = [0 for _ in range(self.meta_dims)]
#                 query_dataset.append((img_path, pid_int, camid0, clothes_id, aux))
#             else:
#                 if self.with_mask:
#                     mask_path = osp.join(self.query_mask_dir, osp.basename(img_path)[:-4] + '.npy')
#                     query_dataset.append((img_path, pid_int, camid0, clothes_id, mask_path))
#                 else:
#                     query_dataset.append((img_path, pid_int, camid0, clothes_id))

#         for img_path in gallery_img_paths:
#             m = pattern.search(img_path)
#             if m is None:
#                 continue
#             pid, camid, clothes, _ = m.groups()
#             pid_int, camid_int = int(pid), int(camid)
#             if not self._cam_allowed(camid_int):
#                 continue

#             clothes_id = clothes2label[pid + clothes]
#             camid0 = camid_int - 1

#             if self.aux_info:
#                 k_abs = self._normpath(img_path)
#                 aux = imgdir2attribute.get(img_path, imgdir2attribute.get(k_abs))
#                 if aux is None:
#                     aux = [0 for _ in range(self.meta_dims)]
#                 gallery_dataset.append((img_path, pid_int, camid0, clothes_id, aux))
#             else:
#                 if self.with_mask:
#                     mask_path = osp.join(self.gallery_mask_dir, osp.basename(img_path)[:-4] + '.npy')
#                     gallery_dataset.append((img_path, pid_int, camid0, clothes_id, mask_path))
#                 else:
#                     gallery_dataset.append((img_path, pid_int, camid0, clothes_id))

#         num_imgs_query = len(query_dataset)
#         num_imgs_gallery = len(gallery_dataset)

#         return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes



# vc_clothes.py
import os
import re
import glob
import logging
import numpy as np
import os.path as osp

class VC_Clothes(object):
    """
    VC-Clothes
    Reference: Wang et al. When Person Re-identification Meets Changing Clothes. CVPRW, 2020.
    """
    dataset_dir = 'VC-Clothes'

    def __init__(self, root='data', aux_info=False, meta_dir='PAR_PETA_105.txt', meta_dims=105, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.aux_info = aux_info
        self.meta_dir = meta_dir
        self.meta_dims = meta_dims

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self.mode = kwargs.get('mode', 'all')
        self.with_mask = bool(kwargs.get('with_mask', False))

        self.train_mask_dir = osp.join(self.dataset_dir, 'human_parsing/train')
        self.query_mask_dir = osp.join(self.dataset_dir, 'human_parsing/query')
        self.gallery_mask_dir = osp.join(self.dataset_dir, 'human_parsing/gallery')

        self._check_before_run()

        # 训练集处理
        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.train_dir)
        
        # 测试集处理：默认模式由 self.mode 决定
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
            self._process_dir_test(self.query_dir, self.gallery_dir, mode=self.mode)

        # 为了兼容多模式测试，显式保存不同模式下的数量
        _, _, _, self.num_query_imgs_sc, self.num_gallery_imgs_sc, _ = self._process_dir_test(self.query_dir, self.gallery_dir, mode='sc')
        _, _, _, self.num_query_imgs_cc, self.num_gallery_imgs_cc, _ = self._process_dir_test(self.query_dir, self.gallery_dir, mode='cc')

        logger = logging.getLogger('reid.dataset')
        logger.info("=> VC-Clothes loaded (Mode: {})".format(self.mode))
        
        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.num_query_imgs = num_query_imgs
        self.pid2clothes = pid2clothes

    def _check_before_run(self):
        for d in [self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir]:
            if not osp.exists(d): raise RuntimeError(f"'{d}' is not available")

    @staticmethod
    def _normpath(p: str) -> str:
        return osp.normpath(p)

    def _cam_allowed(self, camid_raw: int, mode: str) -> bool:
        # SC: Cam 2 & 3; CC: Cam 3 & 4
        if mode == 'sc': return camid_raw in [2, 3]
        if mode == 'cc': return camid_raw in [3, 4]
        return True

    def _process_dir_train(self, dir_path):
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')))
        pattern = re.compile(r'(\d+)-(\d+)-(\d+)-(\d+)')
        pid_container, clothes_container = set(), set()
        
        for img_path in img_paths:
            m = pattern.search(img_path)
            if m:
                pid, _, clothes, _ = m.groups()
                pid_container.add(int(pid))
                clothes_container.add(pid + clothes)

        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {cid: label for label, cid in enumerate(clothes_container)}
        
        dataset = []
        pid2clothes = np.zeros((len(pid_container), len(clothes_container)))
        for img_path in img_paths:
            m = pattern.search(img_path)
            if not m: continue
            pid, camid, clothes, _ = m.groups()
            pid_int, camid_int = int(pid), int(camid)
            pid_lbl, clothes_lbl = pid2label[pid_int], clothes2label[pid + clothes]
            dataset.append((img_path, pid_lbl, camid_int-1, clothes_lbl))
            pid2clothes[pid_lbl, clothes_lbl] = 1
            
        return dataset, len(pid_container), len(dataset), len(clothes_container), pid2clothes

    def _process_dir_test(self, query_path, gallery_path, mode='all'):
        pattern = re.compile(r'(\d+)-(\d+)-(\d+)-(\d+)')
        def _collect(path_list, mde):
            data = []
            pids, clothes = set(), set()
            for p in path_list:
                match = pattern.search(p)
                if not match: continue
                pid, camid, clo, _ = match.groups()
                if self._cam_allowed(int(camid), mde):
                    data.append(p)
                    pids.add(int(pid))
                    clothes.add(pid + clo)
            return data, pids, clothes

        q_paths, q_pids, q_clos = _collect(sorted(glob.glob(osp.join(query_path, '*.jpg'))), mode)
        g_paths, g_pids, g_clos = _collect(sorted(glob.glob(osp.join(gallery_path, '*.jpg'))), mode)

        all_pids = sorted(q_pids | g_pids)
        all_clos = sorted(q_clos | g_clos)
        clo2lbl = {c: i for i, c in enumerate(all_clos)}

        def _fill(paths):
            ds = []
            for p in paths:
                m = pattern.search(p)
                pid, camid, clo, _ = m.groups()
                ds.append((p, int(pid), int(camid)-1, clo2lbl[pid + clo]))
            return ds

        return _fill(q_paths), _fill(g_paths), len(all_pids), len(q_paths), len(g_paths), len(all_clos)