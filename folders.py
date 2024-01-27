import torch
import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
# import csv
# from openpyxl import load_workbook
import pandas as pd
from models_blip.BLIP.blip_pretrain import BLIP_Pretrain
import scipy.io as scio
from sentence_transformers import SentenceTransformer
# from FlagEmbedding import LLMEmbedder


class LIVEFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        refpath = os.path.join(root, 'refimgs')
        refname = getFileName(refpath, '.bmp')

        jp2kroot = os.path.join(root, 'jp2k')
        jp2kname = self.getDistortionTypeFileName(jp2kroot, 227)

        jpegroot = os.path.join(root, 'jpeg')
        jpegname = self.getDistortionTypeFileName(jpegroot, 233)

        wnroot = os.path.join(root, 'wn')
        wnname = self.getDistortionTypeFileName(wnroot, 174)

        gblurroot = os.path.join(root, 'gblur')
        gblurname = self.getDistortionTypeFileName(gblurroot, 174)

        fastfadingroot = os.path.join(root, 'fastfading')
        fastfadingname = self.getDistortionTypeFileName(fastfadingroot, 174)

        imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname

        dmos = scipy.io.loadmat(os.path.join(root, 'dmos_realigned.mat'))
        labels = dmos['dmos_new'].astype(np.float32)

        orgs = dmos['orgs']
        refnames_all = scipy.io.loadmat(os.path.join(root, 'refnames_all.mat'))
        refnames_all = refnames_all['refnames_all']

        sample = []

        for i in range(0, len(index)):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = train_sel * ~orgs.astype(np.bool_)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[1].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((imgpath[item], labels[0][item]))
                # print(self.imgpath[item])
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = '%s%s%s' % ('img', str(index), '.bmp')
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename


class LIVEChallengeFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        imgpath = scipy.io.loadmat(os.path.join(root, 'Data', 'AllImages_release.mat'))
        imgpath = imgpath['AllImages_release']
        imgpath = imgpath[7:1169]
        mos = scipy.io.loadmat(os.path.join(root, 'Data', 'AllMOS_release.mat'))
        labels = mos['AllMOS_release'].astype(np.float32)
        labels = labels[0][7:1169]

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'Images', imgpath[item][0][0]), labels[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class CSIQFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        refpath = os.path.join(root, 'src_imgs')
        refname = getFileName(refpath, '.png')
        txtpath = os.path.join(root, 'csiq_label.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[0]))
            target.append(words[1])
            ref_temp = words[0].split(".")
            refnames_all.append(ref_temp[0] + '.' + ref_temp[-1])

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        sample = []

        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'dst_imgs_all', imgnames[item]), labels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)

        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class Koniq_10kFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'koniq10k_scores_and_distributions.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos = np.array(float(row['MOS_zscore'])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, '1024x768', imgname[item]), mos_all[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class BIDFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        imgname = []
        mos_all = []

        xls_file = os.path.join(root, 'DatabaseGrades.xlsx')
        workbook = load_workbook(xls_file)
        booksheet = workbook.active
        rows = booksheet.rows
        count = 1
        for row in rows:
            count += 1
            img_num = (booksheet.cell(row=count, column=1).value)
            img_name = "DatabaseImage%04d.JPG" % (img_num)
            imgname.append(img_name)
            mos = (booksheet.cell(row=count, column=2).value)
            mos = np.array(mos)
            mos = mos.astype(np.float32)
            mos_all.append(mos)
            if count == 587:
                break

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, imgname[item]), mos_all[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class TID2013Folder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        refpath = os.path.join(root, 'reference_images')
        refname = getTIDFileName(refpath, '.bmp.BMP')
        txtpath = os.path.join(root, 'mos_with_names.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[1]))
            target.append(words[0])
            ref_temp = words[1].split("_")
            refnames_all.append(ref_temp[0][1:])
        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        sample = []
        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'distorted_images', imgnames[item]), labels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class AGIQA_3k_Folder(data.Dataset):
    def __init__(self, root, index, transform=None, patch_num=1):
        data_df = pd.read_csv(os.path.join(root, "data.csv"))
        data_df = data_df[data_df.index.isin(index)]
        self.transform = transform
        self.df = data_df
        self.root = root
        self.patch_num = 1
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config="/home/huangyixin/hw1/homework1/config"
                                                                          "/med_config.json").cuda()

    def __getitem__(self, idx):
        row = self.df.iloc[(idx // self.patch_num)]
        prompt = row["prompt"]
        align_score = row["mos_align"]
        quality_score = row["mos_quality"]
        name = row["name"]
        img_path = os.path.join(self.root, "image", name)
        sample = pil_loader(img_path)
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35,
                                         return_tensors="pt").to("cuda")
        if self.transform is not None:
            sample = self.transform(sample)
        sample_ = torch.unsqueeze(sample, dim=0).cuda()
        image_embeds = self.blip.visual_encoder(sample_)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).cuda()
        text_output = self.blip.text_encoder(text_input.input_ids,
                                             attention_mask=text_input.attention_mask,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=True,
                                             )
        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        # txt_features = txt_features.view(1, 42, 42)
        txt_features = txt_features.view(672, 1, 1)

        return sample, txt_features, align_score

    def __len__(self):
        return len(self.df) * self.patch_num


class AIGCIQA2023_Folder(data.Dataset):
    def __init__(self, root, index, transform=None, patch_num=1):
        self.transform = transform
        self.root = root
        self.patch_num = patch_num
        # quality_socres = scio.loadmat(os.path.join(root, 'DATA/MOS/mosz1.mat'))['MOSz']
        # auth_scores = scio.loadmat(os.path.join(root, 'DATA/MOS/mosz2.mat'))['MOSz']
        align_scores = scio.loadmat(os.path.join(root, 'DATA/MOS/mosz3.mat'))['MOSz']
        # process the dataframe to gather the data by prompts
        prompt_lst = pd.read_excel(os.path.join(root, 'AIGCIQA2023_Prompts.xlsx'), header=None)[2]
        # print(scores)
        img_lst = []
        for i in range(len(prompt_lst)):
            for j in range(6):
                for k in range(4):
                    pic_idx = j * 400 + i * 4 + k
                    if pic_idx not in index:
                        continue
                    img_path = os.path.join(root, "Image/allimg/", f"{pic_idx}.png")
                    # auth = auth_scores[pic_idx][0]
                    # quali = quality_socres[pic_idx][0]
                    align = align_scores[pic_idx][0]
                    img_lst.append((prompt_lst[i], align, img_path))
        self.img_lst = img_lst
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config="/home/huangyixin/hw1/homework1/config"
                                                                          "/med_config.json").cuda()

    def __getitem__(self, idx):
        prompt, align, img_path = self.img_lst[idx]
        sample = pil_loader(img_path)
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35,
                                         return_tensors="pt").to("cuda")
        if self.transform is not None:
            sample = self.transform(sample)
        sample_ = torch.unsqueeze(sample, dim=0).cuda()
        image_embeds = self.blip.visual_encoder(sample_)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).cuda()
        text_output = self.blip.text_encoder(text_input.input_ids,
                                             attention_mask=text_input.attention_mask,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=True,
                                             )
        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        # txt_features = txt_features.view(1, 42, 42)
        txt_features = txt_features.view(672, 1, 1)

        return sample, txt_features, align

    def __len__(self):
        return len(self.img_lst) * self.patch_num


class AGIQA_3k_Text_Folder(data.Dataset):
    def __init__(self, root, index, transform=None, patch_num=1):
        data_df = pd.read_csv(os.path.join(root, "data.csv"))
        data_df = data_df[data_df.index.isin(index)]
        self.transform = transform
        self.df = data_df
        self.root = root
        self.patch_num = patch_num
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config="/home/huangyixin/hw1/homework1/config"
                                                                          "/med_config.json").cuda()
        self.phrase2vec = SentenceTransformer('./keyphrase-mpnet-v1')

    def __getitem__(self, idx):
        row = self.df.iloc[(idx // self.patch_num)]
        prompt = row["prompt"]
        align_score = row["mos_align"]
        quality_score = row["mos_quality"]
        name = row["name"]
        img_path = os.path.join(self.root, "image", name)
        sample = pil_loader(img_path)
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35,
                                         return_tensors="pt").to("cuda")
        if self.transform is not None:
            sample = self.transform(sample)
        sample_ = torch.unsqueeze(sample, dim=0).cuda()
        image_embeds = self.blip.visual_encoder(sample_)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).cuda()
        text_output = self.blip.text_encoder(text_input.input_ids,
                                             attention_mask=text_input.attention_mask,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=True,
                                             )
        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        # txt_features = txt_features.view(1, 42, 42)
        txt_features = txt_features.view(672, 1, 1)
        embeddings = self.phrase2vec.encode(prompt)

        return embeddings, txt_features, align_score

    def __len__(self):
        return len(self.df) * self.patch_num


class AIGCIQA2023_Text_Folder(data.Dataset):
    def __init__(self, root, index, transform=None, patch_num=1):
        self.transform = transform
        self.root = root
        self.patch_num = patch_num
        align_scores = scio.loadmat(os.path.join(root, 'DATA/MOS/mosz3.mat'))['MOSz']
        prompt_lst = pd.read_excel(os.path.join(root, 'AIGCIQA2023_Prompts.xlsx'), header=None)[2]
        # print(scores)
        img_lst = []
        for i in range(len(prompt_lst)):
            for j in range(6):
                for k in range(4):
                    if (j * 4 + k) not in index:
                        continue
                    pic_idx = j * 400 + i * 4 + k
                    img_path = os.path.join(root, "Image/allimg/", f"{pic_idx}.png")
                    align = align_scores[pic_idx][0]
                    img_lst.append((prompt_lst[i], align, img_path))
        self.img_lst = img_lst
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config="/home/huangyixin/hw1/homework1/config"
                                                                          "/med_config.json").cuda()
        self.phrase2vec = SentenceTransformer('./keyphrase-mpnet-v1')
        # self.model = LLMEmbedder('./llm-embedder', use_fp16=False)

    def __getitem__(self, idx):
        prompt, align, img_path = self.img_lst[idx]
        sample = pil_loader(img_path)
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35,
                                         return_tensors="pt").to("cuda")
        if self.transform is not None:
            sample = self.transform(sample)
        sample_ = torch.unsqueeze(sample, dim=0).cuda()
        image_embeds = self.blip.visual_encoder(sample_)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).cuda()
        text_output = self.blip.text_encoder(text_input.input_ids,
                                             attention_mask=text_input.attention_mask,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=True,
                                             )
        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        # txt_features = txt_features.view(1, 42, 42)
        txt_features = txt_features.view(672, 1, 1)

        embeddings = self.phrase2vec.encode(prompt)
        # Encode for a specific task (qa, icl, chat, lrlm, tool, convsearch)
        # task = "qa"
        # embeddings = self.model.encode_queries(prompt, task=task)
        # print(embeddings.shape)

        return embeddings, txt_features, align

    def __len__(self):
        return len(self.img_lst) * self.patch_num


class AGIQA_3k_All_Folder(data.Dataset):
    def __init__(self, root, index, transform=None, patch_num=1):
        data_df = pd.read_csv(os.path.join(root, "data.csv"))
        data_df = data_df[data_df.index.isin(index)]
        self.transform = transform
        self.df = data_df
        self.root = root
        self.patch_num = patch_num
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config="/home/huangyixin/hw1/homework1/config"
                                                                          "/med_config.json").cuda()

    def __getitem__(self, idx):
        row = self.df.iloc[(idx // self.patch_num)]
        prompt = row["prompt"]
        align_score = row["mos_align"]
        quality_score = row["mos_quality"]
        name = row["name"]
        img_path = os.path.join(self.root, "image", name)
        sample = pil_loader(img_path)
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35,
                                         return_tensors="pt").to("cuda")
        if self.transform is not None:
            sample = self.transform(sample)
        sample_ = torch.unsqueeze(sample, dim=0).cuda()
        image_embeds = self.blip.visual_encoder(sample_)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).cuda()
        text_output = self.blip.text_encoder(text_input.input_ids,
                                             attention_mask=text_input.attention_mask,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=True,
                                             )
        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        # txt_features = txt_features.view(1, 42, 42)
        txt_features = txt_features.view(672, 1, 1)

        return sample, txt_features, align_score, quality_score

    def __len__(self):
        return len(self.df) * self.patch_num


class AIGCIQA2023_All_Folder(data.Dataset):
    def __init__(self, root, index, transform=None, patch_num=1):
        self.transform = transform
        self.root = root
        self.patch_num = patch_num
        quality_socres = scio.loadmat(os.path.join(root, 'DATA/MOS/mosz1.mat'))['MOSz']
        auth_scores = scio.loadmat(os.path.join(root, 'DATA/MOS/mosz2.mat'))['MOSz']
        align_scores = scio.loadmat(os.path.join(root, 'DATA/MOS/mosz3.mat'))['MOSz']
        # process the dataframe to gather the data by prompts
        prompt_lst = pd.read_excel(os.path.join(root, 'AIGCIQA2023_Prompts.xlsx'), header=None)[2]
        # print(scores)
        img_lst = []
        for i in range(len(prompt_lst)):
            for j in range(6):
                for k in range(4):
                    pic_idx = j * 400 + i * 4 + k
                    if pic_idx not in index:
                        continue
                    img_path = os.path.join(root, "Image/allimg/", f"{pic_idx}.png")
                    align = align_scores[pic_idx][0]
                    quality = quality_socres[pic_idx][0]
                    auth = auth_scores[pic_idx][0]
                    img_lst.append((prompt_lst[i], quality, auth, align, img_path))
        self.img_lst = img_lst
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config="/home/huangyixin/hw1/homework1/config"
                                                                          "/med_config.json").cuda()

    def __getitem__(self, idx):
        prompt, quality, auth, align, img_path = self.img_lst[idx]
        sample = pil_loader(img_path)
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35,
                                         return_tensors="pt").to("cuda")
        if self.transform is not None:
            sample = self.transform(sample)
        sample_ = torch.unsqueeze(sample, dim=0).cuda()
        image_embeds = self.blip.visual_encoder(sample_)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).cuda()
        text_output = self.blip.text_encoder(text_input.input_ids,
                                             attention_mask=text_input.attention_mask,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=True,
                                             )
        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        # txt_features = txt_features.view(1, 42, 42)
        txt_features = txt_features.view(672, 1, 1)

        return sample, txt_features, quality, auth, align

    def __len__(self):
        return len(self.img_lst) * self.patch_num


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
