###################################################################################################
# Dataset reader for the xView2 dataset 
# Description of dataset: https://xview2.org/dataset
# Author: Hans
# Creating date: 11/04/2019
###################################################################################################
from torch.utils.data.dataset import Dataset

class xView2Dataset(Dataset):
    def __init__(self, ann_file, root, isTrain=True):
        self.root = root
        self.load_size = (286, 286)
        self.crop_size = (256, 256)
        self.isTrain = isTrain
        self.get_files()
    
    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.frame1_files)

    def __getitem__(self, index):
        frame1_file = self.frame1_files[index]
        frame2_file = self.frame2_files[index]
        label_file = self.label_files[index]
        
        frame1_image = Image.open(frame1_file).convert('RGB')
        frame2_image = Image.open(frame2_file).convert('RGB')
        label_image = Image.open(label_file).convert('L')
        label_image = self.process_mask(label_image)

        image1, image2, label = self.transforms(frame1_image, frame2_image, label_image)
        image = torch.cat([image1, image2], dim=0)
        return image, label, index

    def get_img_info(self, idx):
        return {"height": self.crop_size[0], "width": self.crop_size[1]}
    
    def process_mask(self, image):
        """Process the mask image (filter the labels)
        Params: 
            image: <PIL.Image> change mask image 
        Return: 
            mask: <PIL.Image> object of binary mask 
        """
        image = np.array(image)
        image[image == 5] = 1   # set un-classified to undestroyed
        return Image.fromarray(image)
    
    def get_files(self):
        """Read target, moving and mask file paths.
        """
        def _get_files_by_names(files, name_set, postfix):
            ret = []
            for f in files: 
                name = osp.basename(f).split("_%s" % postfix)[0]
                if name in name_set:
                    ret.append(f)
            return ret

        frame1_files  = sorted(glob.glob(osp.join(self.root, 'images', "*_pre_disaster*")))
        frame2_files  = sorted(glob.glob(osp.join(self.root, "images", "*_post_disaster*")))
        label_files  = sorted(glob.glob(osp.join(self.root, "masks", "*_change*")))
        assert len(frame1_files) == len(frame2_files) == len(label_files), \
               "%d, %d, %d" % (len(frame1_files), len(frame2_files), len(label_files))

        file_names = [osp.basename(f).split("_pre")[0] for f in frame1_files]
        file_names = sorted(list(set(file_names)))
        if self.isTrain:
            name_set = train_test_split(file_names, train_size=0.8, random_state=0)[0]
        else: 
            name_set = train_test_split(file_names, train_size=0.8, random_state=0)[1]
        self.frame1_files = _get_files_by_names(frame1_files, name_set, 'pre')
        self.frame2_files = _get_files_by_names(frame2_files, name_set, 'post')
        self.label_files = _get_files_by_names(label_files, name_set, 'change')

    def transforms(self, frame1, frame2, mask):
        if self.isTrain: 
            # Resize
            resize_image = transforms.Resize(size=self.load_size, interpolation=Image.BICUBIC)
            resize_mask = transforms.Resize(size=self.load_size, interpolation=Image.NEAREST)
            frame1 = resize_image(frame1)
            frame2 = resize_image(frame2)
            mask = resize_mask(mask)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(frame1, self.crop_size)
            frame1 = TF.crop(frame1, i, j, h, w)
            i, j, h, w = transforms.RandomCrop.get_params(frame2, self.crop_size)
            frame2 = TF.crop(frame2, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                frame1 = TF.hflip(frame1)
                frame2 = TF.hflip(frame2)
                mask = TF.hflip(mask)

            # Random vertical flipping
            if random.random() > 0.5:
                frame1 = TF.vflip(frame1)
                frame2 = TF.vflip(frame2)
                mask = TF.vflip(mask)
            
            # Color jittering 
            color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
            frame1 = color_jitter(frame1)
            frame2 = color_jitter(frame2)
        else: 
            # Resize
            resize_image = transforms.Resize(size=self.crop_size, interpolation=Image.BICUBIC)
            resize_mask = transforms.Resize(size=self.crop_size, interpolation=Image.NEAREST)
            frame1 = resize_image(frame1)
            frame2 = resize_image(frame2)
            mask = resize_mask(mask)

        # Transform to tensor
        frame1 = TF.to_tensor(frame1)
        frame2 = TF.to_tensor(frame2)
        mask =  torch.from_numpy(np.array(mask)).unsqueeze(0)
        # Nominalize tensor
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        frame1 = normalize(frame1)
        frame2 = normalize(frame2)
        return frame1, frame2, mask