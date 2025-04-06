import lpips
from PIL import Image
import os
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
transform2=transforms.Compose([transforms.ToTensor()])

loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

if __name__ == "__main__":
    
    gt_path = "./dataset/Rain12/target/"
    im_path = "./dataset/Rain12/R2A_retrain/"
    
    # gt_path = "./dataset/DDN_SIRR_syn/target/"
    # im_path = "./result/DDN_SIRR_syn/n2v/"
    
    
    # gt_path = "./dataset/Rain12/target/"
    # im_path = "./result/Rain12/mm_bsn/"
    
    # gt_path = "./dataset/Rain100L_train/target/"
    # im_path = "./result/Rain100L_train/average_rlcn_w_NSS/"
    
    # gt_path = "./dataset/Rain12/target/"
    # im_path = "./dataset/Rain12/my/result_1/"

    # gt_path = rf"A:\SSL_Derain_TMM\dataset\GT-Rain\target_new/"
    # im_path = rf"A:\SSL_Derain_TMM\dataset\GT-Rain\result_mm_bsn/"

    #im_path = './out(convolution,^6,ks=3)/'
    print(im_path)
    gt_folder = os.listdir(gt_path)
    im_folder = os.listdir(im_path)
    
    sum =0
    num =0
    
    for i in range(len(gt_folder)):
        gt=Image.open(gt_path+gt_folder[i]).convert('RGB')
        tensor_gt=transform2(gt)
        im=Image.open(os.path.join(im_path,gt_folder[i])).convert('RGB')
        tensor_im=transform2(im)

        sum += loss_fn_alex(tensor_gt, tensor_im).item()
        num += 1
    
    # for i in range(len(im_folder)):
    #     gt=Image.open(gt_path+gt_folder[i]).convert('RGB')
    #     tensor_gt=transform2(gt)
    #     im=Image.open(os.path.join(im_path, gt_folder[i][:-4]+".jpg")).convert('RGB')
    #     tensor_im=transform2(im)

    #     sum += loss_fn_alex(tensor_gt, tensor_im).item()
    #     num += 1
    
    
    print(sum/num)