from modules import *
from utils import *
import warnings
import os.path
import argparse
import matplotlib.pyplot as plt
import time


def parse_input():
    parser = argparse.ArgumentParser()

    # noise parameters
    parser.add_argument('--sigma', type=int, default=15, help='noise sigma: 15, 25, 50')
    parser.add_argument('--seed', type=int, default=8, help='random seed')

    # path
    parser.add_argument('--in_path', type=str, default="C:/Users/takao/Desktop/denoising_raw_data")
    parser.add_argument('--out_path', type=str, default="C:/Users/takao/Desktop/denoising_processed_data/lidia")
    parser.add_argument('--save', action='store_true', help='save output image')

    # memory consumption
    parser.add_argument('--max_chunk', type=int, default=40000)

    # gpu
    parser.add_argument('--cuda', action='store_true', help='use CUDA during inference')

    # additional parameters
    parser.add_argument('--plot', action='store_true', help='plot the processed image')

    opt = parser.parse_args()

    assert 15 == opt.sigma or 25 == opt.sigma or 50 == opt.sigma, "supported sigma values: 15, 25, 50"

    return opt


def denoise_bw_func():
    arch_opt = ArchitectureOptions(rgb=False, small_network=False)
    opt = parse_input()
    # sys.stdout = Logger('output/log.txt')
    pad_offs, _ = calc_padding(arch_opt)
    nl_denoiser = NonLocalDenoiser(pad_offs, arch_opt)
    criterion = nn.MSELoss(reduction='mean')

    state_file_name0 = '../models/model_state_sigma_blind_bw.pt'
    assert os.path.isfile(state_file_name0), "The model path is incorrect"

    if opt.cuda:
        if torch.cuda.is_available():
            nl_denoiser.cuda()
            model_state0 = torch.load(state_file_name0)
        else:
            warnings.warn("CUDA isn't supported")
            model_state0 = torch.load(state_file_name0, map_location=torch.device('cpu'), weights_only=True)
    else:
        model_state0 = torch.load(state_file_name0, map_location=torch.device('cpu'), weights_only=True)

    nl_denoiser.patch_denoise_net.load_state_dict(model_state0['state_dict'])

    filenames = [f for f in os.listdir(opt.in_path) if f.endswith(".png")]
    print("type(nl_denoiser):", type(nl_denoiser))
    for filename in filenames:
        start = time.time()
        test_image_c = load_image_from_file(os.path.join(opt.in_path, filename))
        print("test_image_c.shape:", test_image_c.shape)
        test_image_dn = process_image(nl_denoiser, test_image_c, opt.max_chunk)
        print("test_image_dn.shape:", test_image_dn.shape)
        exit()
        end = time.time()
        test_image_dn = test_image_dn.clamp(-1, 1).cpu()
        psnr_dn = -10 * math.log10(criterion(test_image_dn / 2, test_image_c / 2).item())

        print('Denoising a grayscale image with sigma = {} done. Time: {} seconds, output PSNR = {:.2f}'.format(opt.sigma, end-start, psnr_dn))
        # sys.stdout.flush()

        if opt.plot:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(tensor_to_ndarray_uint8(test_image_c).squeeze(), cmap='gray', vmin=0, vmax=255)
            axs[0].set_title('Original')
            axs[1].imshow(tensor_to_ndarray_uint8(test_image_dn).squeeze(), cmap='gray', vmin=0, vmax=255)
            axs[1].set_title('Denoised, PSNR = {:.2f}'.format(psnr_dn))
            plt.draw()
            plt.pause(1)

        if opt.save:
            savepath = os.path.join(opt.out_path, "processed_"+filename)
            imageio.imwrite(savepath, tensor_to_ndarray_uint8(test_image_dn).squeeze())

        if opt.plot:
            plt.show()


if __name__ == '__main__':
    denoise_bw_func()
