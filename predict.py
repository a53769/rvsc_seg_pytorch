import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image



from unet import UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks
from utils import plot_img_and_mask, dataset, patient

from torchvision import transforms


def load_patient_images(path, normalize=True):
    p = patient.PatientData(path)

    # reshape to account for channel dimension
    images = np.asarray(p.images, dtype='float64')[:,None,:,:]

    # maybe normalize images
    if normalize:
        dataset.normalize(images, axis=(1,2))

    return images, p.index, p.labeled, p.rotated, p.endocardium_masks

'''暂时不需要输出边界点坐标
def get_contours(mask):
    mask_image = np.where(mask > 0.5, 255, 0).astype('uint8')
    im2, coords, hierarchy = cv2.findContours(mask_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not coords:
        print("No contour detected.")
        coords = np.ones((1, 1, 1, 2), dtype='int')
    if len(coords) > 1:
        print("Multiple contours detected.")
        lengths = [len(coord) for coord in coords]
        coords = [coords[np.argmax(lengths)]]

    coords = np.squeeze(coords[0], axis=(1,))
    coords = np.append(coords, coords[:1], axis=0)

    return coords
'''

def predict_img(net, full_img, scale_factor=0.5, out_threshold=0.5, use_dense_crf=True, use_gpu=False):

    net.eval()
    img_height = full_img.size[1]
    img_width = full_img.size[0]

    img = resize_and_crop(full_img, scale=scale_factor)
    img = normalize(img)

    left_square, right_square = split_img_into_squares(img)

    left_square = hwc_to_chw(left_square)
    right_square = hwc_to_chw(right_square)

    X_left = torch.from_numpy(left_square).unsqueeze(0)
    X_right = torch.from_numpy(right_square).unsqueeze(0)

    if use_gpu:
        X_left = X_left.cuda()
        X_right = X_right.cuda()

    with torch.no_grad():
        output_left = net(X_left)
        output_right = net(X_right)

        left_probs = output_left.squeeze(0)
        right_probs = output_right.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )

        left_probs = tf(left_probs.cpu())
        right_probs = tf(right_probs.cpu())

        left_mask_np = left_probs.squeeze().cpu().numpy()
        right_mask_np = right_probs.squeeze().cpu().numpy()

    full_mask = merge_masks(left_mask_np, right_mask_np, img_width)

    # if use_dense_crf:
    #     full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='CP301.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images',
                        default=['E:\workspace\cardiac-segmentation\\test-assets\patient09'])

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=True)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=True)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=True)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":

    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=1, n_classes=1)

    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))

        images, patient_number, frame_indices, rotated, true_masks = load_patient_images(fn)

        # if img.size[0] < img.size[1]:
        #     print("Error: image height larger than the width")
        predictions = []
        for i, image in enumerate(images):
            image = torch.FloatTensor(image)
            image = image.cuda()
            mask_pred = net(image[None, :, :, :])  # feed one at a time
            predictions.append((image[:, :, 0], mask_pred[0, :, :, 1]))
            # mask = predict_img(net=net,
            #                    full_img=img,
            #                    scale_factor=args.scale,
            #                    out_threshold=args.mask_threshold,
            #                    use_dense_crf=not args.no_crf,
            #                    use_gpu=not args.cpu)


            if args.viz:
                print("Visualizing results for image {}, close to continue ...".format(fn))
                image = image[0]
                mask_pred = mask_pred[0][0]
                plot_img_and_mask(image, mask_pred, true_masks[i])

