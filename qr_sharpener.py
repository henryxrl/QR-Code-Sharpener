import argparse
from pathlib import Path

import cv2
import numpy as np


class QRSharpener:
    def __init__(self, filepath, qr_dim=0, output_QR_size=0, debug=True):
        self.filepath = filepath
        self.img_orig = cv2.imread(filepath)
        self.img_annotated = self.img_orig.copy()
        self.img_cropped = None
        self.img_cropped_annotated = None
        
        self.auto_detect = True
        self.qr_dim = qr_dim    # QR code dimension; use user input if cannot be determined automatically
        self.img_qr_cv2 = None  # QR code extracted using cv2 method, less robust
        self.img_qr = None  # QR code extracted using QRSharpener
        self.img_qr_size = output_QR_size
        self.qr_tiles = None
        
        self.debug = debug
        if self.debug:
            self.debug_dir = Path('debug')
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def extract_QR(self):
        self.crop_QR()
        self.sharpen_QR()
        if self.debug and self.auto_detect:
            self.save_QR(self.img_qr_cv2, self.img_qr_size, 'debug_cv2_method')
        outpath = self.save_QR(self.img_qr, self.img_qr_size, 'sharpened')
        print(f'QR code extracted to "{outpath}".')

    def sharpen_QR(self):
        src_img = self.img_cropped
        if not self.auto_detect:
            src_img = self.img_orig
        im_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        im_bw_orig = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        
        # Crop out white borders
        gray = 255*(im_bw_orig < 128).astype(np.uint8) # To invert the image
        coords = cv2.findNonZero(gray)
        offset_x, offset_y, crop_w, crop_h = cv2.boundingRect(coords) # Find minimum spanning bounding box
        im_bw = im_bw_orig[offset_y:offset_y+crop_h, offset_x:offset_x+crop_w]
        
        if self.debug:
            self.img_cropped_annotated = cv2.cvtColor(im_bw_orig, cv2.COLOR_GRAY2BGR)
        else:
            self.img_cropped_annotated = src_img.copy()
        self.img_qr = np.zeros((self.qr_dim, self.qr_dim), dtype=np.int32)
        
        img_width = im_bw.shape[1]
        img_height = im_bw.shape[0]
        if self.img_qr_size == 0:
            self.img_qr_size = max(img_width, img_height)
        if self.debug:
            print(f'im_bw.shape: {im_bw.shape}')
        
        wTile = int(np.ceil(img_width / self.qr_dim))
        hTile = int(np.ceil(img_height / self.qr_dim))
        if self.debug:
            print(f'wTile: {wTile}, hTile: {hTile}')

        # Total remainders
        remainderX = self.qr_dim * wTile - img_width
        remainderY = self.qr_dim * hTile - img_height
        if self.debug:
            print(f'remainderX: {remainderX}, remainderY: {remainderY}')

        # Set up remainders per tile
        remaindersX = np.ones((self.qr_dim-1, 1), dtype=np.int32) * int(np.floor(remainderX / (self.qr_dim-1)))
        remaindersY = np.ones((self.qr_dim-1, 1), dtype=np.int32) * int(np.floor(remainderY / (self.qr_dim-1)))
        remaindersX[0:np.remainder(remainderX, self.qr_dim-1)] += 1
        remaindersY[0:np.remainder(remainderY, self.qr_dim-1)] += 1
        remaindersX = remaindersX.squeeze()
        remaindersY = remaindersY.squeeze()
        if self.debug:
            print(f'remaindersX: {remaindersX}')
            print(f'remaindersY: {remaindersY}')
        
        # Initialize array of tile boxes
        self.qr_tiles = np.zeros((self.qr_dim * self.qr_dim), dtype=object)
        count = 0
        k = 0
        y = 0
        for i in range(self.qr_dim):
            x = 0
            for j in range(self.qr_dim):
                self.qr_tiles[k] = (x, y, wTile, hTile, x + wTile//2, y + hTile//2, count)
                
                # Find the dominant color of the tile
                tile = im_bw[y:y+hTile, x:x+wTile]
                center = [hTile//2, wTile//2]       # center of the tile should be enough to determine the color
                self.img_qr[i, j] = tile[center[0], center[1]]
                
                # Annotate the image
                if self.debug:
                    # save individual tiles
                    cv2.imwrite(f'{self.debug_dir}/{Path(self.filepath).stem}_tile_{count}.png', tile)
                    
                    # draw a rectangle around the tile
                    cv2.rectangle(self.img_cropped_annotated, (x + offset_x, y + offset_y), (x + offset_x + wTile, y + offset_y + hTile), (0, 255, 0), 1)
                    
                    # draw the tile number
                    cv2.putText(self.img_cropped_annotated, str(count), (x + offset_x + center[1], y + offset_y + center[0]), cv2.FONT_HERSHEY_SIMPLEX, min(wTile, hTile)/25*0.2, (255, 0, 255), 1, cv2.LINE_AA)
                    
                # draw a red dot onto where we consumed the pixel from
                cv2.circle(self.img_cropped_annotated, (x + offset_x + center[1], y + offset_y + center[0]), round(min(wTile, hTile)/25*2), (0, 0, 255), thickness=-1)
                
                count += 1
                k += 1
                if j < (self.qr_dim-1):
                    x = x + wTile - remaindersX[j]
            
            if i < (self.qr_dim-1):
                y = y + hTile - remaindersY[i]

        if self.debug:
            self.show_image(self.img_cropped_annotated, 'QR tiled')
            self.save_img(self.img_cropped_annotated, 'debug_tiled')

    def crop_QR(self):
        # initialize the cv2 QRCode detector
        detector = cv2.QRCodeDetector()

        # detect and decode
        data, bbox, straight_qrcode = detector.detectAndDecode(self.img_orig)

        # if there is a QR code
        if bbox is not None:
            self.qr_dim = straight_qrcode.shape[0]
            self.img_qr_cv2 = straight_qrcode

            if self.debug:
                print(f'QRCode data: {data}')
                print(f'BoundingBox: {bbox}')
                print(f'straight_qrcode.shape: {straight_qrcode.shape}')

            # draw bounding box
            n_lines = len(bbox[0])
            bbox1 = bbox.astype(int)
            for i in range(n_lines):
                # draw all lines
                point1 = tuple(bbox1[0, [i][0]])
                point2 = tuple(bbox1[0, [(i+1) % n_lines][0]])
                cv2.line(self.img_annotated, point1, point2, color=(255, 255, 0), thickness=2)

            # display the result
            if self.debug:
                self.show_image(self.img_annotated, 'QR code bbox')
                self.save_img(self.img_annotated, 'debug_bbox')

            # warp the image to get a top-down view of the QRCode
            (tl, tr, br, bl) = bbox.squeeze()
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            maxHeight = max(int(heightA), int(heightB))
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(bbox.squeeze(), dst)
            self.img_cropped = cv2.warpPerspective(self.img_orig, M, (maxWidth, maxHeight))

            # display the result
            if self.debug:
                self.show_image(self.img_cropped, 'warped QR code')
                self.save_img(self.img_cropped, 'debug_warped')
            
            self.auto_detect = True
        else:
            print('QR code not detected. Trying the parameters inputted by user...')
            print('[NOTE] Make sure the input image is cropped to the QR code only without any warping, otherwise the program will not work.')
            self.auto_detect = False

    def save_QR(self, img_qr, img_qr_size, filename='refined'):
        img_qr_scaled = cv2.resize(img_qr, (img_qr_size, img_qr_size), interpolation=cv2.INTER_NEAREST)
        return self.save_img(img_qr_scaled, filename)

    def save_img(self, img, filename='refined'):
        outpath = f'{Path(self.filepath).parent / Path(self.filepath).stem}_{filename}.png'
        cv2.imwrite(outpath, img)
        return outpath

    def closest_dividable_by(self, n, m):
        return m * round(n / m)

    def gaussian_2d(self, x_dim, y_dim, sigma=1, mu=0):
        x, y = np.meshgrid(np.linspace(-1, 1, y_dim), np.linspace(-1, 1, x_dim))
        d = np.sqrt(x*x + y*y)
        return np.exp(-((d-mu)**2 / (2.0 * sigma**2)))

    def show_image(self, img, title='image'):
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='QR Code Sharpening')
    parser.add_argument('filepath', type=str, help='Input image file path')
    parser.add_argument('--qr_dim', type=int, default=29, help='The dimension of your QR code. Default is 29 (QR code version 3). Only used when this parameter cannot be detected automatically.')
    parser.add_argument('--size', type=int, default=0, help='Output QR code size. A value of 0 means the size will be automatically determined.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode, which will print and show intermediate results.')
    args = parser.parse_args()

    # Sharpen the QR code
    sharpener = QRSharpener(args.filepath, qr_dim=args.qr_dim, output_QR_size=args.size, debug=args.debug)
    sharpener.extract_QR()
