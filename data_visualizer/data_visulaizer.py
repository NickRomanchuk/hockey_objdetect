
class DataVisualizer():
    def __init__(self, path):
        data_path = path

    def save_bbox(self, folder_path, object=None):
        i = 0
        for _, player in tracks['players'][0].items():
            i += 1
            bbox = player['bbox']
            frame = video_frames[0]

            # crop bbox from frame
            cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

            # save the cropped image
            cv2.imwrite(f'output_videos/trimmed_boxes/cropped_img_{i}.jpg', cropped_image)

            corpus_dir = Path('data')
files = list(corpus_dir.glob(pattern='*.jpg'))
files
for f in files:
    image = Image.open(f)
    image = cv2.imread(str(f))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    ret, thresh1 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU |
                                          cv2.THRESH_BINARY_INV)
    cv2.imwrite('threshold_image.jpg',thresh1)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,30))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 6)
    cv2.imwrite('dilation_image.jpg',dilation)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    im2 = img.copy()
crop_number=1 