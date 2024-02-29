from PIL import Image

def crop_to_square_and_save(inputpath, outputpath):
    # 打开图片
    with Image.open(inputpath) as img:
        # 获取图片的宽度和高度
        width, height = img.size
        
        # 计算裁剪的新边长，取宽和高中的较小值
        new_edge_length = min(width, height)
        
        # 计算裁剪框的左上角和右下角坐标
        left = (width - new_edge_length)/2
        top = (height - new_edge_length)/2
        right = (width + new_edge_length)/2
        bottom = (height + new_edge_length)/2
        
        # 裁剪图片
        img_cropped = img.crop((left, top, right, bottom))
        
        # 保存裁剪后的图片
        img_cropped.save(outputpath, "PNG")

if __name__=="__main__":
  crop_to_square_and_save("path/to/your/input/image.jpg", "path/to/your/output/image.png")
