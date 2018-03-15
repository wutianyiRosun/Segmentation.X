

png_path='/home/wty/AllDataSet/pascal-context/ms/'
save_dir='/home/wty/AllDataSet/pascal-context/ms_colormap/'
load('color60.mat')
img_list = dir( [png_path '*.png']);
for i = 1:length(img_list)
	image_path = [png_path img_list(i).name]
    pngdata=imread(image_path)
    rbgPred=colorEncode(pngdata, color60);
    imwrite(rbgPred, [save_dir  img_list(i).name])
end



