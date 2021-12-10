echo ""
echo "# Create Python virtual environment and install requirements"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


echo ""
echo "# Clone MMSegmentation"
git clone https://github.com/MichiHo/mmsegmentation.git
cd mmsegmentation
git checkout -b studienprojekt


echo ""
echo "# Download and convert imagenet-pretrained model."
mkdir pretrain
cd pretrain
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
mv swin_base_patch4_window12_384_22k.pth swin_base_patch4_window12_384_22k_oldkeys.pth
python3 ../tools/model_converters/swin2mmseg.py swin_base_patch4_window12_384_22k.pth swin_base_patch4_window12_384_22k_oldkeys.pth swin_base_patch4_window12_384_22k.pth swin_base_patch4_window12_384_22k.pth

deactivate