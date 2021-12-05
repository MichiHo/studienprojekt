echo ""
echo "# Download labelmefacade"
git clone https://github.com/cvjena/labelmefacade.git

echo ""
echo "# Download cmp_facade"
mkdir cmp_facade
cd cmp_facade
wget -O cmp_base.zip https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip
wget -O cmp_ext.zip https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_extended.zip
unzip -o cmp_base.zip
unzip -o cmp_ext.zip
mkdir all
cp base/* all
cp extended/* all