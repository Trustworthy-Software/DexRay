apk=$(pwd)/$1
path=$(pwd)/$2
basename=$(basename $apk .apk)
output=$path/$basename.apk
workingdir=/tmp/

. $(pwd)/set_global_vars.sh

source $(pwd)/Obfuscapk/venv/bin/activate
cd Obfuscapk/src

python3 -m obfuscapk.cli -i -o ClassRename -o FieldRename -o MethodRename -o ConstStringEncryption -o MethodOverload -o CallIndirection -o Reflection -o Rebuild -o NewSignature -o NewAlignment $apk -d $output -w $workingdir
