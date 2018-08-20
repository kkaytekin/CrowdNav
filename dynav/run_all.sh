info="_no_timestep_simplified_state"

cp scripts/cadrl.sh scripts/copy/cadrl.sh
cp scripts/cadrl_lstm.sh scripts/copy/cadrl_lstm.sh
cp scripts/srl.sh scripts/copy/srl.sh
cp scripts/sarl.sh scripts/copy/sarl.sh

sed -i '$ s/$/'"$info"'/' scripts/copy/cadrl.sh
sed -i '$ s/$/'"$info"'/' scripts/copy/cadrl_lstm.sh
sed -i '$ s/$/'"$info"'/' scripts/copy/srl.sh
sed -i '$ s/$/'"$info"'/' scripts/copy/sarl.sh

sbatch scripts/copy/cadrl.sh
sbatch scripts/copy/cadrl_lstm.sh
sbatch scripts/copy/srl.sh
sbatch scripts/copy/sarl.sh
