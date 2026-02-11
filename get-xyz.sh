mwfn="/home/pfliuiuc/bin/mwfn/Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn_noGUI"
for i in $(seq 1 1 5); do
	mkdir -p REPLICA-$i
	cd REPLICA-$i 
	for j in $(seq 1 1 100); do 
		
		echo -e "xyz\n" | $mwfn /expanse/lustre/projects/uic414/bhanson2/12-6-4-NBFIX/redo-snapshots/REPLICA-$i/MM/$j.pdb 
		
	done
	cd ../
done
