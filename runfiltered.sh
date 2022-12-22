python -u ./occ-tests-binary-fdr.py | tee log.txt
python -u ./occ-tests-simple-fdr.py | tee -a log.txt
tar -czf results.tar.gz results*/ | tee -a log.txt
echo "Done" | tee -a log.txt
